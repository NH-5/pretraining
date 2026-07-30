"""Ex1: a small char-level decoder-only GPT with three deliberate TODOs."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from utils import (
    CharTokenizer,
    autocast_context,
    get_batch,
    prepare_tiny_shakespeare,
    seed_everything,
    select_device,
    split_token_stream,
)


DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "input.txt"


@dataclass(frozen=True)
class TrainConfig:
    """Small defaults intended for minutes, not hours."""

    block_size: int = 128
    batch_size: int = 32
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    learning_rate: float = 3e-4
    max_steps: int = 3_000
    log_interval: int = 100
    sample_tokens: int = 300
    seed: int = 42


def build_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    """Return a boolean mask broadcastable to attention scores [B, H, T, T]."""
    # TODO(你)[EX01_CAUSAL_MASK]: 见指南 §3.1、§6.3。
    #   方向:第 t 行只能保留位置 0..t，未来位置必须被屏蔽。
    #   结构:返回 bool 张量；True 表示可见，形状可为 [T, T]。
    #   完成标准:对 T=4 画出矩阵，并断言任意 j>i 的元素均为 False。

    C = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
    )

    return C


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention; masking itself is the learner's task."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_size = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_head,
                self.head_dim,
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        visible = build_causal_mask(sequence_length, hidden.device)
        scores = scores.masked_fill(~visible, float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_size,
        )
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    """The Ex1 baseline intentionally uses a plain GELU FFN."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


class Block(nn.Module):
    """Pre-Norm decoder block, matching guide §3.3."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class CharGPT(nn.Module):
    """A minimal decoder-only language model with tied embeddings."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.language_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        self.language_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Keep initial logits unsaturated for stable next-token learning."""
        # 见指南 §3.3、§7.2：weight tying 会让 embedding 同时充当
        # unembedding，因此必须使用小尺度初始化，避免初始时极端偏向复制输入。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size {self.block_size}."
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.language_head(self.final_norm(hidden))


def forward_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross-entropy over every supervised position."""
    # TODO(你)[EX01_TOKEN_LOSS]: 见指南 §6.1、§6.3。
    #   方向:这是 B*T 个 V 分类，不是每句只算一个 loss。
    #   结构:logits=[B,T,V]，targets=[B,T]；先对齐成损失函数需要的形状。
    #   完成标准:标量 loss 可反传，打乱 targets 后 loss 应明显变差。

    V = logits.shape[2]

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1)
    )

    return loss


@torch.no_grad()
def generate(
    model: CharGPT,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.8,
) -> torch.Tensor:
    """Autoregressively append tokens to a prompt."""
    # TODO(你)[EX01_AUTOREGRESSIVE_GENERATION]: 见指南 §3.1、§6.3。
    #   方向:训练能并行算 T 个位置；生成必须一次追加一个 token。
    #   结构:每步裁到 block_size，取最后位置 logits，采样后拼回序列。
    #   完成标准:输出长度=输入长度+max_new_tokens，且前缀保持不变。

    model.eval()

    for _ in range(max_new_tokens):
        context = prompt[:, - model.block_size:]
        logits = model(context)

        logits = logits[:, -1,:]
        logits = logits / temperature

        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        prompt = torch.cat(
            [prompt, next_token],
            dim=1
        )

    return prompt


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _check_causal_mask() -> str:
    """Validate shape, dtype, diagonal visibility, and absence of future leakage."""
    mask = build_causal_mask(4, torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    if mask.shape != (4, 4):
        raise AssertionError(f"expected shape (4, 4), got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise AssertionError(f"expected torch.bool, got {mask.dtype}")
    if mask.device.type != "cpu":
        raise AssertionError(f"expected requested cpu device, got {mask.device}")
    if not torch.equal(mask, expected):
        raise AssertionError(f"unexpected T=4 mask:\n{mask.to(torch.int8)}")
    if torch.triu(mask, diagonal=1).any():
        raise AssertionError("future leakage detected above the diagonal")
    return "T=4 mask:\n" + str(mask.to(torch.int8))


def _check_token_loss() -> str:
    """Check that every token position contributes to one differentiable scalar."""
    logits = torch.tensor(
        [
            [
                [8.0, -8.0, -8.0],
                [-8.0, 8.0, -8.0],
                [-8.0, -8.0, 8.0],
            ]
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 1, 2]])
    loss = forward_loss(logits, targets)
    if loss.ndim != 0:
        raise AssertionError(f"loss must be scalar, got shape {tuple(loss.shape)}")
    if not torch.isfinite(loss):
        raise AssertionError(f"loss must be finite, got {loss.item()}")
    if loss.item() >= 1e-4:
        raise AssertionError(f"near-perfect predictions should have tiny loss, got {loss.item()}")

    loss.backward()
    if logits.grad is None:
        raise AssertionError("loss.backward() produced no logits gradient")
    positions_with_gradient = logits.grad.abs().sum(dim=-1) > 0
    if not positions_with_gradient.all():
        raise AssertionError(
            "every token position must contribute gradient, got "
            f"{positions_with_gradient.tolist()}"
        )

    one_position_wrong = torch.tensor([[1, 1, 2]])
    wrong_loss = forward_loss(logits.detach(), one_position_wrong)
    if wrong_loss.item() <= 1.0:
        raise AssertionError(
            "changing an earlier target barely changed loss; "
            "you may only be supervising the final position"
        )
    return f"perfect_loss={loss.item():.6g}, one_wrong_loss={wrong_loss.item():.6g}"


class _DeterministicNextTokenModel(nn.Module):
    """A test double whose next token is always (current token + 1) mod V."""

    block_size = 4
    vocab_size = 5

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[1] > self.block_size:
            raise AssertionError(
                "generate must crop its context to model.block_size before forward"
            )
        batch_size, sequence_length = token_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            fill_value=-100.0,
            device=token_ids.device,
        )
        next_ids = ((token_ids + 1) % self.vocab_size).unsqueeze(-1)
        return logits.scatter(dim=-1, index=next_ids, value=100.0)


class _TemperatureProbeModel(nn.Module):
    """A test double with a controllable two-token sampling distribution."""

    block_size = 4
    vocab_size = 2

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape
        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=token_ids.device,
        )
        logits[:, :, 0] = 2.0
        return logits


def _check_generation() -> str:
    """Validate prefix preservation, context cropping, and one-token-at-a-time append."""
    model = _DeterministicNextTokenModel()
    prompt = torch.tensor([[0, 1]], dtype=torch.long)
    torch.manual_seed(0)
    generated = generate(
        model,  # type: ignore[arg-type]
        prompt,
        max_new_tokens=5,
        temperature=1.0,
    )
    expected = torch.tensor([[0, 1, 2, 3, 4, 0, 1]])
    if generated.shape != expected.shape:
        raise AssertionError(
            f"expected output shape {tuple(expected.shape)}, got {tuple(generated.shape)}"
        )
    if not torch.equal(generated, expected):
        raise AssertionError(
            f"expected sequence {expected.tolist()}, got {generated.tolist()}"
        )
    if not torch.equal(generated[:, : prompt.shape[1]], prompt):
        raise AssertionError("generation changed the original prompt")

    probe_model = _TemperatureProbeModel()
    repeated_prompt = torch.zeros(256, 1, dtype=torch.long)
    torch.manual_seed(123)
    hot = generate(
        probe_model,  # type: ignore[arg-type]
        repeated_prompt,
        max_new_tokens=1,
        temperature=100.0,
    )
    torch.manual_seed(123)
    cold = generate(
        probe_model,  # type: ignore[arg-type]
        repeated_prompt,
        max_new_tokens=1,
        temperature=0.1,
    )
    hot_alternatives = int((hot[:, -1] == 1).sum().item())
    cold_alternatives = int((cold[:, -1] == 1).sum().item())
    if not 80 <= hot_alternatives <= 176:
        raise AssertionError(
            "high temperature should retain sampling diversity; "
            f"got token-1 count {hot_alternatives}/256 "
            "(argmax or ignored temperature is likely)"
        )
    if cold_alternatives > 5:
        raise AssertionError(
            "low temperature should concentrate samples on the top token; "
            f"got token-1 count {cold_alternatives}/256"
        )
    return (
        f"generated={generated.tolist()}; "
        f"temperature probe hot/cold token-1={hot_alternatives}/{cold_alternatives}"
    )


def _run_learning_target_check(
    todo_id: str,
    check: Callable[[], str],
) -> tuple[str, str]:
    """Return PASS/PENDING/FAIL without confusing an untouched TODO with a bug."""
    try:
        detail = check()
    except NotImplementedError:
        return "PENDING", "implementation still raises NotImplementedError"
    except Exception as error:
        return "FAIL", f"{type(error).__name__}: {error}"
    return "PASS", str(detail)


def check_scaffold() -> None:
    """Validate scaffold wiring and automatically judge completed learning targets."""
    sample = "To be, or not to be."
    tokenizer = CharTokenizer.from_text(sample)
    encoded = tokenizer.encode(sample)
    if tokenizer.decode(encoded) != sample:
        raise RuntimeError("Tokenizer round trip failed.")
    config = replace(
        TrainConfig(),
        vocab_size=tokenizer.vocab_size,
        block_size=8,
        n_layer=2,
        n_embd=64,
    )
    torch.manual_seed(config.seed)
    model = CharGPT(config)
    generator = torch.Generator().manual_seed(config.seed)
    probe_inputs = torch.randint(
        0,
        config.vocab_size,
        (4, config.block_size),
        generator=generator,
    )
    probe_targets = torch.randint(
        0,
        config.vocab_size,
        (4, config.block_size),
        generator=generator,
    )
    model.eval()
    with torch.no_grad():
        probe_logits = model(probe_inputs)
        initial_loss = F.cross_entropy(
            probe_logits.reshape(-1, config.vocab_size),
            probe_targets.reshape(-1),
        ).item()
    expected_initial_loss = math.log(config.vocab_size)
    if not math.isfinite(initial_loss) or abs(
        initial_loss - expected_initial_loss
    ) > 0.5:
        raise RuntimeError(
            "Model initialization produced saturated logits: "
            f"loss={initial_loss:.4f}, expected near ln(V)={expected_initial_loss:.4f}."
        )
    print("Ex1 scaffold: PASS")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(
        f"Initial random-token loss: {initial_loss:.4f} "
        f"(ln(V)={expected_initial_loss:.4f})"
    )
    print("\nLearning-target checks:")
    checks = (
        ("EX01_CAUSAL_MASK", _check_causal_mask),
        ("EX01_TOKEN_LOSS", _check_token_loss),
        ("EX01_AUTOREGRESSIVE_GENERATION", _check_generation),
    )
    counts = {"PASS": 0, "PENDING": 0, "FAIL": 0}
    for todo_id, check in checks:
        status, detail = _run_learning_target_check(todo_id, check)
        counts[status] += 1
        print(f"[{status}] {todo_id}")
        for line in detail.splitlines():
            print(f"  {line}")
    print(
        "\nSummary: "
        f"{counts['PASS']} passed, {counts['PENDING']} pending, {counts['FAIL']} failed"
    )
    if counts["FAIL"]:
        raise RuntimeError("One or more learning-target checks failed.")


def train(config: TrainConfig, data_path: Path) -> None:
    """Train only after the three learning-target TODOs have been filled."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Run the 'prepare' subcommand first."
        )
    seed_everything(config.seed)
    device = select_device()
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    config = replace(config, vocab_size=tokenizer.vocab_size)
    train_tokens, _ = split_token_stream(tokenizer.encode(text))
    generator = torch.Generator().manual_seed(config.seed)

    model = CharGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print(
        f"device={device.type} params={count_parameters(model):,} "
        f"tokens={len(train_tokens):,}"
    )

    model.train()
    for step in range(1, config.max_steps + 1):
        inputs, targets = get_batch(
            train_tokens,
            batch_size=config.batch_size,
            block_size=config.block_size,
            device=device,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss = forward_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            print(f"step={step:04d} train_loss={loss.item():.4f}")

    model.eval()
    first_character = text[0]
    prompt = torch.tensor(
        [tokenizer.encode(first_character)],
        dtype=torch.long,
        device=device,
    )
    sampled = generate(
        model,
        prompt,
        max_new_tokens=config.sample_tokens,
    )
    print("--- sample ---")
    print(tokenizer.decode(sampled[0].tolist()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Validate non-TODO scaffold wiring.")

    prepare_parser = subparsers.add_parser("prepare", help="Download tiny Shakespeare.")
    prepare_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)

    train_parser = subparsers.add_parser("train", help="Run training after TODOs are done.")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    train_parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    train_parser.add_argument("--block-size", type=int, default=TrainConfig.block_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        check_scaffold()
    elif args.command == "prepare":
        character_count = prepare_tiny_shakespeare(args.data)
        print(f"Prepared {character_count:,} characters at {args.data}")
    elif args.command == "train":
        config = replace(
            TrainConfig(),
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )
        train(config, args.data)


if __name__ == "__main__":
    main()
