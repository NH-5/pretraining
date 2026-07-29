"""Ex1: a small char-level decoder-only GPT with three deliberate TODOs."""

from __future__ import annotations

import argparse
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


TODO_IDS = (
    "EX01_CAUSAL_MASK",
    "EX01_TOKEN_LOSS",
    "EX01_AUTOREGRESSIVE_GENERATION",
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
    raise NotImplementedError("TODO[EX01_CAUSAL_MASK]")


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
        self.language_head.weight = self.token_embedding.weight

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
    raise NotImplementedError("TODO[EX01_TOKEN_LOSS]")


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
    raise NotImplementedError("TODO[EX01_AUTOREGRESSIVE_GENERATION]")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def check_scaffold() -> None:
    """Validate all non-TODO wiring without revealing an implementation."""
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
    model = CharGPT(config)
    print("Ex1 scaffold: PASS")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size}")
    print(f"Model parameters: {count_parameters(model):,}")
    print("Unresolved learning targets:")
    for todo_id in TODO_IDS:
        print(f"  - {todo_id}")


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
        if step == 1 or step % config.log_interval == 0:
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
