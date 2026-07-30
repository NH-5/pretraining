"""Ex9: compare one modern decoder component against a fixed baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from components import GroupedQueryAttention, SwiGLU, apply_rope
from exercises.checking import CheckCase, run_checks
from model import ModelConfig, TinyGPT, causal_attention, next_token_loss


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch_size: int = 16
    learning_rate: float = 3e-4
    seed: int = 29


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch(
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(train_config.seed + step)
    token_stream = torch.empty(
        train_config.batch_size,
        model_config.block_size + 1,
        dtype=torch.long,
    )
    token_stream[:, :2] = torch.randint(
        0,
        model_config.vocab_size,
        (train_config.batch_size, 2),
        generator=generator,
    )
    # The next token depends on two preceding tokens, so attention is genuinely useful.
    for position in range(2, model_config.block_size + 1):
        token_stream[:, position] = (
            token_stream[:, position - 1] + token_stream[:, position - 2]
        ) % model_config.vocab_size
    inputs = token_stream[:, :-1]
    targets = token_stream[:, 1:]
    return inputs.to(device), targets.to(device)


def train_variant(
    component: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> list[float]:
    torch.manual_seed(train_config.seed)
    device = choose_device()
    model = TinyGPT(model_config, component).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
    )
    losses: list[float] = []
    for step in range(train_config.steps):
        inputs, targets = make_batch(
            model_config,
            train_config,
            step=step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _check_model_scaffold() -> str:
    config = ModelConfig()
    counts: dict[str, int] = {}
    for component in ("baseline", "rope", "swiglu", "gqa"):
        model = TinyGPT(config, component)
        counts[component] = sum(parameter.numel() for parameter in model.parameters())
    if any(count <= 0 for count in counts.values()):
        raise RuntimeError("A component model has no parameters.")
    return "; ".join(f"{name}={count:,}" for name, count in counts.items())


def _check_causal_attention() -> str:
    torch.manual_seed(901)
    query = torch.randn(1, 2, 4, 3)
    key = torch.randn(1, 2, 4, 3)
    value = torch.randn(1, 2, 4, 3)
    original = causal_attention(
        query,
        key,
        value,
        dropout=0.0,
        training=False,
    )
    if original.shape != query.shape:
        raise RuntimeError(
            f"Causal attention output shape {original.shape} != {query.shape}."
        )
    changed_value = value.clone()
    changed_value[:, :, 2:, :] += 100.0
    changed = causal_attention(
        query,
        key,
        changed_value,
        dropout=0.0,
        training=False,
    )
    if not torch.allclose(
        original[:, :, :2, :],
        changed[:, :, :2, :],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError("Changing future values changed an earlier attention output.")
    return "output shape preserved; positions 0-1 cannot see changed positions 2-3"


def _check_next_token_loss() -> str:
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 3]])
    loss = next_token_loss(logits, targets)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Loss must be one finite scalar.")
    loss.backward()
    if logits.grad is None or not torch.all(logits.grad.abs().sum(dim=-1) > 0):
        raise RuntimeError("Every token position must contribute a gradient.")

    good_logits = torch.full((1, 4, 5), -4.0)
    good_logits.scatter_(-1, targets.unsqueeze(-1), 4.0)
    wrong_targets = targets.clone()
    wrong_targets[0, 2] = 4
    if not next_token_loss(good_logits, wrong_targets) > next_token_loss(
        good_logits,
        targets,
    ):
        raise RuntimeError("One wrong token target should increase loss.")
    return "finite scalar; all 4 positions have gradients; wrong target raises loss"


def _check_rope() -> str:
    torch.manual_seed(902)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 5, 8)
    rotated_query, rotated_key = apply_rope(query, key)
    for name, original, rotated in (
        ("query", query, rotated_query),
        ("key", key, rotated_key),
    ):
        if (
            rotated.shape != original.shape
            or rotated.dtype != original.dtype
            or rotated.device != original.device
        ):
            raise RuntimeError(f"RoPE changed {name} shape/dtype/device.")
        if not torch.allclose(
            rotated[:, :, 0, :],
            original[:, :, 0, :],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError(f"RoPE position 0 changed the {name}.")
        original_pair_norms = original.reshape(2, 3, 5, 4, 2).norm(dim=-1)
        rotated_pair_norms = rotated.reshape(2, 3, 5, 4, 2).norm(dim=-1)
        if not torch.allclose(
            rotated_pair_norms,
            original_pair_norms,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(f"RoPE did not preserve paired {name} L2 norms.")
    if torch.allclose(rotated_query[:, :, 1:, :], query[:, :, 1:, :]):
        raise RuntimeError("RoPE did not rotate any nonzero position.")
    return "shape/dtype/device, position-0 identity, and pair-norm checks passed"


def _check_swiglu() -> str:
    torch.manual_seed(903)
    module = SwiGLU(embedding_size=8, dropout=0.0)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = module(hidden)
    if output.shape != hidden.shape:
        raise RuntimeError(f"SwiGLU output shape {output.shape} != {hidden.shape}.")
    output.square().mean().backward()
    for name in ("gate_projection", "up_projection", "down_projection"):
        gradient = getattr(module, name).weight.grad
        if gradient is None or not torch.any(gradient != 0):
            raise RuntimeError(f"SwiGLU gradient did not reach {name}.weight.")
    return "input/output shape matched; gradients reached all 3 projections"


def _check_gqa() -> str:
    torch.manual_seed(904)
    module = GroupedQueryAttention(
        embedding_size=8,
        num_query_heads=4,
        num_kv_heads=2,
        dropout=0.0,
    )
    module.eval()
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    output = module(hidden)
    if output.shape != hidden.shape:
        raise RuntimeError(f"GQA output shape {output.shape} != {hidden.shape}.")
    output.square().mean().backward()
    for name in (
        "query_projection",
        "key_projection",
        "value_projection",
        "output_projection",
    ):
        gradient = getattr(module, name).weight.grad
        if gradient is None or not torch.any(gradient != 0):
            raise RuntimeError(f"GQA gradient did not reach {name}.weight.")

    with torch.no_grad():
        clean_hidden = hidden.detach()
        original = module(clean_hidden)
        changed_hidden = clean_hidden.clone()
        changed_hidden[:, 2:, :] += 100.0
        changed = module(changed_hidden)
    if not torch.allclose(
        original[:, :2, :],
        changed[:, :2, :],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError("GQA leaked changed future tokens into earlier outputs.")
    return "shape/backward checks passed; positions 0-1 cannot see positions 2-3"


def check_scaffold() -> None:
    run_checks(
        "Ex9 learning-target checks (RoPE/SwiGLU/GQA choose one):",
        [
            CheckCase("model construction", _check_model_scaffold),
            CheckCase("EX09_REUSE_CAUSAL_ATTENTION", _check_causal_attention),
            CheckCase("EX09_REUSE_TOKEN_LOSS", _check_next_token_loss),
            CheckCase("EX09_ROPE", _check_rope),
            CheckCase("EX09_SWIGLU", _check_swiglu),
            CheckCase("EX09_GQA", _check_gqa),
        ],
    )


def compare(component: str) -> None:
    model_config = ModelConfig()
    train_config = TrainConfig()
    baseline_losses = train_variant("baseline", model_config, train_config)
    component_losses = train_variant(component, model_config, train_config)
    tail = min(20, train_config.steps)
    baseline_tail_mean = sum(baseline_losses[-tail:]) / tail
    component_tail_mean = sum(component_losses[-tail:]) / tail
    print(f"steps={train_config.steps}")
    print(f"baseline final loss={baseline_losses[-1]:.6f}")
    print(f"{component} final loss={component_losses[-1]:.6f}")
    print(f"delta={component_losses[-1] - baseline_losses[-1]:+.6f}")
    print(f"baseline last-{tail} mean={baseline_tail_mean:.6f}")
    print(f"{component} last-{tail} mean={component_tail_mean:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--component",
        choices=("rope", "swiglu", "gqa"),
        required=True,
    )
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold()
    else:
        compare(args.component)


if __name__ == "__main__":
    main()
