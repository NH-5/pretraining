"""Ex9: compare one modern decoder component against a fixed baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from components import GroupedQueryAttention, SwiGLU
from model import ModelConfig, TinyGPT, next_token_loss


TODO_IDS = (
    "EX09_REUSE_CAUSAL_ATTENTION",
    "EX09_REUSE_TOKEN_LOSS",
    "EX09_ROPE",
    "EX09_SWIGLU",
    "EX09_GQA",
)


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


def check_scaffold() -> None:
    config = ModelConfig()
    counts = {}
    for component in ("baseline", "rope", "swiglu", "gqa"):
        model = TinyGPT(config, component)
        counts[component] = sum(parameter.numel() for parameter in model.parameters())
    print("Ex9 scaffold: PASS")
    for component, count in counts.items():
        print(f"{component:8s} parameters={count:,}")
    print("Only one of ROPE/SWIGLU/GQA is required for completion.")
    for todo_id in TODO_IDS:
        print(f"  - {todo_id}")


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
