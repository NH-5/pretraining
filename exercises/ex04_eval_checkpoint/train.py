"""Ex4: verify evaluation, perplexity, and true checkpoint resume semantics."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn

from utils import (
    evaluate_average_loss,
    load_training_checkpoint,
    next_token_loss,
    perplexity_from_loss,
    save_training_checkpoint,
)


TODO_IDS = (
    "EX04_REUSE_TOKEN_LOSS",
    "EX04_EVAL_AVERAGE",
    "EX04_PERPLEXITY",
    "EX04_SAVE_CHECKPOINT",
    "EX04_LOAD_CHECKPOINT",
)


@dataclass(frozen=True)
class DemoConfig:
    vocab_size: int = 32
    embedding_size: int = 48
    batch_size: int = 16
    sequence_length: int = 12
    first_steps: int = 20
    resumed_steps: int = 10
    learning_rate: float = 1e-2
    seed: int = 7


class TinyNextTokenModel(nn.Module):
    """A small non-attention model so Ex4 isolates checkpoint semantics."""

    def __init__(self, config: DemoConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(token_ids))


def make_batch_factory(
    config: DemoConfig,
    *,
    seed_offset: int,
) -> Callable[[int], tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic batches where the target token is x+1 modulo V."""

    def factory(batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(
            config.seed + seed_offset + batch_index
        )
        inputs = torch.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.sequence_length),
            generator=generator,
        )
        targets = (inputs + 1) % config.vocab_size
        return inputs, targets

    return factory


def train_range(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch_factory: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    *,
    start_step: int,
    end_step: int,
) -> list[float]:
    model.train()
    losses: list[float] = []
    for step in range(start_step, end_step):
        inputs, targets = batch_factory(step)
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses


def new_training_state(
    config: DemoConfig,
) -> tuple[
    TinyNextTokenModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    model = TinyNextTokenModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def check_scaffold() -> None:
    config = DemoConfig()
    torch.manual_seed(config.seed)
    model, optimizer, scheduler = new_training_state(config)
    if not list(model.parameters()) or not optimizer.param_groups:
        raise RuntimeError("Training state was not constructed.")
    if scheduler.get_last_lr()[0] != config.learning_rate:
        raise RuntimeError("Scheduler wiring is incorrect.")
    print("Ex4 scaffold: PASS")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    for todo_id in TODO_IDS:
        print(f"  - {todo_id}")


def run_resume_demo(checkpoint_path: Path) -> None:
    config = DemoConfig()
    batch_factory = make_batch_factory(config, seed_offset=0)
    validation_factory = make_batch_factory(config, seed_offset=10_000)

    torch.manual_seed(config.seed)
    model, optimizer, scheduler = new_training_state(config)
    before_losses = train_range(
        model,
        optimizer,
        scheduler,
        batch_factory,
        start_step=0,
        end_step=config.first_steps,
    )
    validation_before_save = evaluate_average_loss(
        model,
        validation_factory,
        num_batches=5,
    )
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=config.first_steps,
        validation_loss=validation_before_save,
        metadata={
            "model_type": "TinyNextTokenModel",
            "config": asdict(config),
        },
    )

    uninterrupted_losses = train_range(
        model,
        optimizer,
        scheduler,
        batch_factory,
        start_step=config.first_steps,
        end_step=config.first_steps + config.resumed_steps,
    )
    restored_model, restored_optimizer, restored_scheduler = new_training_state(config)
    restored_step, saved_validation_loss, metadata = load_training_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    if metadata["config"] != asdict(config):
        raise RuntimeError("Checkpoint reconstruction metadata changed.")
    validation_after_load = evaluate_average_loss(
        restored_model,
        validation_factory,
        num_batches=5,
    )
    if validation_after_load != saved_validation_loss:
        raise RuntimeError(
            "Validation loss changed across save/load: "
            f"{saved_validation_loss} -> {validation_after_load}"
        )

    after_losses = train_range(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        batch_factory,
        start_step=restored_step,
        end_step=restored_step + config.resumed_steps,
    )
    if len(after_losses) != len(uninterrupted_losses) or any(
        not math.isclose(resumed, reference, rel_tol=0.0, abs_tol=1e-12)
        for resumed, reference in zip(after_losses, uninterrupted_losses, strict=True)
    ):
        raise RuntimeError("Resumed loss curve differs from uninterrupted training.")
    maximum_parameter_difference = max(
        (
            reference.detach() - resumed.detach()
        ).abs().max().item()
        for reference, resumed in zip(
            model.parameters(),
            restored_model.parameters(),
            strict=True,
        )
    )
    if maximum_parameter_difference > 1e-12:
        raise RuntimeError(
            "Resumed parameters differ from uninterrupted training: "
            f"{maximum_parameter_difference:.3e}"
        )
    print(f"last loss before checkpoint: {before_losses[-1]:.6f}")
    print(f"first loss after resume:    {after_losses[0]:.6f}")
    print(f"validation loss:            {validation_after_load:.6f}")
    print(f"perplexity:                 {perplexity_from_loss(validation_after_load):.6f}")
    print(f"resumed step:               {restored_step}")
    print(f"resume/reference max diff:  {maximum_parameter_difference:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "out" / "resume_demo.pt",
    )
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold()
    else:
        run_resume_demo(args.checkpoint)


if __name__ == "__main__":
    main()
