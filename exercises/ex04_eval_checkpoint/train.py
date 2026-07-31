"""Ex4: verify evaluation, perplexity, and true checkpoint resume semantics."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exercises.checking import CheckCase, run_checks
from utils import (
    evaluate_average_loss,
    load_training_checkpoint,
    next_token_loss,
    perplexity_from_loss,
    save_training_checkpoint,
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


class _EvaluationProbeModel(nn.Module):
    """A dropout-bearing model that exposes missing eval-mode handling."""

    def __init__(self, config: DemoConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embedding(token_ids)))


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


def _check_training_state_wiring() -> str:
    config = DemoConfig()
    torch.manual_seed(config.seed)
    model, optimizer, scheduler = new_training_state(config)
    if not list(model.parameters()) or not optimizer.param_groups:
        raise RuntimeError("Training state was not constructed.")
    if scheduler.get_last_lr()[0] != config.learning_rate:
        raise RuntimeError("Scheduler wiring is incorrect.")
    return f"model parameters={sum(p.numel() for p in model.parameters()):,}"


def _check_next_token_loss() -> str:
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 3]])
    loss = next_token_loss(logits, targets)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Loss must be one finite scalar.")
    expected = math.log(5.0)
    if not math.isclose(loss.item(), expected, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError(
            "Loss must be the mean over all token positions: "
            f"uniform V=5 expected ln(5)={expected:.6f}, got {loss.item():.6f}."
        )
    loss.backward()
    if logits.grad is None:
        raise RuntimeError("Loss did not create logits gradients.")
    position_gradient = logits.grad.abs().sum(dim=-1)
    if not torch.all(position_gradient > 0):
        raise RuntimeError("Every token position must contribute a gradient.")

    good_logits = torch.full((1, 4, 5), -4.0)
    good_logits.scatter_(-1, targets.unsqueeze(-1), 4.0)
    wrong_targets = targets.clone()
    wrong_targets[0, 1] = 4
    good_loss = next_token_loss(good_logits, targets)
    wrong_loss = next_token_loss(good_logits, wrong_targets)
    if not wrong_loss > good_loss:
        raise RuntimeError("Making one token target wrong should increase loss.")
    return (
        f"mean loss=ln(5)={loss.item():.6f}; all 4 positions have gradients; "
        "wrong target raises loss"
    )


def _check_evaluation() -> str:
    config = DemoConfig(
        vocab_size=7,
        embedding_size=8,
        batch_size=2,
        sequence_length=4,
    )
    torch.manual_seed(config.seed)
    model = _EvaluationProbeModel(config)
    model.train()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    factory = make_batch_factory(config, seed_offset=400)

    incoming_mode = model.training
    model.eval()
    expected_batch_losses: list[float] = []
    with torch.no_grad():
        for batch_index in range(3):
            inputs, targets = factory(batch_index)
            expected_batch_losses.append(
                F.cross_entropy(
                    model(inputs).reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="mean",
                ).item()
            )
    model.train(incoming_mode)
    expected = sum(expected_batch_losses) / len(expected_batch_losses)

    first = evaluate_average_loss(model, factory, num_batches=3)
    second = evaluate_average_loss(model, factory, num_batches=3)
    if not isinstance(first, float) or not math.isfinite(first):
        raise RuntimeError("Evaluation must return one finite Python float.")
    if not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Deterministic evaluation changed: {first} != {second}.")
    if not math.isclose(first, expected, rel_tol=1e-7, abs_tol=1e-7):
        raise RuntimeError(
            "Evaluation must average all requested batches in eval mode: "
            f"expected {expected:.9f}, got {first:.9f}."
        )
    if not model.training:
        raise RuntimeError("Evaluation did not restore the model's training mode.")
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter, parameters_before[name]):
            raise RuntimeError(f"Evaluation changed parameter {name}.")
    model.eval()
    third = evaluate_average_loss(model, factory, num_batches=3)
    if model.training:
        raise RuntimeError("Evaluation changed a model that was already in eval mode.")
    if not math.isclose(third, first, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Evaluation result depends on the model's incoming mode.")
    return (
        f"3-batch eval average={first:.6f}; dropout disabled; "
        "incoming mode and parameters restored"
    )


def _check_perplexity() -> str:
    at_zero = perplexity_from_loss(0.0)
    at_log_two = perplexity_from_loss(math.log(2.0))
    at_one = perplexity_from_loss(1.0)
    if not math.isclose(at_zero, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"loss=0 must produce PPL=1, got {at_zero}.")
    if not math.isclose(at_log_two, 2.0, rel_tol=1e-12):
        raise RuntimeError(f"loss=ln(2) must produce PPL=2, got {at_log_two}.")
    if not at_one > at_zero:
        raise RuntimeError("Perplexity must increase when average loss increases.")
    return "loss=0 -> PPL=1; loss=ln(2) -> PPL=2; monotonicity passed"


def _assert_nested_equal(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
            raise RuntimeError(f"Checkpoint state differs at {path}.")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError(f"Checkpoint mapping differs at {path}.")
        for key in expected:
            _assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise RuntimeError(f"Checkpoint sequence differs at {path}.")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint value differs at {path}: {actual!r} != {expected!r}."
        )


def _check_checkpoint_round_trip() -> str:
    torch.manual_seed(404)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    features = torch.tensor([[1.0, -2.0, 0.5], [0.5, 1.5, -1.0]])
    optimizer.zero_grad(set_to_none=True)
    model(features).square().mean().backward()
    optimizer.step()
    scheduler.step()

    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    metadata = {
        "model_type": "Linear",
        "config": {"in_features": 3, "out_features": 2},
        "tokenizer_vocab": ["a", "b", "c"],
    }
    rng_at_save = torch.get_rng_state().clone()

    with TemporaryDirectory(prefix="ex04-check-") as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "nested" / "resume.pt"
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=7,
            validation_loss=1.25,
            metadata=metadata,
        )
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise RuntimeError("Checkpoint file was not created.")

        torch.set_rng_state(rng_at_save)
        expected_random = torch.rand(5)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.add_(3.0)
        optimizer.param_groups[0]["lr"] = 0.123
        scheduler.step()
        torch.manual_seed(999)

        step, validation_loss, restored_metadata = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        actual_random = torch.rand(5)

    _assert_nested_equal(model.state_dict(), expected_model, "model")
    _assert_nested_equal(optimizer.state_dict(), expected_optimizer, "optimizer")
    _assert_nested_equal(scheduler.state_dict(), expected_scheduler, "scheduler")
    if step != 7 or validation_loss != 1.25 or restored_metadata != metadata:
        raise RuntimeError("Checkpoint did not restore step/loss/metadata exactly.")
    if not torch.equal(actual_random, expected_random):
        raise RuntimeError("Checkpoint did not restore the CPU RNG state.")
    return "model/optimizer/scheduler/step/metadata/CPU RNG round trip passed"


def check_scaffold() -> None:
    run_checks(
        "Ex4 learning-target checks:",
        [
            CheckCase("training-state wiring", _check_training_state_wiring),
            CheckCase("EX04_REUSE_TOKEN_LOSS", _check_next_token_loss),
            CheckCase("EX04_EVAL_AVERAGE", _check_evaluation),
            CheckCase("EX04_PERPLEXITY", _check_perplexity),
            CheckCase(
                "EX04_SAVE_CHECKPOINT / EX04_LOAD_CHECKPOINT",
                _check_checkpoint_round_trip,
            ),
        ],
    )


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
