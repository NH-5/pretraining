"""Ex5: isolate warmup/cosine, accumulation, clipping, and bf16 effects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exercises.checking import CheckCase, SkipCheck, run_checks
from utils import (
    MicroBatch,
    TinyClassifier,
    choose_device,
    make_micro_batches,
    peak_memory_mib,
    precision_context,
    reset_peak_memory,
)


@dataclass(frozen=True)
class ExperimentConfig:
    total_steps: int = 100
    warmup_steps: int = 10
    peak_learning_rate: float = 3e-3
    min_lr_ratio: float = 0.1
    micro_batch_size: int = 16
    accumulation_steps: int = 4
    input_size: int = 256
    hidden_size: int = 1024
    num_classes: int = 128
    max_grad_norm: float = 1.0
    seed: int = 123


def warmup_cosine_lr(step: int, config: ExperimentConfig) -> float:
    """Return the learning rate for one optimizer step."""
    # TODO(你)[EX05_WARMUP_COSINE]: 见指南 §7.2。
    #   方向:warmup 从小 lr 线性升到 peak，之后 cosine 降到 peak*min_ratio。
    #   边界:处理 warmup_steps=0、step=0、step>=total_steps。
    #   完成标准:画/打印关键点，曲线连续且不出现负学习率。
    raise NotImplementedError("TODO[EX05_WARMUP_COSINE]")


def gradient_accumulation_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batches: list[MicroBatch],
    *,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
) -> tuple[float, float]:
    """Run one optimizer update from several micro batches."""
    # TODO(你)[EX05_GRAD_ACCUMULATION]: 见指南 §7.3、§7.4。
    #   方向:清梯度一次；每个 micro loss 除以累积次数再 backward；最后 clip+step。
    #   提示:分类 loss 可用 nn.functional.cross_entropy，不涉及 Ex1 的序列移位。
    #   完成标准:与拼成一个 global batch 的更新相比，参数差异在容差内。
    raise NotImplementedError("TODO[EX05_GRAD_ACCUMULATION]")


def run_experiment(
    config: ExperimentConfig,
    *,
    precision: str,
    disable_warmup: bool,
) -> dict[str, object]:
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    torch.manual_seed(config.seed)
    device = choose_device()
    if disable_warmup:
        config = replace(config, warmup_steps=0)
    model = TinyClassifier(
        config.input_size,
        config.hidden_size,
        config.num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_learning_rate)
    reset_peak_memory(device)

    losses: list[float] = []
    gradient_norms: list[float] = []
    learning_rates: list[float] = []
    model.train()
    for step in range(config.total_steps):
        learning_rate = warmup_cosine_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        micro_batches = make_micro_batches(
            step=step,
            count=config.accumulation_steps,
            batch_size=config.micro_batch_size,
            input_size=config.input_size,
            num_classes=config.num_classes,
            device=device,
            seed=config.seed,
        )
        loss, grad_norm = gradient_accumulation_update(
            model,
            optimizer,
            micro_batches,
            device=device,
            precision=precision,
            max_grad_norm=config.max_grad_norm,
        )
        losses.append(loss)
        gradient_norms.append(grad_norm)
        learning_rates.append(learning_rate)
        if step in {0, config.warmup_steps, config.total_steps - 1}:
            print(
                f"step={step:03d} lr={learning_rate:.6g} "
                f"loss={loss:.5f} grad_norm={grad_norm:.4f}"
            )

    return {
        "device": device.type,
        "precision": precision,
        "warmup_steps": config.warmup_steps,
        "final_loss": losses[-1],
        "max_grad_norm_seen": max(gradient_norms),
        "peak_memory_mib": peak_memory_mib(device),
        "losses": losses,
        "learning_rates": learning_rates,
    }


def _check_micro_batch_wiring() -> str:
    config = ExperimentConfig()
    device = torch.device("cpu")
    batches = make_micro_batches(
        step=0,
        count=config.accumulation_steps,
        batch_size=config.micro_batch_size,
        input_size=config.input_size,
        num_classes=config.num_classes,
        device=device,
        seed=config.seed,
    )
    if len(batches) != config.accumulation_steps:
        raise RuntimeError("Micro-batch factory failed.")
    return (
        f"{len(batches)} micro-batches; global batch="
        f"{config.micro_batch_size * config.accumulation_steps}"
    )


def _check_warmup_cosine() -> str:
    config = replace(
        ExperimentConfig(),
        total_steps=20,
        warmup_steps=4,
        peak_learning_rate=1e-2,
        min_lr_ratio=0.1,
    )
    learning_rates = [
        warmup_cosine_lr(step, config) for step in range(config.total_steps + 1)
    ]
    if any(
        not math.isfinite(rate)
        or rate < 0
        or rate > config.peak_learning_rate * (1.0 + 1e-12)
        for rate in learning_rates
    ):
        raise RuntimeError("Learning rates must stay finite and within [0, peak].")
    warmup = learning_rates[: config.warmup_steps + 1]
    decay = learning_rates[config.warmup_steps :]
    if any(left > right + 1e-12 for left, right in zip(warmup, warmup[1:])):
        raise RuntimeError("Warmup learning rates must be non-decreasing.")
    if any(left + 1e-12 < right for left, right in zip(decay, decay[1:])):
        raise RuntimeError("Cosine-decay learning rates must be non-increasing.")
    if learning_rates[0] >= config.peak_learning_rate:
        raise RuntimeError("Warmup step 0 must start below the peak learning rate.")
    if not math.isclose(
        max(warmup),
        config.peak_learning_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Warmup must reach the configured peak learning rate.")
    minimum = config.peak_learning_rate * config.min_lr_ratio
    if not math.isclose(
        learning_rates[-1],
        minimum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("The schedule must reach peak_lr * min_lr_ratio.")

    no_warmup = replace(config, warmup_steps=0)
    if not math.isclose(
        warmup_cosine_lr(0, no_warmup),
        no_warmup.peak_learning_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("warmup_steps=0 must start at the peak learning rate.")
    if not math.isclose(
        warmup_cosine_lr(config.total_steps + 5, config),
        minimum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Steps beyond total_steps must stay at the minimum lr.")

    decay_span = config.total_steps - config.warmup_steps
    for progress in (0.25, 0.75):
        step = config.warmup_steps + round(decay_span * progress)
        expected = minimum + 0.5 * (
            1.0 + math.cos(math.pi * progress)
        ) * (config.peak_learning_rate - minimum)
        actual = learning_rates[step]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                "Decay has correct endpoints/monotonicity but is not cosine: "
                f"progress={progress:.2f}, expected {expected:.12g}, "
                f"got {actual:.12g}."
            )
    return (
        f"lr[0]={learning_rates[0]:.6g}, "
        f"peak={max(warmup):.6g}, lr[20]={learning_rates[-1]:.6g}; "
        "25%/75% cosine anchors passed"
    )


def _check_gradient_accumulation() -> str:
    torch.manual_seed(505)
    device = torch.device("cpu")
    candidate = TinyClassifier(6, 8, 3).to(device)
    reference = TinyClassifier(6, 8, 3).to(device)
    reference.load_state_dict(candidate.state_dict())
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    micro_batches = make_micro_batches(
        step=0,
        count=3,
        batch_size=4,
        input_size=6,
        num_classes=3,
        device=device,
        seed=505,
    )
    max_grad_norm = 0.4
    reported_loss, reported_norm = gradient_accumulation_update(
        candidate,
        candidate_optimizer,
        micro_batches,
        device=device,
        precision="fp32",
        max_grad_norm=max_grad_norm,
    )

    features = torch.cat([batch.features for batch in micro_batches], dim=0)
    labels = torch.cat([batch.labels for batch in micro_batches], dim=0)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = F.cross_entropy(reference(features), labels)
    reference_loss.backward()
    reference_norm = nn.utils.clip_grad_norm_(
        reference.parameters(),
        max_norm=max_grad_norm,
    )
    reference_optimizer.step()

    if not math.isfinite(float(reported_loss)) or not math.isfinite(
        float(reported_norm)
    ):
        raise RuntimeError("Accumulation must report finite loss and gradient norm.")
    if not math.isclose(
        float(reported_loss),
        reference_loss.item(),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Reported micro-batch mean loss differs from the global-batch loss."
        )
    if not math.isclose(
        float(reported_norm),
        float(reference_norm),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Reported pre-clip gradient norm differs from the global-batch "
            f"reference: expected {float(reference_norm):.9f}, "
            f"got {float(reported_norm):.9f}."
        )
    maximum_difference = max(
        (left - right).abs().max().item()
        for left, right in zip(
            candidate.parameters(),
            reference.parameters(),
            strict=True,
        )
    )
    if maximum_difference > 1e-6:
        raise RuntimeError(
            "Accumulated update differs from one global-batch update: "
            f"max_abs_diff={maximum_difference:.3e}."
        )
    return (
        f"global loss={reference_loss.item():.6f}; "
        f"pre-clip grad norm={float(reference_norm):.6f}; "
        f"parameter max_abs_diff={maximum_difference:.3e}"
    )


def _check_bf16_runtime() -> str:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SkipCheck(
            "bf16 validation needs a CUDA GPU with bf16 support; "
            "the local M1 path intentionally uses fp32."
        )
    device = torch.device("cuda")
    model = TinyClassifier(8, 16, 4).to(device)
    features = torch.randn(2, 8, device=device)
    with precision_context(device, "bf16"):
        output = model(features)
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"Expected bf16 output under autocast, got {output.dtype}.")
    return "CUDA bf16 autocast produced bfloat16 activations"


def check_scaffold() -> None:
    run_checks(
        "Ex5 learning-target checks:",
        [
            CheckCase("micro-batch wiring", _check_micro_batch_wiring),
            CheckCase("EX05_WARMUP_COSINE", _check_warmup_cosine),
            CheckCase("EX05_GRAD_ACCUMULATION + grad clip", _check_gradient_accumulation),
            CheckCase("bf16 runtime", _check_bf16_runtime),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    run_parser.add_argument("--disable-warmup", action="store_true")
    run_parser.add_argument("--steps", type=int, default=ExperimentConfig.total_steps)
    run_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "check":
        check_scaffold()
        return

    result = run_experiment(
        replace(ExperimentConfig(), total_steps=args.steps),
        precision=args.precision,
        disable_warmup=args.disable_warmup,
    )
    summary = json.dumps(result, ensure_ascii=False, indent=2)
    print(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
