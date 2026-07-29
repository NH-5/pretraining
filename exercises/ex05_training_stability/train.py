"""Ex5: isolate warmup/cosine, accumulation, clipping, and bf16 effects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import torch
from torch import nn

from utils import (
    MicroBatch,
    TinyClassifier,
    choose_device,
    make_micro_batches,
    peak_memory_mib,
    precision_context,
    reset_peak_memory,
)


TODO_IDS = ("EX05_WARMUP_COSINE", "EX05_GRAD_ACCUMULATION")


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


def check_scaffold() -> None:
    config = ExperimentConfig()
    device = choose_device()
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
    print("Ex5 scaffold: PASS")
    print(f"device={device.type}; default precision=fp32")
    print(
        "global batch per optimizer step="
        f"{config.micro_batch_size * config.accumulation_steps}"
    )
    for todo_id in TODO_IDS:
        print(f"  - {todo_id}")


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
