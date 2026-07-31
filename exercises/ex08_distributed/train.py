"""Ex8: compare manual data parallel, DDP, and FSDP on 2+ processes."""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from io import StringIO
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exercises.checking import CheckCase, SkipCheck, run_checks
from utils import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
    peak_memory_mib,
)


@dataclass(frozen=True)
class Config:
    hidden_size: int = 128
    vocab_size: int = 256
    micro_batch_size: int = 8
    sequence_length: int = 64
    steps: int = 20
    learning_rate: float = 1e-2
    seed: int = 17


_CHECK_CONTEXT: DistributedContext | None = None


class TokenClassifier(nn.Module):
    """A flat classification workload that does not reveal Ex1 sequence loss."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.vocab_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def average_gradients(model: nn.Module, world_size: int) -> None:
    """Make every replica hold the mean gradient across ranks."""
    # TODO(你)[EX08_ALL_REDUCE]: 见指南 §9.1。
    #   对每个非空 grad 做 SUM all-reduce，再除 world_size；操作是 in-place。
    #   完成标准:各 rank step 后参数 max_abs_diff < 1e-6。
    raise NotImplementedError("TODO[EX08_ALL_REDUCE]")


def wrap_ddp(model: nn.Module, context: DistributedContext) -> nn.Module:
    """Use PyTorch DDP after manually understanding its gradient collective."""
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return DistributedDataParallel(model)


def wrap_fsdp(model: nn.Module, context: DistributedContext) -> nn.Module:
    """Wrap with PyTorch FSDP and document what is sharded."""
    # TODO(你)[EX08_FSDP_WRAP]: 见指南 §9.2。
    #   方向:使用 torch.distributed.fsdp.FullyShardedDataParallel。
    #   完成标准:说明 params/grad/optimizer state 各在哪里分片，并报每卡显存。
    raise NotImplementedError("TODO[EX08_FSDP_WRAP]")


def verify_against_single_card(
    *,
    initial_state: dict[str, torch.Tensor],
    distributed_model: nn.Module,
    context: DistributedContext,
    config: Config,
) -> float:
    """Return max parameter error versus an equivalent single-card reference run."""
    # TODO(你)[EX08_SINGLE_CARD_EQUIVALENCE]: 见指南 §9.1。
    #   方向:rank 0 重建各 rank 的确定性 batch，用相同初始权重跑同样 steps。
    #   所有 rank 都会进入本函数；rank 0 算差异后 broadcast，避免 collective 死锁。
    #   完成标准:manual/DDP 与 reference 的 max_abs_diff 在容差内。
    raise NotImplementedError("TODO[EX08_SINGLE_CARD_EQUIVALENCE]")


def estimate_mfu(
    *,
    num_parameters: int,
    global_tokens_per_second: float,
    peak_tflops_per_gpu: float,
    world_size: int,
) -> float:
    """Reuse the Ex6 estimate on measured distributed throughput."""
    # TODO(你)[EX08_MFU]: 复用并解释 Ex6 的 MFU 公式。
    #   完成标准:使用真实 peak TFLOP/s，不拿产品宣传的低精度峰值冒充 fp32。
    raise NotImplementedError("TODO[EX08_MFU]")


def make_local_batch(
    config: Config,
    *,
    rank: int,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(
        config.seed + rank * 100_000 + step
    )
    features = torch.randn(
        config.micro_batch_size * config.sequence_length,
        config.hidden_size,
        generator=generator,
    ).to(device)
    labels = torch.randint(
        0,
        config.vocab_size,
        (config.micro_batch_size * config.sequence_length,),
        generator=generator,
    ).to(device)
    return features, labels


def unwrap_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        name: value.detach().cpu().clone()
        for name, value in unwrapped.state_dict().items()
    }


def run(
    strategy: str,
    peak_tflops: float | None,
    config: Config,
) -> None:
    context = initialize_distributed()
    try:
        torch.manual_seed(config.seed)
        model: nn.Module = TokenClassifier(config).to(context.device)
        initial_state = unwrap_state_dict(model)
        if strategy == "ddp":
            model = wrap_ddp(model, context)
        elif strategy == "fsdp":
            model = wrap_fsdp(model, context)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)

        dist.barrier()
        start = time.perf_counter()
        final_loss = 0.0
        for step in range(config.steps):
            features, labels = make_local_batch(
                config,
                rank=context.rank,
                step=step,
                device=context.device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            # This is ordinary flat classification; Ex1's [B,T,V] loss stays TODO.
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            if strategy == "manual":
                average_gradients(model, context.world_size)
            optimizer.step()
            final_loss = loss.item()
        dist.barrier()
        elapsed = time.perf_counter() - start

        global_tokens = (
            config.steps
            * config.micro_batch_size
            * config.sequence_length
            * context.world_size
        )
        report: dict[str, Any] = {
            "rank": context.rank,
            "strategy": strategy,
            "backend": context.backend,
            "device": str(context.device),
            "final_local_loss": final_loss,
            "elapsed_seconds": elapsed,
            "global_tokens_per_second": global_tokens / elapsed,
            "peak_memory_mib": peak_memory_mib(context.device),
        }
        gathered: list[dict[str, Any] | None] | None = (
            [None] * context.world_size if context.rank == 0 else None
        )
        dist.gather_object(report, gathered, dst=0)
        difference = verify_against_single_card(
            initial_state=initial_state,
            distributed_model=model,
            context=context,
            config=config,
        )
        if context.rank == 0:
            print("Per-rank reports:")
            for item in gathered or []:
                print(item)
            print(f"single-card max_abs_diff={difference:.3e}")
            if peak_tflops is not None:
                # initial_state was captured before DDP/FSDP sharding, so this is global N.
                params = sum(value.numel() for value in initial_state.values())
                mfu = estimate_mfu(
                    num_parameters=params,
                    global_tokens_per_second=global_tokens / elapsed,
                    peak_tflops_per_gpu=peak_tflops,
                    world_size=context.world_size,
                )
                print(f"estimated MFU={mfu:.2%}")
    finally:
        cleanup_distributed()


def _check_model_wiring() -> str:
    config = Config()
    model = TokenClassifier(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not dist.is_available() or parameter_count <= 0:
        raise RuntimeError("Distributed/model scaffold is unavailable.")
    return f"torch.distributed available; toy parameters={parameter_count:,}"


def _torchrun_context() -> DistributedContext:
    if _CHECK_CONTEXT is not None:
        return _CHECK_CONTEXT
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise SkipCheck(
            "Needs 2 processes. Run: uv run torchrun --nnodes=1 "
            "--nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 "
            "exercises/ex08_distributed/train.py check"
        )
    raise RuntimeError("The multi-process check did not initialize its process group.")


def _check_all_reduce() -> str:
    context = _torchrun_context()

    class GradientProbeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Parameter(torch.zeros(2, 3))
            self.second = nn.Parameter(torch.zeros(4))
            self.unused = nn.Parameter(torch.zeros(1))

    model = GradientProbeModel().to(context.device)
    rank_factor = float(context.rank + 1)
    model.first.grad = torch.full_like(model.first, rank_factor)
    second_pattern = torch.tensor(
        [1.0, -2.0, 3.0, -4.0],
        device=context.device,
    )
    model.second.grad = second_pattern * rank_factor
    average_gradients(model, context.world_size)
    mean_rank_factor = (context.world_size + 1) / 2
    expected_first = torch.full_like(model.first, mean_rank_factor)
    expected_second = second_pattern * mean_rank_factor
    if model.first.grad is None or not torch.allclose(
        model.first.grad,
        expected_first,
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            f"Rank {context.rank} did not average the first parameter gradient."
        )
    if model.second.grad is None or not torch.allclose(
        model.second.grad,
        expected_second,
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            f"Rank {context.rank} did not average every parameter gradient."
        )
    if model.unused.grad is not None:
        raise RuntimeError("A parameter with grad=None must remain untouched.")

    # Also exercise the function at its real call site: one optimizer update
    # from rank-local losses must match a global-batch reference update.
    torch.manual_seed(808)
    candidate = nn.Linear(3, 2).to(context.device)
    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in candidate.state_dict().items()
    }
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.05)
    local_generator = torch.Generator().manual_seed(8_080 + context.rank)
    local_features = torch.randn(3, 3, generator=local_generator).to(context.device)
    local_labels = torch.randint(
        0,
        2,
        (3,),
        generator=local_generator,
    ).to(context.device)
    candidate_optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(candidate(local_features), local_labels).backward()
    average_gradients(candidate, context.world_size)
    candidate_optimizer.step()

    reference = nn.Linear(3, 2).to(context.device)
    reference.load_state_dict(initial_state)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    global_batches = []
    for rank in range(context.world_size):
        generator = torch.Generator().manual_seed(8_080 + rank)
        features = torch.randn(3, 3, generator=generator).to(context.device)
        labels = torch.randint(0, 2, (3,), generator=generator).to(context.device)
        global_batches.append((features, labels))
    global_features = torch.cat([batch[0] for batch in global_batches], dim=0)
    global_labels = torch.cat([batch[1] for batch in global_batches], dim=0)
    reference_optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(reference(global_features), global_labels).backward()
    reference_optimizer.step()
    manual_difference = max(
        (left - right).abs().max().item()
        for left, right in zip(
            candidate.parameters(),
            reference.parameters(),
            strict=True,
        )
    )
    if manual_difference > 1e-6:
        raise RuntimeError(
            "Manual data-parallel update differs from the global-batch update: "
            f"max_abs_diff={manual_difference:.3e}."
        )
    return (
        f"rank={context.rank}; world_size={context.world_size}; "
        f"all parameter gradients averaged by factor={mean_rank_factor:.6f}; "
        f"grad=None preserved; manual update diff={manual_difference:.3e}"
    )


def _check_single_card_equivalence() -> str:
    context = _torchrun_context()
    config = Config(
        hidden_size=8,
        vocab_size=11,
        micro_batch_size=2,
        sequence_length=3,
        steps=2,
        learning_rate=1e-2,
    )
    torch.manual_seed(config.seed)
    model: nn.Module = TokenClassifier(config).to(context.device)
    initial_state = unwrap_state_dict(model)
    model = wrap_ddp(model, context)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    for step in range(config.steps):
        features, labels = make_local_batch(
            config,
            rank=context.rank,
            step=step,
            device=context.device,
        )
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(features), labels).backward()
        optimizer.step()
    difference = verify_against_single_card(
        initial_state=initial_state,
        distributed_model=model,
        context=context,
        config=config,
    )
    if not math.isfinite(difference) or difference > 1e-6:
        raise RuntimeError(
            "DDP differs from the equivalent global-batch reference: "
            f"max_abs_diff={difference:.3e}."
        )

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    parameters = list(unwrapped.parameters())
    if len(parameters) < 2:
        raise RuntimeError("Equivalence probe needs at least two model parameters.")
    with torch.no_grad():
        parameters[-1].add_(0.01)
    corrupted_difference = verify_against_single_card(
        initial_state=initial_state,
        distributed_model=model,
        context=context,
        config=config,
    )
    if not math.isfinite(corrupted_difference) or not math.isclose(
        corrupted_difference,
        0.01,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise RuntimeError(
            "Equivalence verification did not inspect every parameter: "
            "after corrupting the last parameter by 0.01 it reported "
            f"{corrupted_difference:.3e}."
        )
    return (
        f"rank={context.rank}; clean max_abs_diff={difference:.3e}; "
        f"last-parameter corruption detected={corrupted_difference:.3e}"
    )


def _check_fsdp_wrapper() -> str:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SkipCheck(
            "FSDP validation needs a 2+ GPU CUDA machine; local M1/Gloo "
            "can validate manual all-reduce and DDP only."
        )
    context = _torchrun_context()
    from torch.distributed.fsdp import FullyShardedDataParallel

    config = Config(
        hidden_size=8,
        vocab_size=11,
        micro_batch_size=1,
        sequence_length=2,
        steps=1,
    )
    torch.manual_seed(config.seed)
    model = wrap_fsdp(TokenClassifier(config).to(context.device), context)
    if not isinstance(model, FullyShardedDataParallel):
        raise RuntimeError("wrap_fsdp must return FullyShardedDataParallel.")
    features, labels = make_local_batch(
        config,
        rank=context.rank,
        step=0,
        device=context.device,
    )
    F.cross_entropy(model(features), labels).backward()
    dist.barrier()
    return f"rank={context.rank}; FSDP forward/backward passed"


def _check_mfu() -> str:
    actual = estimate_mfu(
        num_parameters=int(1e9),
        global_tokens_per_second=100_000,
        peak_tflops_per_gpu=1000,
        world_size=1,
    )
    if not math.isclose(actual, 0.6, rel_tol=1e-12):
        raise RuntimeError(f"Expected MFU=0.6, got {actual}.")
    return "N=1B, 100k token/s, 1000 TFLOP/s -> MFU=60%"


def check_scaffold() -> None:
    global _CHECK_CONTEXT
    if int(os.environ.get("WORLD_SIZE", "1")) >= 2:
        _CHECK_CONTEXT = initialize_distributed()
    output_context = (
        redirect_stdout(StringIO())
        if _CHECK_CONTEXT is not None and _CHECK_CONTEXT.rank != 0
        else nullcontext()
    )
    try:
        with output_context:
            run_checks(
                "Ex8 learning-target checks:",
                [
                    CheckCase("distributed model wiring", _check_model_wiring),
                    CheckCase("EX08_ALL_REDUCE", _check_all_reduce),
                    CheckCase(
                        "EX08_SINGLE_CARD_EQUIVALENCE",
                        _check_single_card_equivalence,
                    ),
                    CheckCase("EX08_FSDP_WRAP", _check_fsdp_wrapper),
                    CheckCase("EX08_MFU", _check_mfu),
                ],
            )
    finally:
        if _CHECK_CONTEXT is not None:
            cleanup_distributed()
            _CHECK_CONTEXT = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--strategy",
        choices=("manual", "ddp", "fsdp"),
        required=True,
    )
    run_parser.add_argument("--peak-tflops", type=float)
    run_parser.add_argument("--hidden-size", type=int, default=Config.hidden_size)
    run_parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=Config.micro_batch_size,
    )
    run_parser.add_argument(
        "--sequence-length",
        type=int,
        default=Config.sequence_length,
    )
    run_parser.add_argument("--steps", type=int, default=Config.steps)
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold()
    else:
        run(
            args.strategy,
            args.peak_tflops,
            Config(
                hidden_size=args.hidden_size,
                micro_batch_size=args.micro_batch_size,
                sequence_length=args.sequence_length,
                steps=args.steps,
            ),
        )


if __name__ == "__main__":
    main()
