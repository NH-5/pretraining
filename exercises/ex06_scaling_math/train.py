"""Ex6: turn N, D, FLOPs, GPU-days, and MFU into checked calculations."""

from __future__ import annotations

import argparse
import math

from utils import format_quantity, parse_quantity


TODO_IDS = (
    "EX06_TRAINING_FLOPS",
    "EX06_CHINCHILLA_TOKENS",
    "EX06_GPU_DAYS",
    "EX06_MFU",
)


def training_flops(num_parameters: float, num_tokens: float) -> float:
    """Estimate dense Transformer training compute."""
    # TODO(你)[EX06_TRAINING_FLOPS]: 见指南 §8.1。
    #   完成标准:输入 N=1B、D=20B 时返回有限正数，单位明确为 FLOPs。
    raise NotImplementedError("TODO[EX06_TRAINING_FLOPS]")


def chinchilla_tokens(num_parameters: float) -> float:
    """Return the guide's compute-optimal token budget."""
    # TODO(你)[EX06_CHINCHILLA_TOKENS]: 见指南 §8.2。
    #   完成标准:70B 参数得到指南中的 token 数量级。
    raise NotImplementedError("TODO[EX06_CHINCHILLA_TOKENS]")


def gpu_days(
    total_flops: float,
    *,
    peak_tflops_per_gpu: float,
    utilization: float,
    num_gpus: int,
) -> float:
    """Convert compute to wall-clock GPU days using effective throughput."""
    # TODO(你)[EX06_GPU_DAYS]: 见指南 §8.1、§9.4。
    #   方向:TFLOP/s 要换成 FLOP/s；乘 MFU 和卡数；秒再换成天。
    #   完成标准:卡数翻倍时墙钟天数减半，其他量不变。
    raise NotImplementedError("TODO[EX06_GPU_DAYS]")


def estimate_mfu(
    *,
    num_parameters: float,
    measured_tokens_per_second: float,
    peak_tflops_per_gpu: float,
    num_gpus: int,
) -> float:
    """Estimate achieved model FLOPs utilization from measured throughput."""
    # TODO(你)[EX06_MFU]: 见指南 §8.1、§9.4。
    #   方向:先用每 token 约多少 FLOPs 得到 achieved FLOP/s，再除峰值总吞吐。
    #   完成标准:结果在 [0,1]；超过 1 时主动报输入/公式问题。
    raise NotImplementedError("TODO[EX06_MFU]")


def validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}.")


def check_scaffold() -> None:
    cases = {"7B": 7e9, "1.4T": 1.4e12, "3e20": 3e20}
    for text, expected in cases.items():
        actual = parse_quantity(text)
        if actual != expected:
            raise RuntimeError(f"Parser mismatch for {text}: {actual} != {expected}")
    print("Ex6 scaffold: PASS")
    print("Parser examples:", ", ".join(f"{key}={value:g}" for key, value in cases.items()))
    for todo_id in TODO_IDS:
        print(f"  - {todo_id}")


def run_self_test() -> None:
    """Machine-check formulas after the learner fills all TODOs."""
    flops = training_flops(1e9, 20e9)
    expected_flops = 1.2e20
    if not math.isclose(flops, expected_flops, rel_tol=1e-12):
        raise RuntimeError(f"FLOPs check failed: {flops} != {expected_flops}")
    if chinchilla_tokens(70e9) != 1.4e12:
        raise RuntimeError("Chinchilla 70B check failed.")
    one_gpu = gpu_days(
        flops,
        peak_tflops_per_gpu=1000,
        utilization=0.5,
        num_gpus=1,
    )
    two_gpu = gpu_days(
        flops,
        peak_tflops_per_gpu=1000,
        utilization=0.5,
        num_gpus=2,
    )
    if not math.isclose(one_gpu, 2 * two_gpu, rel_tol=1e-12):
        raise RuntimeError("GPU scaling check failed.")
    print("Ex6 formula self-test: PASS")


def estimate(args: argparse.Namespace) -> None:
    num_parameters = parse_quantity(args.parameters)
    num_tokens = parse_quantity(args.tokens)
    validate_positive("parameters", num_parameters)
    validate_positive("tokens", num_tokens)
    flops = training_flops(num_parameters, num_tokens)
    days = gpu_days(
        flops,
        peak_tflops_per_gpu=args.peak_tflops,
        utilization=args.utilization,
        num_gpus=args.num_gpus,
    )
    print(f"parameters:       {format_quantity(num_parameters)}")
    print(f"tokens:           {format_quantity(num_tokens)}")
    print(f"tokens/parameter: {num_tokens / num_parameters:.3f}")
    print(f"training FLOPs:   {flops:.6e}")
    print(f"wall-clock days:  {days:.3f}")
    print(f"GPU-days:         {days * args.num_gpus:.3f}")
    if args.measured_tokens_per_second is not None:
        mfu = estimate_mfu(
            num_parameters=num_parameters,
            measured_tokens_per_second=args.measured_tokens_per_second,
            peak_tflops_per_gpu=args.peak_tflops,
            num_gpus=args.num_gpus,
        )
        print(f"estimated MFU:    {mfu:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("self-test")
    estimate_parser = subparsers.add_parser("estimate")
    estimate_parser.add_argument("--parameters", required=True)
    estimate_parser.add_argument("--tokens", required=True)
    estimate_parser.add_argument("--peak-tflops", type=float, required=True)
    estimate_parser.add_argument("--utilization", type=float, required=True)
    estimate_parser.add_argument("--num-gpus", type=int, required=True)
    estimate_parser.add_argument("--measured-tokens-per-second", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        check_scaffold()
    elif args.command == "self-test":
        run_self_test()
    else:
        estimate(args)


if __name__ == "__main__":
    main()
