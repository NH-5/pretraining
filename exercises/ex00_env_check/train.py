"""Ex0: inspect the PyTorch accelerator environment and run a tiny smoke test."""

from __future__ import annotations

import platform
import subprocess
import sys

import torch


def _apple_hardware() -> tuple[str | None, str | None]:
    """Return the Apple chip and unified-memory labels without exposing identifiers."""
    if platform.system() != "Darwin":
        return None, None

    result = subprocess.run(
        ["system_profiler", "SPHardwareDataType"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None, None

    chip: str | None = None
    memory: str | None = None
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Chip:"):
            chip = line.partition(":")[2].strip()
        elif line.startswith("Memory:"):
            memory = line.partition(":")[2].strip()
        if chip is not None and memory is not None:
            break
    return chip, memory


def _mps_status() -> tuple[bool, bool]:
    """Return whether this PyTorch build contains MPS and whether it is usable now."""
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False, False
    return mps_backend.is_built(), mps_backend.is_available()


def _select_device(mps_available: bool) -> torch.device:
    """Prefer CUDA, then Apple MPS, while retaining a CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def _gib(byte_count: int) -> float:
    """Convert bytes to GiB for CUDA device reporting."""
    return byte_count / 1024**3


def _print_accelerator_details(mps_available: bool) -> None:
    """Report the accelerator model and its memory semantics."""
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(f"CUDA GPU {index}: {properties.name}")
            print(f"CUDA GPU {index} VRAM: {_gib(properties.total_memory):.2f} GiB")
        return

    if mps_available:
        chip, memory = _apple_hardware()
        print(f"Apple GPU model: {chip or 'unknown Apple Silicon'}")
        print(
            "Apple GPU memory: "
            f"{memory or 'unknown'} unified memory "
            "(shared with CPU; no dedicated CUDA VRAM)"
        )
        return

    print("Accelerator model: none detected")
    print("Accelerator memory: N/A")


def _recommended_precision(device: torch.device) -> str:
    """Follow the repository rule: CUDA bf16 when supported, otherwise fp32."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def _run_tensor_smoke_test(device: torch.device) -> None:
    """Execute a known matrix product on the selected device."""
    matrix = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
        device=device,
    )
    actual = matrix @ matrix
    expected = torch.tensor(
        [[7.0, 10.0], [15.0, 22.0]],
        dtype=torch.float32,
        device=device,
    )
    if not torch.equal(actual, expected):
        raise RuntimeError(f"Tensor smoke test failed: got {actual}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    print(f"Tensor smoke test: PASS (result sum={actual.sum().item():.1f})")


def main() -> None:
    """Print the Ex0 acceptance evidence."""
    mps_built, mps_available = _mps_status()
    device = _select_device(mps_available)

    print("=== Ex0 environment check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"OS: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS built: {mps_built}")
    print(f"MPS available: {mps_available}")
    print(f"Selected device: {device.type}")
    print(f"Recommended precision: {_recommended_precision(device)}")
    _print_accelerator_details(mps_available)
    _run_tensor_smoke_test(device)


if __name__ == "__main__":
    main()
