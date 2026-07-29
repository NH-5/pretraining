"""Non-target helpers for Ex5 stability experiments."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MicroBatch:
    features: torch.Tensor
    labels: torch.Tensor


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def precision_context(
    device: torch.device,
    precision: str,
) -> AbstractContextManager[object]:
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"Unsupported precision: {precision}")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This repository uses bf16 only on supported CUDA devices.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def make_micro_batches(
    *,
    step: int,
    count: int,
    batch_size: int,
    input_size: int,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> list[MicroBatch]:
    if num_classes > input_size:
        raise ValueError("The learnable label rule requires num_classes <= input_size.")
    batches: list[MicroBatch] = []
    for micro_index in range(count):
        generator = torch.Generator().manual_seed(seed + step * count + micro_index)
        features = torch.randn(batch_size, input_size, generator=generator)
        # A fixed rule makes the stream learnable while batches still change each step.
        # Ordinary classification is used here so Ex5 isolates optimizer stability.
        labels = features[:, :num_classes].argmax(dim=1)
        batches.append(
            MicroBatch(features=features.to(device), labels=labels.to(device))
        )
    return batches


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return None


class TinyClassifier(nn.Module):
    """A cheap workload for isolating scheduler/precision behavior."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
