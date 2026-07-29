"""Utilities for Ex1 that are not the learning targets of the exercise."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
import urllib.request

import torch


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class CharTokenizer:
    """A deterministic character vocabulary built from one corpus."""

    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a sorted vocabulary so repeated runs assign the same IDs."""
        chars = tuple(sorted(set(text)))
        if not chars:
            raise ValueError("The corpus is empty.")
        return cls(chars=chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        stoi = {char: index for index, char in enumerate(self.chars)}
        try:
            return [stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"Character {error.args[0]!r} is outside the vocabulary.") from error

    def decode(self, token_ids: list[int]) -> str:
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in token_ids):
            raise ValueError("A token ID is outside the vocabulary.")
        return "".join(self.chars[token_id] for token_id in token_ids)


def prepare_tiny_shakespeare(destination: Path) -> int:
    """Download the public-domain tiny Shakespeare corpus once."""
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if len(existing) >= 500_000:
            return len(existing)

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        TINY_SHAKESPEARE_URL,
        headers={"User-Agent": "pretraining-study-repository/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    text = payload.decode("utf-8")
    if len(text) < 500_000:
        raise RuntimeError(
            "Downloaded corpus is unexpectedly small; refusing to train on a partial file."
        )
    destination.write_text(text, encoding="utf-8")
    return len(text)


def split_token_stream(
    token_ids: list[int],
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a contiguous token stream without leaking validation suffixes."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    split_index = int(len(token_ids) * train_fraction)
    if split_index == 0 or split_index == len(token_ids):
        raise ValueError("The corpus is too short for the requested split.")
    tokens = torch.tensor(token_ids, dtype=torch.long)
    return tokens[:split_index], tokens[split_index:]


def get_batch(
    token_stream: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample contiguous contexts; targets are the same stream shifted by one."""
    number_of_starts = len(token_stream) - block_size
    if number_of_starts < 1:
        raise ValueError("Token stream must contain at least block_size + 1 tokens.")
    starts = torch.randint(
        low=0,
        high=number_of_starts,
        size=(batch_size,),
        generator=generator,
    )
    offsets = torch.arange(block_size)
    inputs = token_stream[starts[:, None] + offsets]
    targets = token_stream[starts[:, None] + offsets + 1]
    return inputs.to(device), targets.to(device)


def select_device() -> torch.device:
    """Use CUDA first, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device) -> AbstractContextManager[object]:
    """Follow the repository precision rule from guide §7.4."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def seed_everything(seed: int) -> None:
    """Make data sampling and parameter initialization reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
