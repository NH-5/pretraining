"""I/O, normalization, token counting, and audit helpers for Ex7."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import random
from typing import Any
import unicodedata


Record = dict[str, Any]


def read_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{line_number} must contain a string 'text'.")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_text(text: str) -> str:
    """Apply conservative normalization before hashing or filtering."""
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def require_tokenizers() -> Any:
    try:
        import tokenizers
    except ImportError as error:
        raise RuntimeError(
            "Ex7 reuses the Ex3 tokenizer. Confirm first, then run: uv add tokenizers"
        ) from error
    return tokenizers


def load_tokenizer(path: Path) -> Any:
    tokenizers = require_tokenizers()
    return tokenizers.Tokenizer.from_file(str(path))


def count_tokens(records: Iterable[Record], tokenizer: Any) -> int:
    return sum(len(tokenizer.encode(record["text"]).ids) for record in records)


def audit_sample(records: list[Record], *, count: int, seed: int) -> list[Record]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    generator = random.Random(seed)
    if count >= len(records):
        return list(records)
    indices = sorted(generator.sample(range(len(records)), count))
    return [records[index] for index in indices]
