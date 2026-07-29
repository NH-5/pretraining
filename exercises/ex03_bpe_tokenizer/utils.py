"""Reporting helpers for Ex3; tokenizer training remains in train.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RatioRow:
    tokenizer: str
    language: str
    characters: int
    tokens: int

    @property
    def characters_per_token(self) -> float:
        if self.tokens == 0:
            raise ValueError("Tokenizer produced zero tokens for non-empty text.")
        return self.characters / self.tokens


def require_tokenizers() -> Any:
    """Import the optional dependency only in modes that truly need it."""
    try:
        import tokenizers
    except ImportError as error:
        raise RuntimeError(
            "Ex3 needs Hugging Face tokenizers. When you reach this exercise, "
            "ask before running: uv add tokenizers"
        ) from error
    return tokenizers


def load_tokenizer(path: Path) -> Any:
    tokenizers = require_tokenizers()
    if not path.exists():
        raise FileNotFoundError(path)
    return tokenizers.Tokenizer.from_file(str(path))


def measure(
    tokenizer: Any,
    *,
    tokenizer_name: str,
    language: str,
    text: str,
) -> RatioRow:
    if not text:
        raise ValueError("Comparison text must not be empty.")
    encoding = tokenizer.encode(text)
    return RatioRow(
        tokenizer=tokenizer_name,
        language=language,
        characters=len(text),
        tokens=len(encoding.ids),
    )


def format_table(rows: list[RatioRow]) -> str:
    headers = ("tokenizer", "language", "characters", "tokens", "chars/token")
    body = [
        (
            row.tokenizer,
            row.language,
            str(row.characters),
            str(row.tokens),
            f"{row.characters_per_token:.3f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in body))
        for column in range(len(headers))
    ]

    def render(row: tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in body)])
