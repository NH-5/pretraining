"""Parsing and display helpers for Ex6 scaling calculations."""

from __future__ import annotations

import re


_SUFFIXES = {
    "": 1.0,
    "K": 1e3,
    "M": 1e6,
    "B": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
}
_QUANTITY_PATTERN = re.compile(
    r"^\s*(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
    r"\s*(?P<suffix>[KMBTPE]?)\s*$",
    re.IGNORECASE,
)


def parse_quantity(value: str) -> float:
    """Parse values such as 7B, 1.4T, or 3e20."""
    match = _QUANTITY_PATTERN.match(value)
    if match is None:
        raise ValueError(f"Invalid quantity: {value!r}")
    number = float(match.group("number"))
    suffix = match.group("suffix").upper()
    return number * _SUFFIXES[suffix]


def format_quantity(value: float) -> str:
    """Render a large non-negative number with a compact SI suffix."""
    if value < 0:
        raise ValueError("Quantity must be non-negative.")
    for suffix, scale in reversed(tuple(_SUFFIXES.items())):
        if suffix and value >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"
