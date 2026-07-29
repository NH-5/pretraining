"""Stream a bounded Hugging Face text slice into local JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def require_datasets() -> Any:
    try:
        import datasets
    except ImportError as error:
        raise RuntimeError(
            "FineWeb streaming needs Hugging Face datasets. When Ex7 starts, "
            "confirm first, then run: uv add datasets"
        ) from error
    return datasets


def export_slice(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    text_column: str,
    limit: int,
    output: Path,
) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    datasets = require_datasets()
    if config_name is None:
        stream = datasets.load_dataset(
            dataset_name,
            split=split,
            streaming=True,
        )
    else:
        stream = datasets.load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for source_index, row in enumerate(stream):
            if written >= limit:
                break
            text = row.get(text_column)
            if not isinstance(text, str):
                continue
            exported = {
                "text": text,
                "source_index": source_index,
                "url": row.get("url"),
            }
            handle.write(json.dumps(exported, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    written = export_slice(
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        text_column=args.text_column,
        limit=args.limit,
        output=args.output,
    )
    print(f"Wrote {written:,} records to {args.output}")


if __name__ == "__main__":
    main()
