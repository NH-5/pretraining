"""Ex7: normalize, filter, deduplicate, count tokens, and export an audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from utils import (
    Record,
    audit_sample,
    count_tokens,
    load_tokenizer,
    normalize_text,
    read_jsonl,
    write_jsonl,
)


TODO_IDS = (
    "EX07_QUALITY_FILTER",
    "EX07_EXACT_DEDUP",
    "EX07_FUZZY_DEDUP",
)


@dataclass(frozen=True)
class PipelineStats:
    input_documents: int
    filtered_documents: int
    exact_duplicates_removed: int
    fuzzy_duplicates_removed: int
    output_documents: int
    input_tokens: int
    output_tokens: int

    @property
    def document_removal_ratio(self) -> float:
        return 1.0 - self.output_documents / self.input_documents

    @property
    def token_removal_ratio(self) -> float:
        return 1.0 - self.output_tokens / self.input_tokens


def passes_quality_filter(record: Record) -> bool:
    """Decide whether one normalized document is useful training text."""
    # TODO(你)[EX07_QUALITY_FILTER]: 见指南 §4.2。
    #   一次只引入一条可解释规则，例如最短长度或字符重复率。
    #   完成标准:为保留/剔除各写至少 3 个边界样例，并人工抽查误杀。
    raise NotImplementedError("TODO[EX07_QUALITY_FILTER]")


def exact_deduplicate(records: list[Record]) -> tuple[list[Record], int]:
    """Keep one record for each exactly equal normalized text."""
    # TODO(你)[EX07_EXACT_DEDUP]: 见指南 §4.2。
    #   方向:基于 normalize_text 后的内容键去重，保持首次出现顺序。
    #   完成标准:重复输入得到稳定输出；报告 removed 数与 token 变化。
    raise NotImplementedError("TODO[EX07_EXACT_DEDUP]")


def fuzzy_deduplicate(records: list[Record]) -> tuple[list[Record], list[Record]]:
    """Return kept records plus auditable near-duplicate pairs."""
    # TODO(你)[EX07_FUZZY_DEDUP]: 见指南 §4.2。
    #   可选拓展:先定义 shingles/MinHash 相似度与阈值，再做小样例验证。
    #   每个 audit record 至少写 kept_text、removed_text、similarity。
    #   完成标准:不能只报“去掉了多少”；要导出至少 3 对供人工判断。
    raise NotImplementedError("TODO[EX07_FUZZY_DEDUP]")


def normalize_records(records: list[Record]) -> list[Record]:
    normalized: list[Record] = []
    for record in records:
        copy = dict(record)
        copy["text"] = normalize_text(record["text"])
        normalized.append(copy)
    return normalized


def run_pipeline(
    *,
    input_path: Path,
    output_path: Path,
    audit_path: Path,
    tokenizer_path: Path,
    use_fuzzy: bool,
    fuzzy_audit_path: Path | None,
) -> PipelineStats:
    records = normalize_records(read_jsonl(input_path))
    if not records:
        raise ValueError("Input dataset is empty.")
    tokenizer = load_tokenizer(tokenizer_path)
    input_tokens = count_tokens(records, tokenizer)

    quality_kept = [record for record in records if passes_quality_filter(record)]
    filtered_documents = len(records) - len(quality_kept)
    exact_kept, exact_removed = exact_deduplicate(quality_kept)
    if use_fuzzy:
        if fuzzy_audit_path is None:
            raise ValueError("--fuzzy requires --fuzzy-audit.")
        output_records, fuzzy_audit_records = fuzzy_deduplicate(exact_kept)
        write_jsonl(fuzzy_audit_path, fuzzy_audit_records)
        fuzzy_removed = len(fuzzy_audit_records)
    else:
        output_records, fuzzy_removed = exact_kept, 0

    output_tokens = count_tokens(output_records, tokenizer)
    write_jsonl(output_path, output_records)
    write_jsonl(audit_path, audit_sample(output_records, count=5, seed=42))
    return PipelineStats(
        input_documents=len(records),
        filtered_documents=filtered_documents,
        exact_duplicates_removed=exact_removed,
        fuzzy_duplicates_removed=fuzzy_removed,
        output_documents=len(output_records),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def check_scaffold() -> None:
    raw = "  Café\n\n  language   model  "
    if normalize_text(raw) != "Café language model":
        raise RuntimeError("Normalization check failed.")
    sample = [{"text": f"record {index}"} for index in range(10)]
    if len(audit_sample(sample, count=5, seed=42)) != 5:
        raise RuntimeError("Audit sampler failed.")
    print("Ex7 scaffold: PASS")
    print("Optional datasets/tokenizers dependencies are not imported in check mode.")
    for todo_id in TODO_IDS:
        print(f"  - {todo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--audit", type=Path, required=True)
    run_parser.add_argument("--tokenizer", type=Path, required=True)
    run_parser.add_argument("--fuzzy", action="store_true")
    run_parser.add_argument("--fuzzy-audit", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold()
        return

    stats = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        audit_path=args.audit,
        tokenizer_path=args.tokenizer,
        use_fuzzy=args.fuzzy,
        fuzzy_audit_path=args.fuzzy_audit,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    print(f"document removal ratio: {stats.document_removal_ratio:.2%}")
    print(f"token removal ratio:    {stats.token_removal_ratio:.2%}")


if __name__ == "__main__":
    main()
