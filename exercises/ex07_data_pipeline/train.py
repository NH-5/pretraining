"""Ex7: normalize, filter, deduplicate, count tokens, and export an audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exercises.checking import CheckCase, ManualCheck, SkipCheck, run_checks
from utils import (
    Record,
    audit_sample,
    count_tokens,
    load_tokenizer,
    normalize_text,
    read_jsonl,
    write_jsonl,
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


def _check_normalization_and_audit() -> str:
    raw = "  Café\n\n  language   model  "
    if normalize_text(raw) != "Café language model":
        raise RuntimeError("Normalization check failed.")
    sample = [{"text": f"record {index}"} for index in range(10)]
    first = audit_sample(sample, count=5, seed=42)
    second = audit_sample(sample, count=5, seed=42)
    if len(first) != 5 or first != second:
        raise RuntimeError("Audit sampler failed.")
    return "NFC/whitespace normalization passed; seeded 5-record audit is repeatable"


def _check_quality_filter() -> None:
    examples: list[Record] = [
        {"id": "empty", "text": ""},
        {"id": "short", "text": "Short."},
        {"id": "repeated", "text": "spam " * 40},
        {
            "id": "paragraph",
            "text": (
                "A language-model corpus should contain readable, useful prose "
                "rather than accidental boilerplate."
            ),
        },
    ]
    decisions: list[str] = []
    for record in examples:
        decision = passes_quality_filter(record)
        if type(decision) is not bool:
            raise RuntimeError("passes_quality_filter must return bool for every record.")
        decisions.append(f"{record['id']}={'keep' if decision else 'drop'}")
    raise ManualCheck(
        "The rule returned booleans for boundary examples: "
        + ", ".join(decisions)
        + ". Manually review false positives/negatives and record 3 keep + 3 drop cases."
    )


def _check_exact_deduplication() -> str:
    records: list[Record] = [
        {"id": "first-a", "text": "same text"},
        {"id": "b", "text": "different"},
        {"id": "second-a", "text": "same text"},
        {"id": "third-a", "text": "same text"},
        {"id": "c", "text": "last"},
    ]
    kept, removed = exact_deduplicate(records)
    if removed != 2:
        raise RuntimeError(f"Expected 2 exact duplicates removed, got {removed}.")
    kept_ids = [record.get("id") for record in kept]
    if kept_ids != ["first-a", "b", "c"]:
        raise RuntimeError(
            "Exact dedup must retain first-occurrence order; "
            f"got ids={kept_ids}."
        )
    if len(records) != 5:
        raise RuntimeError("exact_deduplicate must not mutate the input list.")
    return "A,B,A,A,C -> A,B,C; removed=2; first-occurrence order preserved"


def _check_optional_fuzzy_deduplication() -> None:
    records: list[Record] = [
        {"id": "a", "text": "The model predicts the next token from context."},
        {"id": "a2", "text": "The model predicts a next token from context."},
        {"id": "b", "text": "A completely unrelated document about gardens."},
        {"id": "b2", "text": "A completely unrelated document about a garden."},
    ]
    try:
        kept, audit = fuzzy_deduplicate(records)
    except NotImplementedError as error:
        raise SkipCheck(
            "EX07_FUZZY_DEDUP is an optional extension; implement it only "
            "after exact dedup is validated."
        ) from error
    if not isinstance(kept, list) or not isinstance(audit, list):
        raise RuntimeError("Fuzzy dedup must return (kept_records, audit_records).")
    required_audit_keys = {"kept_text", "removed_text", "similarity"}
    for index, record in enumerate(audit):
        if not isinstance(record, dict) or not required_audit_keys.issubset(record):
            raise RuntimeError(
                f"Fuzzy audit record {index} lacks {sorted(required_audit_keys)}."
            )
    raise ManualCheck(
        f"Fuzzy dedup ran: kept={len(kept)}, audit_pairs={len(audit)}. "
        "Inspect at least 3 exported pairs before accepting the threshold."
    )


def check_scaffold() -> None:
    run_checks(
        "Ex7 learning-target checks:",
        [
            CheckCase("normalization + audit wiring", _check_normalization_and_audit),
            CheckCase("EX07_QUALITY_FILTER", _check_quality_filter),
            CheckCase("EX07_EXACT_DEDUP", _check_exact_deduplication),
            CheckCase("EX07_FUZZY_DEDUP (optional)", _check_optional_fuzzy_deduplication),
        ],
    )


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
