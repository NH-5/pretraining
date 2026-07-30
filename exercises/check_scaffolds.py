"""Validate exercise structure, TODO contracts, and safe check entry points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXERCISES = ROOT / "exercises"
TODO_PATTERN = re.compile(r"TODO\(你\)\[([A-Z0-9_]+)\]")


@dataclass(frozen=True)
class ExerciseContract:
    directory: str
    required_files: tuple[str, ...]
    todo_ids: tuple[str, ...]
    check_args: tuple[str, ...]


CONTRACTS = (
    ExerciseContract(
        "ex00_env_check",
        ("README.md", "train.py", "notes.md"),
        (),
        (),
    ),
    ExerciseContract(
        "ex01_char_gpt",
        ("README.md", "HINTS.md", "train.py", "utils.py", "notes.md"),
        (
            "EX01_CAUSAL_MASK",
            "EX01_TOKEN_LOSS",
            "EX01_AUTOREGRESSIVE_GENERATION",
        ),
        ("check",),
    ),
    ExerciseContract(
        "ex02_nanogpt_walkthrough",
        ("README.md", "HINTS.md", "train.py", "answers.py", "notes.md"),
        (
            "EX02_EXPLAIN_CAUSAL_MASK",
            "EX02_EXPLAIN_TOKEN_LOSS",
            "EX02_EXPLAIN_TRAIN_VS_INFERENCE",
        ),
        ("check",),
    ),
    ExerciseContract(
        "ex03_bpe_tokenizer",
        ("README.md", "HINTS.md", "train.py", "utils.py", "notes.md"),
        ("EX03_BUILD_BPE", "EX03_INTERPRET_RATIO"),
        ("check",),
    ),
    ExerciseContract(
        "ex04_eval_checkpoint",
        ("README.md", "HINTS.md", "train.py", "utils.py", "notes.md"),
        (
            "EX04_REUSE_TOKEN_LOSS",
            "EX04_EVAL_AVERAGE",
            "EX04_PERPLEXITY",
            "EX04_SAVE_CHECKPOINT",
            "EX04_LOAD_CHECKPOINT",
        ),
        ("check",),
    ),
    ExerciseContract(
        "ex05_training_stability",
        ("README.md", "HINTS.md", "train.py", "utils.py", "notes.md"),
        ("EX05_WARMUP_COSINE", "EX05_GRAD_ACCUMULATION"),
        ("check",),
    ),
    ExerciseContract(
        "ex06_scaling_math",
        ("README.md", "HINTS.md", "train.py", "utils.py", "notes.md"),
        (
            "EX06_TRAINING_FLOPS",
            "EX06_CHINCHILLA_TOKENS",
            "EX06_GPU_DAYS",
            "EX06_MFU",
        ),
        ("check",),
    ),
    ExerciseContract(
        "ex07_data_pipeline",
        (
            "README.md",
            "HINTS.md",
            "train.py",
            "utils.py",
            "prepare_data.py",
            "notes.md",
        ),
        ("EX07_QUALITY_FILTER", "EX07_EXACT_DEDUP", "EX07_FUZZY_DEDUP"),
        ("check",),
    ),
    ExerciseContract(
        "ex08_distributed",
        ("README.md", "HINTS.md", "train.py", "utils.py", "notes.md"),
        (
            "EX08_ALL_REDUCE",
            "EX08_SINGLE_CARD_EQUIVALENCE",
            "EX08_FSDP_WRAP",
            "EX08_MFU",
        ),
        ("check",),
    ),
    ExerciseContract(
        "ex09_modern_component",
        (
            "README.md",
            "HINTS.md",
            "train.py",
            "model.py",
            "components.py",
            "notes.md",
        ),
        (
            "EX09_REUSE_CAUSAL_ATTENTION",
            "EX09_REUSE_TOKEN_LOSS",
            "EX09_ROPE",
            "EX09_SWIGLU",
            "EX09_GQA",
        ),
        ("check",),
    ),
    ExerciseContract(
        "ex10_sft",
        (
            "README.md",
            "HINTS.md",
            "train.py",
            "utils.py",
            "model_adapter.py",
            "sample_instructions.jsonl",
            "notes.md",
        ),
        (
            "EX10_FORMAT_INSTRUCTION",
            "EX10_RESPONSE_LOSS_MASK",
            "EX10_LOAD_BASE_MODEL",
            "EX10_RESPONSE_ONLY_LOSS",
            "EX10_SAVE_SFT",
        ),
        ("check",),
    ),
)


def python_source(directory: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*.py"))
    )


def check_contract(
    contract: ExerciseContract,
    *,
    expect_unfinished: bool,
    run_entry_points: bool,
) -> tuple[int, int]:
    directory = EXERCISES / contract.directory
    missing = [
        filename
        for filename in contract.required_files
        if not (directory / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"{contract.directory} missing files: {missing}")

    source = python_source(directory)
    found = TODO_PATTERN.findall(source)
    unknown = set(found) - set(contract.todo_ids)
    if unknown:
        raise RuntimeError(f"{contract.directory} unknown TODOs: {sorted(unknown)}")
    unresolved = [todo_id for todo_id in contract.todo_ids if todo_id in found]
    duplicates = sorted(
        todo_id for todo_id in set(found) if found.count(todo_id) > 1
    )
    if duplicates:
        raise RuntimeError(
            f"{contract.directory} duplicate TODO markers: {duplicates}"
        )
    if expect_unfinished and len(unresolved) != len(contract.todo_ids):
        completed = sorted(set(contract.todo_ids) - set(unresolved))
        raise RuntimeError(
            f"{contract.directory} expected untouched scaffold TODOs; missing {completed}"
        )

    if run_entry_points:
        command = [
            sys.executable,
            str(directory / "train.py"),
            *contract.check_args,
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{contract.directory} check failed ({result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    return len(unresolved), len(contract.todo_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-unfinished",
        action="store_true",
        help="Require every original learning-target TODO to still be present.",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Skip each exercise's safe check entry point.",
    )
    args = parser.parse_args()

    unresolved_total = 0
    todo_total = 0
    for contract in CONTRACTS:
        unresolved, total = check_contract(
            contract,
            expect_unfinished=args.expect_unfinished,
            run_entry_points=not args.structure_only,
        )
        unresolved_total += unresolved
        todo_total += total
        print(
            f"{contract.directory}: PASS "
            f"(unresolved learning targets {unresolved}/{total})"
        )
    print(
        f"All {len(CONTRACTS)} exercise scaffolds passed; "
        f"unresolved learning targets {unresolved_total}/{todo_total}."
    )


if __name__ == "__main__":
    main()
