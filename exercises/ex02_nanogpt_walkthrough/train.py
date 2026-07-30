"""Ex2: inspect the information flow, then verify three learner explanations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from answers import (
    CAUSAL_MASK_EXPLANATION,
    TOKEN_LOSS_EXPLANATION,
    TRAIN_VS_INFERENCE_EXPLANATION,
)
from exercises.checking import CheckCase, ManualCheck, PendingCheck, run_checks


@dataclass(frozen=True)
class PositionTrace:
    position: int
    visible_prefix: tuple[str, ...]
    next_token_target: str


def build_training_trace(tokens: tuple[str, ...]) -> list[PositionTrace]:
    """Expose data flow without implementing the mask tensor itself."""
    if len(tokens) < 2:
        raise ValueError("Need at least two tokens.")
    return [
        PositionTrace(
            position=position,
            visible_prefix=tokens[: position + 1],
            next_token_target=tokens[position + 1],
        )
        for position in range(len(tokens) - 1)
    ]


def print_training_trace() -> None:
    tokens = ("To", "be", "or", "not", "to", "be")
    print("position | visible prefix       | target")
    print("---------+----------------------+-------")
    for row in build_training_trace(tokens):
        prefix = " ".join(row.visible_prefix)
        print(f"{row.position:^8} | {prefix:<20} | {row.next_token_target}")


def unresolved_answers() -> list[str]:
    answers = {
        "EX02_EXPLAIN_CAUSAL_MASK": CAUSAL_MASK_EXPLANATION,
        "EX02_EXPLAIN_TOKEN_LOSS": TOKEN_LOSS_EXPLANATION,
        "EX02_EXPLAIN_TRAIN_VS_INFERENCE": TRAIN_VS_INFERENCE_EXPLANATION,
    }
    return [todo_id for todo_id, answer in answers.items() if not answer.strip()]


def _check_training_trace() -> str:
    trace = build_training_trace(("A", "B", "C", "D"))
    if [len(row.visible_prefix) for row in trace] != [1, 2, 3]:
        raise RuntimeError("Training trace wiring is incorrect.")
    if [row.next_token_target for row in trace] != ["B", "C", "D"]:
        raise RuntimeError("Next-token targets are not aligned with their prefixes.")
    return "prefix lengths=[1, 2, 3]; targets=['B', 'C', 'D']"


def _review_explanation(todo_id: str, answer: str, guide_section: str) -> None:
    if not answer.strip():
        raise PendingCheck(f"Fill {todo_id} in answers.py; see guide {guide_section}.")
    raise ManualCheck(
        f"Written answer found ({len(answer.strip())} characters). "
        "Explain it aloud without reading; semantic correctness needs human review."
    )


def check_scaffold() -> None:
    run_checks(
        "Ex2 learning-target checks:",
        [
            CheckCase("training trace wiring", _check_training_trace),
            CheckCase(
                "EX02_EXPLAIN_CAUSAL_MASK",
                lambda: _review_explanation(
                    "EX02_EXPLAIN_CAUSAL_MASK",
                    CAUSAL_MASK_EXPLANATION,
                    "§3.1",
                ),
            ),
            CheckCase(
                "EX02_EXPLAIN_TOKEN_LOSS",
                lambda: _review_explanation(
                    "EX02_EXPLAIN_TOKEN_LOSS",
                    TOKEN_LOSS_EXPLANATION,
                    "§6.1、§6.3",
                ),
            ),
            CheckCase(
                "EX02_EXPLAIN_TRAIN_VS_INFERENCE",
                lambda: _review_explanation(
                    "EX02_EXPLAIN_TRAIN_VS_INFERENCE",
                    TRAIN_VS_INFERENCE_EXPLANATION,
                    "§6.3",
                ),
            ),
        ],
    )


def verify_answers() -> None:
    pending = unresolved_answers()
    print_training_trace()
    if pending:
        raise RuntimeError(
            "Complete answers.py before verification: " + ", ".join(pending)
        )
    print("\nWritten answers are present. Now explain all three aloud without reading them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "trace", "verify"),
        help="check scaffold, show a token trace, or verify filled explanations",
    )
    args = parser.parse_args()
    if args.command == "check":
        check_scaffold()
    elif args.command == "trace":
        print_training_trace()
    else:
        verify_answers()


if __name__ == "__main__":
    main()
