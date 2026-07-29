"""Ex2: inspect the information flow, then verify three learner explanations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from answers import (
    CAUSAL_MASK_EXPLANATION,
    TOKEN_LOSS_EXPLANATION,
    TRAIN_VS_INFERENCE_EXPLANATION,
)


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


def check_scaffold() -> None:
    trace = build_training_trace(("A", "B", "C", "D"))
    if [len(row.visible_prefix) for row in trace] != [1, 2, 3]:
        raise RuntimeError("Training trace wiring is incorrect.")
    print("Ex2 scaffold: PASS")
    pending = unresolved_answers()
    print(f"Unresolved explanations: {len(pending)}")
    for todo_id in pending:
        print(f"  - {todo_id}")


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
