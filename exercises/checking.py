"""A small public-check runner shared by all exercise scaffolds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import sys


class CheckStatus(StrEnum):
    PASS = "PASS"
    PENDING = "PENDING"
    FAIL = "FAIL"
    SKIP = "SKIP"
    MANUAL = "MANUAL"


class CheckSignal(Exception):
    """Base class for non-failure states raised by a public check."""


class PendingCheck(CheckSignal):
    """The learning target has not been implemented yet."""


class SkipCheck(CheckSignal):
    """The check cannot run in the current dependency/hardware environment."""


class ManualCheck(CheckSignal):
    """Automatic structure checks passed, but a human judgment remains."""


@dataclass(frozen=True)
class CheckCase:
    name: str
    check: Callable[[], str | None]


def _run_one(case: CheckCase) -> tuple[CheckStatus, str]:
    try:
        detail = case.check()
    except (NotImplementedError, PendingCheck) as error:
        return CheckStatus.PENDING, str(error) or "implementation is still pending"
    except SkipCheck as error:
        return CheckStatus.SKIP, str(error) or "current environment cannot run this check"
    except ManualCheck as error:
        return CheckStatus.MANUAL, str(error) or "human review is required"
    except Exception as error:
        return CheckStatus.FAIL, f"{type(error).__name__}: {error}"
    return CheckStatus.PASS, str(detail) if detail is not None else "behavior check passed"


def run_checks(
    title: str,
    cases: Sequence[CheckCase],
) -> dict[CheckStatus, int]:
    """Print stable statuses and fail the process only for incorrect behavior."""
    print(title)
    counts = {status: 0 for status in CheckStatus}
    for case in cases:
        status, detail = _run_one(case)
        counts[status] += 1
        print(f"[{status}] {case.name}")
        for line in detail.splitlines():
            print(f"  {line}")

    print(
        "\nSummary: "
        f"{counts[CheckStatus.PASS]} passed, "
        f"{counts[CheckStatus.PENDING]} pending, "
        f"{counts[CheckStatus.FAIL]} failed, "
        f"{counts[CheckStatus.SKIP]} skipped, "
        f"{counts[CheckStatus.MANUAL]} manual"
    )
    needs_guidance = (
        counts[CheckStatus.PENDING]
        + counts[CheckStatus.FAIL]
        + counts[CheckStatus.MANUAL]
    )
    hints_path = Path(sys.argv[0]).resolve().parent / "HINTS.md"
    if needs_guidance and hints_path.is_file():
        try:
            shown_path = hints_path.relative_to(Path.cwd())
        except ValueError:
            shown_path = hints_path
        print(
            f"\n下一步提示：打开 {shown_path}，按 TODO ID 搜索；"
            "每次只多看一档提示。"
        )
    if counts[CheckStatus.FAIL]:
        raise RuntimeError("One or more public checks failed.")
    return counts
