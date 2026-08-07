"""Minimal check runner shared by the verification scripts."""

from __future__ import annotations

import sys
import traceback
from typing import Any, Callable


class Checks:
    """Collects pass/fail results for one verification script."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.passed = 0
        self.failed: list[str] = []
        self.skipped: list[str] = []

        print(f"\n=== {title}")

    def check(self, name: str, condition: Any) -> bool:
        if condition:
            self.passed += 1
            return True

        self.failed.append(name)
        print(f"  FAIL  {name}")

        return False

    def equal(self, name: str, actual: Any, expected: Any) -> bool:
        if actual == expected:
            self.passed += 1
            return True

        self.failed.append(name)
        print(f"  FAIL  {name}\n          actual   {actual!r}\n          expected {expected!r}")

        return False

    def frames_equal(self, name: str, actual, expected) -> bool:
        import pandas as pd

        try:
            pd.testing.assert_frame_equal(actual, expected)
        except AssertionError as error:
            self.failed.append(name)
            print(f"  FAIL  {name}\n          {str(error).splitlines()[0]}")

            return False

        self.passed += 1

        return True

    def raises(
        self,
        name: str,
        exception: type[BaseException],
        function: Callable[..., Any],
        *args: Any,
        match: str | None = None,
        **kwargs: Any,
    ) -> bool:
        try:
            function(*args, **kwargs)
        except exception as error:
            if match is not None and match not in str(error):
                self.failed.append(name)
                print(f"  FAIL  {name}\n          {match!r} not in {str(error)!r}")

                return False

            self.passed += 1

            return True
        except Exception as error:
            self.failed.append(name)
            print(f"  FAIL  {name}\n          raised {type(error).__name__}: {error}")

            return False

        self.failed.append(name)
        print(f"  FAIL  {name}\n          nothing raised, expected {exception.__name__}")

        return False

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append(name)
        print(f"  SKIP  {name} — {reason}")

    def report(self) -> int:
        summary = f"{self.passed} passed, {len(self.failed)} failed"

        if self.skipped:
            summary += f", {len(self.skipped)} skipped"

        print(f"  {summary}")

        return 1 if self.failed else 0


def run(main: Callable[[], int]) -> None:
    """Run a verification script's `main`, turning an unexpected error into exit 1."""
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
