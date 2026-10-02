"""Verify package layering and lightweight top-level imports.

Core remains independent from task subpackages. Task subpackages may depend on
core but never on one another. The global CLI is allowed to dispatch to task
subpackages, provided importing it does not eagerly load heavy task runtimes.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from _harness import Checks, run


SRC = Path(__file__).resolve().parent.parent / "src" / "lab"

TASK_SUBPACKAGES = (
    "selection",
    "ner",
    "nel",
)

TORCH_MODULES = (
    "torch",
    "torchcrf",
    "datasets",
    "accelerate",
    "ray",
    "optuna",
)


def imported_modules(path: Path) -> list[str]:
    """Return absolute imports appearing anywhere in one Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)

        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
        ):
            modules.append(node.module)

    return modules


def subpackage_of(module: str) -> str | None:
    """Return the first lab subpackage for an absolute lab import."""
    parts = module.split(".")

    if len(parts) < 2 or parts[0] != "lab":
        return None

    return parts[1]


def forbidden_for(owner: str) -> tuple[str, ...]:
    """Return task subpackages that one source owner may not import."""
    if owner == "core":
        # Core is the shared lower layer and must know nothing about tasks.
        return TASK_SUBPACKAGES

    if owner in TASK_SUBPACKAGES:
        # A task may import itself and core, but never another task.
        return tuple(
            other
            for other in TASK_SUBPACKAGES
            if other != owner
        )

    if owner == "cli":
        # The global CLI is deliberately the composition/dispatch layer.
        # Individual commands import task APIs lazily.
        return ()

    # Root modules such as lab.__init__ should remain task-independent.
    return TASK_SUBPACKAGES


def main() -> int:
    checks = Checks("layering")

    files = sorted(
        path
        for path in SRC.rglob("*.py")
    )

    for path in files:
        relative = path.relative_to(SRC)

        owner = (
            relative.parts[0]
            if len(relative.parts) > 1
            else relative.stem
        )

        modules = imported_modules(path)

        crossing = sorted(
            {
                module
                for module in modules
                if subpackage_of(module)
                in forbidden_for(owner)
            }
        )

        checks.equal(
            f"{relative} imports no forbidden subpackage",
            crossing,
            [],
        )

        # These modules must stay lightweight even syntactically.
        #
        # Task implementations such as selection.representations are allowed
        # to contain local/lazy torch imports inside execution paths, so they
        # are intentionally not included in this static check.
        if owner in (
            "core",
            "cli",
            "__init__",
        ):
            torch = sorted(
                {
                    module
                    for module in modules
                    if module.partition(".")[0]
                    in TORCH_MODULES
                }
            )

            checks.equal(
                f"{relative} stays torch-free",
                torch,
                [],
            )

    checks.check(
        "scanned the source tree",
        len(files) > 20,
    )

    # Runtime import check:
    # importing the shared core, global CLI, and the lazy public selection
    # package must not initialize transformers/torch/task runtimes.
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, lab.core, lab.cli, lab.selection; "
                "print(' '.join(sorted("
                "name for name in "
                "('torch', 'transformers', 'datasets', "
                "'accelerate', 'ray', 'optuna') "
                "if name in sys.modules"
                ")))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    checks.equal(
        (
            "importing lab.core, lab.cli and lab.selection "
            "loads no heavy task runtime"
        ),
        loaded.stdout.split(),
        [],
    )

    return checks.report()


if __name__ == "__main__":
    run(main)