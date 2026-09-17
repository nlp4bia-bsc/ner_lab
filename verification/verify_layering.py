"""The one-way import rule: core never imports a task subpackage, task subpackages never import each other."""

from __future__ import annotations

import ast
from pathlib import Path

from _harness import Checks, run

SRC = Path(__file__).resolve().parent.parent / "src" / "lab"
TASK_SUBPACKAGES = ("ner",)
TORCH_MODULES = ("torch", "torchcrf", "datasets", "accelerate", "ray", "optuna")


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module)

    return modules


def subpackage_of(module: str) -> str | None:
    parts = module.split(".")

    if len(parts) < 2 or parts[0] != "lab":
        return None

    return parts[1]


def forbidden_for(owner: str) -> tuple[str, ...]:
    if owner == "core":
        return TASK_SUBPACKAGES

    if owner in TASK_SUBPACKAGES:
        return tuple(other for other in TASK_SUBPACKAGES if other != owner)

    return TASK_SUBPACKAGES


def main() -> int:
    checks = Checks("layering")
    files = sorted(path for path in SRC.rglob("*.py"))

    for path in files:
        relative = path.relative_to(SRC)
        owner = relative.parts[0] if len(relative.parts) > 1 else relative.stem
        modules = imported_modules(path)

        crossing = sorted(
            {module for module in modules if subpackage_of(module) in forbidden_for(owner)}
        )
        checks.equal(f"{relative} imports no forbidden subpackage", crossing, [])

        if owner in ("core", "cli", "__init__"):
            torch = sorted({module for module in modules if module.partition(".")[0] in TORCH_MODULES})
            checks.equal(f"{relative} stays torch-free", torch, [])

    checks.check("scanned the source tree", len(files) > 20)

    return checks.report()


if __name__ == "__main__":
    run(main)
