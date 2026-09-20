"""Task names addressable from a YAML config, resolved to library functions."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Any, Callable

NAMESPACES = ("core", "ner", "nel")

TASKS: dict[str, tuple[str, str]] = {
    "prepare_dataset": ("lab.core.dataset", "prepare_dataset"),
}

REQUIRES: tuple[str, ...] = ()


def task_table(namespace: str) -> ModuleType:
    """The `tasks` module of one namespace, holding its `TASKS` and `REQUIRES`."""
    if namespace not in NAMESPACES:
        raise ValueError(
            f"Unknown task namespace {namespace!r}. Available namespaces: {', '.join(NAMESPACES)}."
        )

    return importlib.import_module(f"lab.{namespace}.tasks")


def missing_requirements(namespace: str) -> list[str]:
    """Top-level modules a namespace's tasks import that are not installed."""
    return [
        name for name in task_table(namespace).REQUIRES if importlib.util.find_spec(name) is None
    ]


def list_tasks() -> list[tuple[str, list[str]]]:
    """Every task name across namespaces, each with its missing requirements."""
    listed: list[tuple[str, list[str]]] = []

    for namespace in NAMESPACES:
        table = task_table(namespace)
        missing = missing_requirements(namespace)
        listed.extend((f"{namespace}.{task}", missing) for task in sorted(table.TASKS))

    return listed


def resolve_task(name: str) -> Callable[..., Any]:
    """Import and return the function a namespaced task name refers to."""
    namespace, _, task = name.partition(".")

    if not task:
        raise ValueError(
            f"Unknown task {name!r}. Task names are namespaced: "
            f"{', '.join(task_name for task_name, _ in list_tasks())}."
        )

    table = task_table(namespace)

    if task not in table.TASKS:
        raise ValueError(
            f"Unknown task {name!r}. Available tasks: "
            f"{', '.join(f'{namespace}.{known}' for known in sorted(table.TASKS))}."
        )

    module_name, function_name = table.TASKS[task]

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        if getattr(error, "name", None) and error.name.partition(".")[0] in table.REQUIRES:
            raise ValueError(
                f"{name} needs the '{namespace}' extra: pip install lab[{namespace}]"
            ) from error

        raise

    return getattr(module, function_name)
