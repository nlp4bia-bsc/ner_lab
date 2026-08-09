"""Task names addressable from a YAML config, resolved to library functions."""

from __future__ import annotations

import importlib
from typing import Any, Callable

TASKS: dict[str, tuple[str, str]] = {
    "prepare_dataset": ("ner_lab.data.dataset", "prepare_dataset"),
    "train_model": ("ner_lab.training.assessment", "train_model"),
    "search_hyperparameters": ("ner_lab.hpo.search", "search_hyperparameters"),
    "predict_entities": ("ner_lab.inference", "predict_entities"),
}


def resolve_task(name: str) -> Callable[..., Any]:
    """Import and return the function a task name refers to."""
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}. Available tasks: {', '.join(sorted(TASKS))}.")

    module_name, function_name = TASKS[name]

    return getattr(importlib.import_module(module_name), function_name)
