"""Clinical summarization: evaluating generated summaries."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "evaluate_summaries": "lab.summarization.evaluation",
    "load_judge_model": "lab.summarization.evaluation",
    "read_summaries": "lab.summarization.io",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
