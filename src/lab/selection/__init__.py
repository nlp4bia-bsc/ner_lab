"""Task-independent document selection for canonical :mod:`lab` corpora.

The package deliberately depends on :mod:`lab.core`, never on NER or NEL.  Public
objects are imported lazily so ``import lab.selection`` stays lightweight and does
not load scikit-learn, torch, or model weights.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "SelectionRun": "lab.selection.api",
    "build_representations": "lab.selection.api",
    "compare_methods": "lab.selection.api",
    "consensus_select_documents": "lab.selection.api",
    "select_documents": "lab.selection.api",
    "selection_history": "lab.selection.api",
    "validate_selection_input": "lab.selection.api",
    "MethodInfo": "lab.selection.methods",
    "SelectionMethod": "lab.selection.methods",
    "get_method": "lab.selection.methods",
    "list_methods": "lab.selection.methods",
    "method_info": "lab.selection.methods",
    "RepresentationInfo": "lab.selection.representations",
    "RepresentationMethod": "lab.selection.representations",
    "get_representation": "lab.selection.representations",
    "list_representations": "lab.selection.representations",
    "representation_info": "lab.selection.representations",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
