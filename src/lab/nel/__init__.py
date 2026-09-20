"""Normalisation: linking a span table to an ontology through matching, retrieval and reranking."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "Concept": "lab.nel.schemas",
    "EntityLinkingPipeline": "lab.nel.pipeline",
    "GazetteerEntry": "lab.nel.schemas",
    "HierarchyEdge": "lab.nel.schemas",
    "LinkedEntity": "lab.nel.schemas",
    "LinkingResult": "lab.nel.linking",
    "MatchCandidate": "lab.nel.schemas",
    "MentionAnnotation": "lab.nel.schemas",
    "build_matcher": "lab.nel.matching",
    "link_entities": "lab.nel.linking",
    "reciprocal_rank_fusion": "lab.nel.rrf",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
