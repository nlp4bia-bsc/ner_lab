"""The records matching, retrieval, reranking and evaluation pass between them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Mention:
    """One entity mention and its gold code, if it has one."""

    filename: str
    label: str | None
    start_span: int | None
    end_span: int | None
    text: str
    code: str | None
    is_abbreviation: bool | None = None
    is_composite: bool | None = None
    need_context: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GazetteerEntry:
    """One terminology or gazetteer entry."""

    term: str
    code: str
    label: str | None = None
    semantic_tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Concept:
    """An ontology concept with its preferred term and any aliases."""

    code: str
    term: str
    label: str | None = None
    semantic_tag: str | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class HierarchyEdge:
    """A directed parent-to-child relation in an ontology."""

    parent_code: str
    child_code: str
    relation: str | None = None


@dataclass
class Candidate:
    """A concept proposed for a mention by a matcher, retriever or reranker."""

    mention_id: str | None
    filename: str | None
    text: str
    label: str | None
    code: str | None
    term: str | None
    score: float
    method: str
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkedEntity:
    """The linking result for one mention: its ranked candidates and the top prediction."""

    mention: Mention
    candidates: list[Candidate]
    predicted_code: str | None
    predicted_term: str | None
    score: float | None
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


def labels_compatible(mention_label: str | None, entry_label: str | None) -> bool:
    """Whether an entry may be proposed for a mention: true unless both are labelled differently."""
    return not (mention_label and entry_label and mention_label != entry_label)
