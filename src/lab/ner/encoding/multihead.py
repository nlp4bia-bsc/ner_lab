"""Training-only head discovery and per-label overlap resolution."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from lab.core.corpus import count_labels
from lab.ner.encoding.overlaps import resolve_entities


def discover_entity_labels(corpus: pd.DataFrame) -> list[str]:
    """Return sorted labels from authorized training annotations only."""
    labels = sorted(count_labels(corpus))
    if not labels:
        raise ValueError("Cannot discover multihead entities: training corpus has no annotations.")

    return labels


def training_entity_labels(paths: Path | list[Path]) -> list[str]:
    """Discover heads from training partition paths, never validation or holdout."""
    paths = paths if isinstance(paths, list) else [paths]

    return discover_entity_labels(pd.concat([pd.read_parquet(path) for path in paths]))


def resolve_head_entities(doc_id: str, entities: list[dict], text: str, policy: str) -> list[dict]:
    """Resolve intraclass overlaps independently; cross-label overlaps survive."""
    if policy == "none":
        raise ValueError(
            "Multihead cannot represent intraclass overlaps; use overlap_policy='error'."
        )
    resolved = []
    for label in sorted({entity["label"] for entity in entities}):
        same = [entity for entity in entities if entity["label"] == label]
        resolved.extend(resolve_entities(doc_id, same, text, policy))
    identity = lambda entity: (entity["start"], entity["end"], entity["label"])
    if sorted(map(identity, resolved)) != sorted(map(identity, entities)):
        warnings.warn(
            f"Document {doc_id!r}: intraclass overlap policy {policy!r} applied; "
            f"{len(entities)} annotations became {len(resolved)}.",
            stacklevel=2,
        )

    return sorted(resolved, key=lambda entity: (entity["start"], entity["end"], entity["label"]))
