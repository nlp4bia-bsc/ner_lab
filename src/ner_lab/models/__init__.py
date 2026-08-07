"""Token-classification architectures and the factory that builds them."""

from __future__ import annotations

from ner_lab.models.bio import (
    bio_constraint_masks,
    bio_entity,
    is_inside_label,
    normalize_id2label,
    normalize_label2id,
)
from ner_lab.models.registry import (
    BUILTIN_ARCHITECTURES,
    Architecture,
    build_linear_model,
    build_model,
)

__all__ = [
    "BUILTIN_ARCHITECTURES",
    "Architecture",
    "bio_constraint_masks",
    "bio_entity",
    "build_linear_model",
    "build_model",
    "is_inside_label",
    "normalize_id2label",
    "normalize_label2id",
]
