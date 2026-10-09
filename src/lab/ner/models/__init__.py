"""Token-classification architectures and the factory that builds them."""

from __future__ import annotations

from lab.ner.models.bio import (
    bio_constraint_masks,
    bio_entity,
    is_inside_label,
    normalize_id2label,
    normalize_label2id,
)
from lab.ner.models.capacity import check_max_length
from lab.ner.models.registry import (
    BUILTIN_ARCHITECTURES,
    Architecture,
    architecture_name,
    build_linear_model,
    build_model,
)

__all__ = [
    "BUILTIN_ARCHITECTURES",
    "Architecture",
    "architecture_name",
    "bio_constraint_masks",
    "bio_entity",
    "check_max_length",
    "build_linear_model",
    "build_model",
    "is_inside_label",
    "normalize_id2label",
    "normalize_label2id",
]
