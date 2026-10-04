"""Cross-encoder reranking: scoring mention-candidate pairs, and training the scorer."""

from __future__ import annotations

from lab.nel.cross_encoder.examples import (
    pairs_to_labeled_texts,
    prepare_triplets,
    transform_triplets_rankingeval,
    triplets_to_knowledge_graph_labeled_texts,
    triplets_to_labeled_texts,
    triplets_to_texts,
)
from lab.nel.cross_encoder.reranker import CrossEncoderReranker
from lab.nel.cross_encoder.training import (
    DEFAULT_TRAINING_LOGGING_STEPS,
    configure_quiet_transformers_logging,
    resolve_cross_encoder_dataloader_num_workers,
    resolve_cross_encoder_dataloader_pin_memory,
    supported_kwargs,
    training_dataloader,
)

__all__ = [
    "DEFAULT_TRAINING_LOGGING_STEPS",
    "CrossEncoderReranker",
    "configure_quiet_transformers_logging",
    "pairs_to_labeled_texts",
    "prepare_triplets",
    "resolve_cross_encoder_dataloader_num_workers",
    "resolve_cross_encoder_dataloader_pin_memory",
    "supported_kwargs",
    "training_dataloader",
    "transform_triplets_rankingeval",
    "triplets_to_knowledge_graph_labeled_texts",
    "triplets_to_labeled_texts",
    "triplets_to_texts",
]
