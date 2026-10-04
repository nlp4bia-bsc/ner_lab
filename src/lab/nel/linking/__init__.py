"""Linking a span table to an ontology: the `nel.link_entities` task and the pieces it runs."""

from __future__ import annotations

from lab.nel.linking.link import LinkingResult, link_entities
from lab.nel.linking.methods import (
    ENCODER_METHODS,
    METHODS,
    EncoderRetriever,
    IndexSource,
    build_candidate_generator,
    build_reranker,
    resolve_index_dir,
    transformer_faiss_settings,
)
from lab.nel.linking.scoring import score_linking
from lab.nel.linking.tables import (
    GAZETTEER_COLUMNS,
    GOLD_COLUMN,
    LINKED_COLUMNS,
    SOURCE_COLUMN,
    candidate_record,
    completed_frame,
    gazetteer_entries,
    linked_frame,
    mentions_from_spans,
    read_gazetteer,
    read_spans,
)
