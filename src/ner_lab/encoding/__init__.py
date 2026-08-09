"""Turning a corpus into model-ready rows: overlap resolution, windowing, tagging."""

from __future__ import annotations

from ner_lab.encoding.encoder import (
    BUILTIN_STRATEGIES,
    Encoder,
    WindowStrategy,
    describe_encoder,
    encoder_from_description,
    strategy_name,
)
from ner_lab.encoding.overlaps import OverlapPolicy, resolve_entities
from ner_lab.encoding.rows import (
    IGNORE_INDEX,
    build_row,
    compute_max_content_length,
    special_token_template,
)
from ner_lab.encoding.segmentation import (
    merge_ranges,
    merge_sentences_crossing_entities,
    merge_short_sentences,
    sentence_token_ranges,
    split_into_sentences,
    tokenize_document,
)
from ner_lab.encoding.tagging import build_iob2_labels, build_label_vocabulary
from ner_lab.encoding.windowing import (
    build_window,
    select_context_windows,
    select_greedy_windows,
    split_oversized_sentence,
)

__all__ = [
    "BUILTIN_STRATEGIES",
    "IGNORE_INDEX",
    "Encoder",
    "OverlapPolicy",
    "WindowStrategy",
    "build_iob2_labels",
    "build_label_vocabulary",
    "build_row",
    "build_window",
    "compute_max_content_length",
    "describe_encoder",
    "encoder_from_description",
    "merge_ranges",
    "merge_sentences_crossing_entities",
    "merge_short_sentences",
    "resolve_entities",
    "select_context_windows",
    "select_greedy_windows",
    "sentence_token_ranges",
    "special_token_template",
    "split_into_sentences",
    "split_oversized_sentence",
    "strategy_name",
    "tokenize_document",
]
