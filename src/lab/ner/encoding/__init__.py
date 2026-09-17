"""Turning a corpus into model-ready rows: overlap resolution, windowing, tagging."""

from __future__ import annotations

from lab.ner.encoding.encoder import (
    BUILTIN_STRATEGIES,
    Encoder,
    WindowStrategy,
    describe_encoder,
    encoder_from_description,
    strategy_name,
)
from lab.ner.encoding.overlaps import OverlapPolicy, resolve_entities
from lab.ner.encoding.rows import (
    IGNORE_INDEX,
    build_row,
    compute_max_content_length,
    special_token_template,
)
from lab.ner.encoding.tagging import build_iob2_labels, build_label_vocabulary
from lab.ner.encoding.windowing import (
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
    "resolve_entities",
    "select_context_windows",
    "select_greedy_windows",
    "special_token_template",
    "split_oversized_sentence",
    "strategy_name",
]
