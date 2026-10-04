"""
Corpus statistics for a single-language corpus.

Two independent entry points, one per metric table:

    compute_text_stats(frame, base_model, language)  size and shape of the raw documents
    compute_annotation_stats(frame, base_model)       size and shape of the entity layer

Both read the canonical `text` / `entities_json` columns `lab.core.read_corpus` returns
and each return a single-row DataFrame; neither needs the other, and each loads its own
tokenizer from `base_model`. Pass `output_dir` to either to also write it as
`<stem>.json` / `<stem>.parquet`, via the public `write_stats`.

The units:

tokens     Subwords, via `lab.core.tokenize_document` -- the counts the encoding stage
           actually sees.
sentences  pysbd, via `lab.core.split_into_sentences` -- the same segmenter `Encoder` uses.
words      Unicode `\\w+` runs, lowercased. Used only for vocabulary and MATTR, where
           subword types would measure the tokenizer rather than the corpus. Both word-
           and subword-level figures are reported.
"""

from __future__ import annotations

from lab.core.stats.annotations import (
    AnnotationMeasurements,
    compute_annotation_stats,
    mention_token_length,
    span_relations,
)
from lab.core.stats.primitives import (
    DEFAULT_MATTR_WINDOW,
    entities_of,
    load_tokenizer,
    mattr,
    quantiles,
    share,
    words_of,
    write_stats,
)
from lab.core.stats.text import TextMeasurements, compute_text_stats

__all__ = [
    "DEFAULT_MATTR_WINDOW",
    "AnnotationMeasurements",
    "TextMeasurements",
    "compute_annotation_stats",
    "compute_text_stats",
    "entities_of",
    "load_tokenizer",
    "mattr",
    "mention_token_length",
    "quantiles",
    "share",
    "span_relations",
    "words_of",
    "write_stats",
]
