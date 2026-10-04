"""The text table: documents, sentences, tokens, vocabulary and lexical diversity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from lab.core.segmentation import (
    sentence_token_ranges,
    split_into_sentences,
    tokenize_document,
)
from lab.core.stats.primitives import (
    DEFAULT_MATTR_WINDOW,
    load_tokenizer,
    mattr,
    quantiles,
    words_of,
    write_stats,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class TextMeasurements:
    """
    Raw text measurements, accumulated document by document.

    MATTR is accumulated as a running length-weighted sum, so the corpus figure is a
    mean over tokens rather than over documents, and no corpus-wide token stream is
    ever held in memory.
    """

    sentence_counts: list[int] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    word_counts: list[int] = field(default_factory=list)
    sentence_lengths: list[int] = field(default_factory=list)
    word_types: set[str] = field(default_factory=set)
    subword_types: set[int] = field(default_factory=set)
    weighted_word_mattr: float = 0.0
    weighted_subword_mattr: float = 0.0

    def add(
        self, text: str, language: str, tokenizer: PreTrainedTokenizerBase, window: int
    ) -> None:
        tokens = tokenize_document(text, tokenizer)
        token_ids = [token["token_id"] for token in tokens]
        words = words_of(text)
        spans = sentence_token_ranges(split_into_sentences(text, language), tokens)

        self.sentence_counts.append(len(spans))
        self.token_counts.append(len(tokens))
        self.word_counts.append(len(words))
        self.sentence_lengths += [span["token_end"] - span["token_start"] for span in spans]

        self.word_types.update(words)
        self.subword_types.update(token_ids)

        self.weighted_word_mattr += mattr(words, window) * len(words)
        self.weighted_subword_mattr += mattr(token_ids, window) * len(token_ids)

    def row(self, language: str) -> dict:
        tokens = sum(self.token_counts)
        words = sum(self.word_counts)

        return {
            "language": language,
            "documents": len(self.token_counts),
            "sentences": sum(self.sentence_counts),
            "tokens": tokens,
            "words": words,
            "vocabulary_words": len(self.word_types),
            "vocabulary_subwords": len(self.subword_types),
            **quantiles("sentences_per_document", self.sentence_counts, (25, 50, 75)),
            **quantiles("tokens_per_document", self.token_counts, (5, 25, 50, 75, 95)),
            **quantiles("tokens_per_sentence", self.sentence_lengths, (25, 50, 75)),
            "mattr_words": self.weighted_word_mattr / words if words else float("nan"),
            "mattr_subwords": self.weighted_subword_mattr / tokens if tokens else float("nan"),
        }


def compute_text_stats(
    frame: pd.DataFrame,
    base_model: str,
    language: str,
    *,
    mattr_window: int = DEFAULT_MATTR_WINDOW,
    output_dir: str | Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Text-corpus metrics: documents, sentences, tokens, vocabulary, length
    distributions and lexical diversity, as a single-row DataFrame.

    Reads `text` only; the entity layer is ignored. Pass `output_dir` to also write
    `text_stats.json` / `text_stats.parquet` there.
    """
    tokenizer = load_tokenizer(base_model)
    measurements = TextMeasurements()

    if progress:
        print(f"    text stats: {len(frame):,} documents", flush=True)

    for text in frame["text"]:
        measurements.add(text or "", language, tokenizer, mattr_window)

    result = pd.DataFrame([measurements.row(language)])

    if output_dir is not None:
        write_stats(result, output_dir, "text_stats")

    return result
