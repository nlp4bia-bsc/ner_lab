"""What both stats tables share: the tokenizer, words, MATTR, quantiles, and writing a table."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import regex

from lab.core.io import DEFAULT_PARQUET_COMPRESSION
from lab.core.labels import LABEL_ALIASES, normalize_entity_labels
from lab.core.segmentation import split_into_words

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_MATTR_WINDOW = 100

LETTER_OR_DIGIT = regex.compile(r"[\p{L}\p{N}]")
WHITESPACE = re.compile(r"\s+")


def write_stats(frame: pd.DataFrame, output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Write a stats DataFrame as `<stem>.json` and `<stem>.parquet`, returning both paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, parquet_path = directory / f"{stem}.json", directory / f"{stem}.parquet"

    frame.to_json(json_path, orient="records", indent=2, force_ascii=False)
    frame.to_parquet(
        parquet_path, engine="pyarrow", compression=DEFAULT_PARQUET_COMPRESSION, index=False
    )

    return json_path, parquet_path


def load_tokenizer(base_model: str) -> PreTrainedTokenizerBase:
    """
    The fast tokenizer of `base_model`, with its sequence-length limit lifted.

    Documents are tokenized whole and never fed to the model, so the "longer than
    the maximum sequence length" warning would be noise here.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if not tokenizer.is_fast:
        raise ValueError(f"{base_model!r} has no fast tokenizer; offsets drive every token metric.")

    tokenizer.model_max_length = int(1e9)

    return tokenizer


def words_of(text: str) -> list[str]:
    """
    The lowercased `split_into_words` words of `text` that hold a letter or a digit.

    The same UAX #29 words the encoder labels by; a word of punctuation alone is left
    out, as ICU's word break iterator marks it as no word.
    """
    return [
        word["text"].lower()
        for word in split_into_words(text)
        if LETTER_OR_DIGIT.search(word["text"])
    ]


def entities_of(raw: object, normalize_labels: bool = False) -> list[dict]:
    """Decode an `entities_json` cell, optionally folding aliases via `LABEL_ALIASES`."""
    entities = json.loads(str(raw))

    return normalize_entity_labels(entities, LABEL_ALIASES) if normalize_labels else entities


def mattr(sequence: Sequence, window: int) -> float:
    """
    Moving-average type-token ratio: the mean TTR over every window-sized slice.

    Sequences shorter than the window have no slice to average, so their plain TTR
    is used, which keeps short documents in the figure instead of dropping them.
    """
    length = len(sequence)

    if length == 0:
        return float("nan")

    if length <= window:
        return len(set(sequence)) / length

    counts = Counter(sequence[:window])
    types_seen = len(counts)

    for index in range(window, length):
        leaving = sequence[index - window]
        counts[leaving] -= 1

        if not counts[leaving]:
            del counts[leaving]

        counts[sequence[index]] += 1
        types_seen += len(counts)

    return types_seen / ((length - window + 1) * window)


def quantiles(prefix: str, values: Sequence[float], points: Sequence[int]) -> dict[str, float]:
    """Named percentile and mean fields, e.g. `tokens_per_document_p50` and `_mean`."""
    if not values:
        return {f"{prefix}_p{point}": float("nan") for point in points} | {
            f"{prefix}_mean": float("nan")
        }

    return {
        f"{prefix}_p{point}": float(value)
        for point, value in zip(points, np.percentile(values, points))
    } | {f"{prefix}_mean": float(np.mean(values))}


def share(count: int, total: int) -> float:
    """Percentage, or NaN when there is nothing to take a share of."""
    return count / total * 100 if total else float("nan")
