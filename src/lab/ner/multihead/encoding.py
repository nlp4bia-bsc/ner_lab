"""Turning a corpus into rows labelled once per entity type, over windows the types share."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from lab.core.corpus import count_labels
from lab.core.io import read_corpus
from lab.core.sentences import describe_sentence_segmenter
from lab.ner.encoding.encoder import (
    BUILTIN_STRATEGIES,
    WindowStrategy,
    read_parquet_documents,
    resolve_window_strategy,
    strategy_name,
    window_document,
)
from lab.ner.encoding.overlaps import OverlapPolicy, resolve_entities
from lab.ner.encoding.rows import (
    IGNORE_INDEX,
    build_label_ids,
    build_row,
    compute_max_content_length,
    special_token_template,
)
from lab.ner.encoding.tagging import build_label_vocabulary, label_window

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class MultiHeadEncoder:
    """
    Encodes a corpus into one row per window, labelled for every entity type at once.

    `entity_labels` is the head catalogue: one head per type, in this order, which the
    model's heads follow. Windows, tokens and inputs are built exactly as `Encoder`
    builds them; `labels` holds one id per head at each position, `[position][head]`,
    each head numbering its type's tags as `build_label_vocabulary` does (`O=0`,
    `B=1`, `I=2`), with `IGNORE_INDEX` wherever `Encoder` would put it.

    `overlap_policy` resolves overlapping annotations within each type, as a
    single-label `Encoder` for that type would. Annotations of different types are
    never resolved against each other: overlapping across types is what the heads
    are for. Entities of a type with no head are left unlabelled, with a warning.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        entity_labels: Sequence[str],
        language: str,
        max_length: int = 256,
        overlap_policy: OverlapPolicy = "merge_same_label_then_keep_longest",
        strategy: str | WindowStrategy = "greedy",
        context_tokens: int | None = None,
        min_sentence_tokens: int = 4,
        documents_per_batch: int = 500,
    ) -> None:
        self.tokenizer = tokenizer
        self.entity_labels = validate_entity_labels(entity_labels)
        self.language = language
        self.max_length = max_length
        self.overlap_policy = overlap_policy
        self.strategy = strategy
        self.context_tokens = context_tokens
        self.min_sentence_tokens = min_sentence_tokens
        self.documents_per_batch = documents_per_batch

        self.prefix_ids, self.suffix_ids = special_token_template(tokenizer)
        self.max_content_length = compute_max_content_length(
            self.prefix_ids, self.suffix_ids, max_length
        )

        if self.max_content_length < 1:
            raise ValueError(
                f"max_length={max_length} leaves no room for content after this "
                f"tokenizer's {len(self.prefix_ids) + len(self.suffix_ids)} special tokens."
            )

        self.head_label2ids = [
            build_label_vocabulary(label)["label2id"] for label in self.entity_labels
        ]
        self._select_windows = resolve_window_strategy(
            strategy, context_tokens, min_sentence_tokens
        )

    def encode(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Encode an in-memory corpus into one row per window."""
        return self._encode(
            (row.doc_id, row.text, json.loads(row.entities_json))
            for row in corpus[["doc_id", "text", "entities_json"]].itertuples(index=False)
        )

    def encode_parquet(self, path: str | Path) -> pd.DataFrame:
        """Encode a corpus parquet, streamed `documents_per_batch` documents at a time."""
        return self._encode(read_parquet_documents(path, self.documents_per_batch))

    def encode_document(self, doc_id: str, text: str, entities: list[dict]) -> list[dict]:
        """Encode one document into its window rows."""
        entities = resolve_entities_by_type(doc_id, entities, text, self.overlap_policy)

        tokens, windows = window_document(
            text=text,
            entities=entities,
            tokenizer=self.tokenizer,
            language=self.language,
            select_windows=self._select_windows,
            max_content_length=self.max_content_length,
        )

        return [self._row(tokens, window, entities, doc_id) for window in windows]

    def _encode(self, documents: Iterable[tuple[str, str, list[dict]]]) -> pd.DataFrame:
        rows: list[dict] = []
        headless: Counter[str] = Counter()
        heads = set(self.entity_labels)

        for doc_id, text, entities in documents:
            headless.update(
                str(entity["label"]) for entity in entities if str(entity["label"]) not in heads
            )
            rows.extend(self.encode_document(doc_id, text, entities))

        if headless:
            warnings.warn(
                f"{sum(headless.values())} entities of {len(headless)} type(s) with no head "
                f"were left unlabelled: {dict(headless.most_common(10))}. Add the types to "
                "entity_labels to train heads for them.",
                stacklevel=3,
            )

        return pd.DataFrame(rows)

    def _row(self, tokens: list[dict], window: dict, entities: list[dict], doc_id: str) -> dict:
        head_label_ids = []

        for label, label2id in zip(self.entity_labels, self.head_label2ids, strict=True):
            labelled = label_window(tokens, window, entities, label)
            head_label_ids.append(build_label_ids(labelled, label2id))

        unlabelled = [IGNORE_INDEX] * len(self.entity_labels)
        window_tokens = tokens[window["token_start"] : window["token_end"]]

        row = build_row(window_tokens, {}, self.prefix_ids, self.suffix_ids)
        row["labels"] = [
            *([unlabelled] * len(self.prefix_ids)),
            *(list(position) for position in zip(*head_label_ids, strict=True)),
            *([unlabelled] * len(self.suffix_ids)),
        ]
        row["doc_id"] = doc_id
        row["window_start"] = window["start"]
        row["window_end"] = window["end"]

        return row


def resolve_entities_by_type(
    doc_id: str, entities: list[dict], text: str, policy: OverlapPolicy
) -> list[dict]:
    """
    Resolve overlapping annotations within each entity type, never across types.

    Each type's annotations go through `resolve_entities` on their own, so a type keeps
    exactly the entities a single-label encoder for it would keep.
    """
    by_type: dict[str, list[dict]] = {}

    for entity in entities:
        by_type.setdefault(str(entity["label"]), []).append(entity)

    resolved = [
        entity
        for label in sorted(by_type)
        for entity in resolve_entities(doc_id, by_type[label], text, policy)
    ]

    return sorted(resolved, key=lambda entity: (entity["start"], entity["end"], entity["label"]))


def validate_entity_labels(entity_labels: Sequence[str]) -> list[str]:
    """Check a head catalogue is a non-empty list of distinct non-empty strings, kept in order."""
    labels = list(entity_labels)

    if not labels:
        raise ValueError("entity_labels is empty; a multihead model needs at least one head.")

    invalid = [label for label in labels if not isinstance(label, str) or not label.strip()]

    if invalid:
        raise ValueError(f"entity_labels must be non-empty strings; received {invalid[:10]!r}.")

    duplicates = sorted({label for label in labels if labels.count(label) > 1})

    if duplicates:
        raise ValueError(f"entity_labels repeats {duplicates[:10]!r}; each type is one head.")

    return labels


def entity_labels_from_partitions(paths: Sequence[str | Path]) -> list[str]:
    """
    Every entity type annotated in the given corpus parquets, sorted.

    The default head catalogue: given every non-holdout partition of a split, each
    fold of a run trains the same heads.
    """
    labels: set[str] = set()

    for path in paths:
        labels.update(count_labels(read_corpus(path)))

    if not labels:
        raise ValueError(
            f"No entity is annotated in {[str(path) for path in paths][:10]}, so there is "
            "no head to train."
        )

    return sorted(labels)


def describe_multihead_encoder(encoder: MultiHeadEncoder) -> dict[str, Any]:
    """
    The settings that decide how a `MultiHeadEncoder` labels and windows, as plain data.

    The `describe_encoder` counterpart, recorded beside a saved model so inference
    windows its input exactly as training did.
    """
    return {
        "entity_labels": list(encoder.entity_labels),
        "language": encoder.language,
        "sentence_segmenter": describe_sentence_segmenter(encoder.language),
        "max_length": encoder.max_length,
        "strategy": strategy_name(encoder.strategy),
        "context_tokens": encoder.context_tokens,
        "overlap_policy": encoder.overlap_policy,
        "min_sentence_tokens": encoder.min_sentence_tokens,
    }


def multihead_encoder_from_description(
    description: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    strategy: str | WindowStrategy | None = None,
) -> MultiHeadEncoder:
    """
    Rebuild a `MultiHeadEncoder` from `describe_multihead_encoder`'s output.

    Pass `strategy` when the description names a callable this library cannot import.
    """
    settings = {key: value for key, value in description.items() if key != "sentence_segmenter"}
    named = settings.pop("strategy", "greedy")

    if strategy is None and named not in BUILTIN_STRATEGIES:
        raise ValueError(
            f"strategy {named!r} is not one of {BUILTIN_STRATEGIES}, so it was a callable "
            "when this model was trained. Pass strategy= to supply it again."
        )

    return MultiHeadEncoder(
        tokenizer=tokenizer, strategy=strategy if strategy is not None else named, **settings
    )
