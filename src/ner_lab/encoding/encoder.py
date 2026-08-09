"""Turning a corpus into model-ready training rows."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

from ner_lab.encoding.overlaps import OverlapPolicy, resolve_entities
from ner_lab.encoding.rows import build_row, compute_max_content_length, special_token_template
from ner_lab.encoding.segmentation import (
    merge_sentences_crossing_entities,
    sentence_token_ranges,
    split_into_sentences,
    tokenize_document,
)
from ner_lab.encoding.tagging import build_iob2_labels, build_label_vocabulary
from ner_lab.encoding.windowing import select_context_windows, select_greedy_windows

WindowStrategy = Callable[[list[dict], list[dict], list[dict], str, int], list[dict]]

BUILTIN_STRATEGIES = ("greedy", "context")


class Encoder:
    """
    Encodes a corpus into one row per window, for one entity type and tokenizer.

    Configured once and reused across a split's partitions, so every partition
    shares the same `label2id` and token budget. Labels are used exactly as the
    corpus stores them; normalization is a corpus-build decision.

    `strategy` is `"greedy"`, `"context"`, or any callable taking
    `(sentence_ranges, tokens, entities, text, max_content_length)` and returning
    windows. A window carrying `core_token_start`/`core_token_end` has only that
    core labelled, with the flanks left as unlabelled context.

    `require_target_label` rejects a corpus in which `target_label` never occurs,
    which for training means a typo silently producing all-`O` rows. Inference
    turns it off: unannotated documents legitimately carry no entity at all.
    """

    def __init__(
        self,
        tokenizer,
        target_label: str,
        language: str,
        max_length: int = 256,
        overlap_policy: OverlapPolicy = "merge_same_label_then_keep_longest",
        strategy: str | WindowStrategy = "greedy",
        context_tokens: int | None = None,
        min_sentence_tokens: int = 4,
        documents_per_batch: int = 500,
        require_target_label: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.target_label = target_label
        self.language = language
        self.max_length = max_length
        self.overlap_policy = overlap_policy
        self.strategy = strategy
        self.context_tokens = context_tokens
        self.min_sentence_tokens = min_sentence_tokens
        self.documents_per_batch = documents_per_batch
        self.require_target_label = require_target_label

        self.prefix_ids, self.suffix_ids = special_token_template(tokenizer)
        self.max_content_length = compute_max_content_length(
            self.prefix_ids, self.suffix_ids, max_length
        )

        if self.max_content_length < 1:
            raise ValueError(
                f"max_length={max_length} leaves no room for content after this "
                f"tokenizer's {len(self.prefix_ids) + len(self.suffix_ids)} special tokens."
            )

        self.vocabulary = build_label_vocabulary(target_label)
        self.label2id = self.vocabulary["label2id"]
        self.id2label = self.vocabulary["id2label"]

        self._select_windows = self._resolve_strategy()

    def encode(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Encode an in-memory corpus into one row per window."""
        return self._encode(
            (row.doc_id, row.text, json.loads(row.entities_json))
            for row in corpus[["doc_id", "text", "entities_json"]].itertuples(index=False)
        )

    def encode_parquet(self, path: str | Path) -> pd.DataFrame:
        """
        Encode a corpus parquet, streaming it so a large one is never fully loaded.

        Documents are read in `documents_per_batch`-sized batches.
        """
        parquet_file = pq.ParquetFile(path)

        def documents():
            for batch in parquet_file.iter_batches(
                batch_size=self.documents_per_batch,
                columns=["doc_id", "text", "entities_json"],
            ):
                for document in batch.to_pylist():
                    yield (
                        document["doc_id"],
                        document["text"],
                        json.loads(document["entities_json"]),
                    )

        return self._encode(documents())

    def encode_document(self, doc_id: str, text: str, entities: list[dict]) -> list[dict]:
        """Encode one document into its window rows."""
        entities = resolve_entities(doc_id, entities, text, self.overlap_policy)

        sentences = split_into_sentences(text, self.language)
        sentences = merge_sentences_crossing_entities(sentences, entities, text)

        tokens = tokenize_document(text, self.tokenizer)
        ranges = sentence_token_ranges(sentences, tokens)
        windows = self._select_windows(ranges, tokens, entities, text, self.max_content_length)

        return [self._row(tokens, window, entities, doc_id) for window in windows]

    def _encode(self, documents) -> pd.DataFrame:
        rows: list[dict] = []
        labels_seen: set[str] = set()

        for doc_id, text, entities in documents:
            labels_seen.update(str(entity["label"]) for entity in entities)
            rows.extend(self.encode_document(doc_id, text, entities))

        if self.require_target_label and self.target_label not in labels_seen:
            raise ValueError(
                f"target_label {self.target_label!r} does not occur in this corpus. "
                f"Labels present: {sorted(labels_seen) or 'none'}."
            )

        return pd.DataFrame(rows)

    def _row(self, tokens: list[dict], window: dict, entities: list[dict], doc_id: str) -> dict:
        if "core_token_start" in window:
            labelled = [
                *tokens[window["token_start"]:window["core_token_start"]],
                *build_iob2_labels(
                    tokens[window["core_token_start"]:window["core_token_end"]],
                    entities,
                    self.target_label,
                ),
                *tokens[window["core_token_end"]:window["token_end"]],
            ]
        else:
            labelled = build_iob2_labels(
                tokens[window["token_start"]:window["token_end"]], entities, self.target_label
            )

        row = build_row(labelled, self.label2id, self.prefix_ids, self.suffix_ids)
        row["doc_id"] = doc_id
        row["window_start"] = window["start"]
        row["window_end"] = window["end"]

        return row

    def _resolve_strategy(self) -> WindowStrategy:
        if callable(self.strategy):
            return self.strategy

        if self.strategy not in BUILTIN_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}; expected one of {BUILTIN_STRATEGIES} "
                "or a callable."
            )

        if self.strategy == "greedy":
            if self.context_tokens is not None:
                raise ValueError("context_tokens only applies when strategy='context'.")

            return select_greedy_windows

        if self.context_tokens is None:
            raise ValueError("context_tokens is required when strategy='context'.")

        return partial(
            select_context_windows,
            context_tokens=self.context_tokens,
            min_sentence_tokens=self.min_sentence_tokens,
        )


def describe_encoder(encoder: Encoder) -> dict:
    """
    The settings that decide how an `Encoder` windows a corpus, as plain data.

    Recorded next to a saved model so inference can window its input exactly as
    training did. A custom `strategy` is recorded by name only — a callable is
    code, so `encoder_from_description` needs it handed back.
    """
    return {
        "target_label": encoder.target_label,
        "language": encoder.language,
        "max_length": encoder.max_length,
        "strategy": strategy_name(encoder.strategy),
        "context_tokens": encoder.context_tokens,
        "overlap_policy": encoder.overlap_policy,
        "min_sentence_tokens": encoder.min_sentence_tokens,
        "label2id": dict(encoder.label2id),
    }


def encoder_from_description(
    description: dict,
    tokenizer,
    strategy: str | WindowStrategy | None = None,
    **overrides,
) -> Encoder:
    """
    Rebuild an `Encoder` from `describe_encoder`'s output.

    `label2id` is dropped: the vocabulary is derived from `target_label`, and
    rebuilding it is what proves the description and the model agree. Pass
    `strategy` when the description names a callable this library cannot import.
    """
    settings = {key: value for key, value in description.items() if key != "label2id"}
    named = settings.pop("strategy", "greedy")

    if strategy is None and named not in BUILTIN_STRATEGIES:
        raise ValueError(
            f"strategy {named!r} is not one of {BUILTIN_STRATEGIES}, so it was a callable "
            "when this model was trained. Pass strategy= to supply it again."
        )

    return Encoder(
        tokenizer=tokenizer,
        strategy=strategy if strategy is not None else named,
        **{**settings, **overrides},
    )


def strategy_name(strategy: str | WindowStrategy) -> str:
    """A recordable name for a builtin strategy or a user-supplied callable."""
    if isinstance(strategy, str):
        return strategy

    return getattr(strategy, "__name__", type(strategy).__name__)
