"""Turning a corpus into model-ready training rows."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lab.core.segmentation import (
    merge_sentences_crossing_entities,
    sentence_token_ranges,
    tokenize_document,
)
from lab.core.sentences import describe_sentence_segmenter, split_into_sentences
from lab.ner.encoding.overlaps import OverlapPolicy, resolve_entities
from lab.ner.encoding.rows import (
    IGNORE_INDEX,
    build_label_ids,
    build_row,
    compute_max_content_length,
    special_token_template,
)
from lab.ner.encoding.tagging import build_iob2_labels, build_label_vocabulary
from lab.ner.encoding.windowing import select_context_windows, select_greedy_windows

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

WindowStrategy = Callable[[list[dict], list[dict], list[dict], str, int], list[dict]]

BUILTIN_STRATEGIES = ("greedy", "context")


class Encoder:
    """
    Encodes a corpus into one row per window, sharing inputs across entity types.

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

    `target_label=None` discovers a sorted head catalogue on the first `encode`
    call, which must use training data. Reuse that encoder for validation and
    inference; a later corpus never adds heads. `entity_labels` supplies an
    already-discovered catalogue explicitly. Multihead overlaps resolve per type.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        target_label: str | None,
        language: str,
        max_length: int = 256,
        overlap_policy: OverlapPolicy | None = None,
        strategy: str | WindowStrategy = "greedy",
        context_tokens: int | None = None,
        min_sentence_tokens: int = 4,
        documents_per_batch: int = 500,
        require_target_label: bool = True,
        entity_labels: list[str] | None = None,
        include_word_mask: bool = False,
    ) -> None:
        self.include_word_mask = include_word_mask
        self.tokenizer = tokenizer
        self.target_label = target_label
        self.language = language
        self.max_length = max_length
        self.entity_labels = list(entity_labels) if entity_labels is not None else None
        self.overlap_policy = (
            ("error" if target_label is None else "merge_same_label_then_keep_longest")
            if overlap_policy is None
            else overlap_policy
        )
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

        if target_label is None:
            self.vocabulary = {
                "label2id": {"O": 0, "B": 1, "I": 2},
                "id2label": {0: "O", 1: "B", 2: "I"},
            }
        else:
            if entity_labels is not None:
                raise ValueError("entity_labels requires target_label=None.")
            self.vocabulary = build_label_vocabulary(target_label)
        self.label2id = self.vocabulary["label2id"]
        self.id2label = self.vocabulary["id2label"]

        self._select_windows = self._resolve_strategy()

    def encode(self, corpus: pd.DataFrame) -> pd.DataFrame:
        """Encode an in-memory corpus into one row per window."""
        if self.target_label is None and self.entity_labels is None:
            from lab.ner.encoding.multihead import discover_entity_labels

            self.entity_labels = discover_entity_labels(corpus)
        result = self._encode(
            (row.doc_id, row.text, json.loads(row.entities_json))
            for row in corpus[["doc_id", "text", "entities_json"]].itertuples(index=False)
        )
        if self.target_label is None:
            from lab.core.spans import spans_from_corpus

            result.attrs["multihead_gold"] = spans_from_corpus(corpus).to_dict("records")
            result.attrs["entity_labels"] = list(self.entity_labels)

        return result

    def encode_parquet(self, path: str | Path) -> pd.DataFrame:
        """
        Encode a corpus parquet, streaming it so a large one is never fully loaded.

        Documents are read in `documents_per_batch`-sized batches.
        """
        if self.target_label is None:
            return self.encode(pd.read_parquet(path))

        parquet_file = pq.ParquetFile(path)

        return self._encode(_parquet_documents(parquet_file, self.documents_per_batch))

    def encode_document(self, doc_id: str, text: str, entities: list[dict]) -> list[dict]:
        """Encode one document into its window rows."""
        if self.target_label is None:
            from lab.ner.encoding.multihead import resolve_head_entities
            from lab.ner.encoding.tagging import validate_entity_labels

            self.entity_labels = validate_entity_labels(self.entity_labels or [])
            entities = resolve_head_entities(doc_id, entities, text, self.overlap_policy)
        else:
            if self.include_word_mask:
                entities = [entity for entity in entities if entity["label"] == self.target_label]
            entities = resolve_entities(doc_id, entities, text, self.overlap_policy)

        sentences = split_into_sentences(text, self.language)
        window_entities = [] if self.include_word_mask else entities
        sentences = merge_sentences_crossing_entities(sentences, window_entities, text)

        tokens = tokenize_document(text, self.tokenizer)
        ranges = sentence_token_ranges(sentences, tokens)
        windows = self._select_windows(
            ranges, tokens, window_entities, text, self.max_content_length
        )

        rows = [self._row(tokens, window, entities, doc_id) for window in windows]
        if self.include_word_mask:
            for row in rows:
                labels = np.asarray(row["labels"])
                eligible = labels if labels.ndim == 1 else labels[:, 0]
                row["crf_word_mask"] = (eligible != -100).tolist()
        return rows

    def _encode(self, documents: Iterable[tuple[str, str, list[dict]]]) -> pd.DataFrame:
        rows: list[dict] = []
        labels_seen: set[str] = set()

        for doc_id, text, entities in documents:
            labels_seen.update(str(entity["label"]) for entity in entities)
            rows.extend(self.encode_document(doc_id, text, entities))

        if self.target_label is None:
            unknown = sorted(labels_seen - set(self.entity_labels or []))
            if unknown:
                warnings.warn(
                    f"Unknown entity labels {unknown[:10]} have no heads; gold remains in scoring.",
                    stacklevel=2,
                )
        elif self.require_target_label and self.target_label not in labels_seen:
            raise ValueError(
                f"target_label {self.target_label!r} does not occur in this corpus. "
                f"Labels present: {sorted(labels_seen) or 'none'}."
            )

        return pd.DataFrame(rows)

    def _row(self, tokens: list[dict], window: dict, entities: list[dict], doc_id: str) -> dict:
        if self.target_label is None:
            return self._multihead_row(tokens, window, entities, doc_id)
        if "core_token_start" in window:
            labelled = [
                *tokens[window["token_start"] : window["core_token_start"]],
                *build_iob2_labels(
                    tokens[window["core_token_start"] : window["core_token_end"]],
                    entities,
                    self.target_label,
                ),
                *tokens[window["core_token_end"] : window["token_end"]],
            ]
        else:
            labelled = build_iob2_labels(
                tokens[window["token_start"] : window["token_end"]], entities, self.target_label
            )

        row = build_row(labelled, self.label2id, self.prefix_ids, self.suffix_ids)
        row["doc_id"] = doc_id
        row["window_start"] = window["start"]
        row["window_end"] = window["end"]

        return row

    def _multihead_row(
        self, tokens: list[dict], window: dict, entities: list[dict], doc_id: str
    ) -> dict:
        start, end = window["token_start"], window["token_end"]
        core_start = window.get("core_token_start", start)
        core_end = window.get("core_token_end", end)
        labels = []
        row = None
        for label in self.entity_labels:
            labelled = [
                *tokens[start:core_start],
                *build_iob2_labels(tokens[core_start:core_end], entities, label),
                *tokens[core_end:end],
            ]
            vocabulary = build_label_vocabulary(label)["label2id"]
            head_labels = build_label_ids(labelled, vocabulary)
            labels.append(
                [IGNORE_INDEX] * len(self.prefix_ids)
                + head_labels
                + [IGNORE_INDEX] * len(self.suffix_ids)
            )
            if row is None:
                row = build_row(labelled, vocabulary, self.prefix_ids, self.suffix_ids)
        row["labels"] = np.asarray(labels, dtype=np.int64).T.tolist()
        row.update(doc_id=doc_id, window_start=window["start"], window_end=window["end"])

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
    code, so `encoder_from_description` needs it handed back. The sentence segmenter is
    recorded for provenance only: rebuilding picks it from `language` again.
    """
    return {
        **({"entity_labels": encoder.entity_labels} if encoder.target_label is None else {}),
        **({"include_word_mask": True} if encoder.include_word_mask else {}),
        "target_label": encoder.target_label,
        "language": encoder.language,
        "sentence_segmenter": describe_sentence_segmenter(encoder.language),
        "max_length": encoder.max_length,
        "strategy": strategy_name(encoder.strategy),
        "context_tokens": encoder.context_tokens,
        "overlap_policy": encoder.overlap_policy,
        "min_sentence_tokens": encoder.min_sentence_tokens,
        "label2id": dict(encoder.label2id),
    }


def encoder_from_description(
    description: dict,
    tokenizer: PreTrainedTokenizerBase,
    strategy: str | WindowStrategy | None = None,
    **overrides: Any,
) -> Encoder:
    """
    Rebuild an `Encoder` from `describe_encoder`'s output.

    `label2id` is dropped: the vocabulary is derived from `target_label`, and
    rebuilding it is what proves the description and the model agree. Pass
    `strategy` when the description names a callable this library cannot import.
    """
    recorded = ("label2id", "sentence_segmenter")
    settings = {key: value for key, value in description.items() if key not in recorded}
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


def encode_partition(
    encoder: Encoder,
    path_or_paths: Path | list[Path],
    cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Encode one partition, or concatenate several into one training frame.

    `cache` keeps each fold parquet's rows keyed by path, so a fold that is train
    in one rotation and validation in another is encoded once.
    """
    paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    frames = []

    for path in paths:
        key = str(path)

        if cache is None:
            frames.append(encoder.encode_parquet(path))
        else:
            if key not in cache:
                cache[key] = encoder.encode_parquet(path)

            frames.append(cache[key])

    if len(frames) == 1:
        return frames[0].reset_index(drop=True)

    if encoder.target_label is None:
        gold = [record for frame in frames for record in frame.attrs["multihead_gold"]]
        clean = [frame.copy() for frame in frames]
        for frame in clean:
            frame.attrs = {}
        result = pd.concat(clean, ignore_index=True)
        result.attrs.update(multihead_gold=gold, entity_labels=list(encoder.entity_labels))

        return result

    return pd.concat(frames, ignore_index=True)


def _parquet_documents(
    parquet_file: pq.ParquetFile, documents_per_batch: int
) -> Iterator[tuple[str, str, list[dict]]]:
    for batch in parquet_file.iter_batches(
        batch_size=documents_per_batch,
        columns=["doc_id", "text", "entities_json"],
    ):
        for document in batch.to_pylist():
            yield (
                document["doc_id"],
                document["text"],
                json.loads(document["entities_json"]),
            )
