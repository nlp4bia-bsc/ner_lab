"""Span reconstruction and binary diagnostics without competition across entity types."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from lab.core.spans import span_dataframe
from lab.ner.evaluation.multihead import IDENTITY, multihead_spans, strict_multihead_metrics
from lab.ner.models.binary_decoding import (
    BinaryDecoding,
    adapted_probabilities,
    binary_probabilities,
    decode_binary,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def categorical_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Project binary gold to a BIO track per entity while sharing token/offset cells."""
    result = rows.copy()
    result.attrs.pop("label_schema", None)
    result["labels"] = [
        np.where(
            np.asarray(row)[..., 0] == -100,
            -100,
            np.asarray(row)[..., 0] + 2 * np.asarray(row)[..., 1],
        ).tolist()
        for row in rows.labels
    ]

    return result


def decoding_settings(rows: pd.DataFrame, settings: BinaryDecoding | None = None) -> BinaryDecoding:
    """Resolve explicitly supplied settings before the encoder-attached training defaults."""
    return settings or BinaryDecoding(**rows.attrs.get("multilabel_decoding", {}))


def binary_predictions(
    rows: pd.DataFrame,
    logits: np.ndarray,
    entity_labels: list[str],
    settings: BinaryDecoding | None = None,
    include_probabilities: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return independent BIO tags and optional conditional legal-state probabilities."""
    settings = decoding_settings(rows, settings)
    values = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    if (
        values.ndim != 4
        or values.shape[0] != len(rows)
        or values.shape[2:] != (len(entity_labels), 2)
    ):
        raise ValueError("Multilabel predictions must have shape [rows,L,K,2].")
    mask = torch.zeros(values.shape[:2], dtype=torch.bool)
    for index, row in enumerate(rows.itertuples(index=False)):
        mask[index, : len(row.word_mask)] = torch.as_tensor(row.word_mask)
    with torch.no_grad():
        tags = decode_binary(values, mask, settings, entity_labels)
        probabilities = (
            adapted_probabilities(values, settings, entity_labels).numpy()
            if include_probabilities
            else None
        )

    return tags.numpy(), probabilities


def multilabel_spans(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    texts: dict[str, str] | None = None,
    include_scores: bool = False,
    settings: BinaryDecoding | None = None,
) -> pd.DataFrame:
    """Reuse independent BIO reconstruction and offsets, scoring chosen conditional legal states."""
    decisions, probabilities = binary_predictions(
        rows, predictions, entity_labels, settings, include_scores
    )

    return multihead_spans(
        categorical_rows(rows),
        tokenizer,
        entity_labels,
        texts=texts,
        tag_ids=decisions,
        include_scores=include_scores,
        probabilities=probabilities,
    )


def multilabel_token_metrics(
    rows: pd.DataFrame, predictions: np.ndarray, settings: BinaryDecoding | None = None
) -> dict[str, float | int]:
    """Binary B/I diagnostics use channel threshold .5; span selection uses its own decoder."""
    labels = rows.attrs["entity_labels"]
    values = torch.as_tensor(predictions, dtype=torch.float32)
    chosen = binary_probabilities(values, decoding_settings(rows, settings), labels).numpy() >= 0.5
    gold = np.full(chosen.shape, -100)
    for index, row in enumerate(rows.labels):
        gold[index, : len(row)] = row
    valid = gold != -100
    result = {}
    for head, label in enumerate(labels):
        true = gold[:, :, head] == 1
        pred = chosen[:, :, head] & valid[:, :, head]
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        result[f"binary_token_precision__{label}"] = tp / (tp + fp) if tp + fp else 0.0
        result[f"binary_token_recall__{label}"] = tp / (tp + fn) if tp + fn else 0.0
        result.update(
            {
                f"binary_token_tp__{label}": tp,
                f"binary_token_fp__{label}": fp,
                f"binary_token_fn__{label}": fn,
                f"binary_token_f1__{label}": 2 * tp / (2 * tp + fp + fn)
                if 2 * tp + fp + fn
                else 0.0,
            }
        )
    result["binary_token_accuracy"] = (
        float((chosen[valid] == gold[valid]).mean()) if valid.any() else 0.0
    )

    return result


def evaluate_multilabel(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    include_tokens: bool = True,
    settings: BinaryDecoding | None = None,
) -> dict:
    """Score original gold, preserve unknown types, and expose representability losses."""
    projected = categorical_rows(rows)
    reconstructed = multihead_spans(projected, tokenizer, entity_labels)
    original = rows.attrs.get("multihead_gold")
    gold = span_dataframe(original) if original is not None else reconstructed
    gold = gold[gold.filename.isin(rows.doc_id.astype(str))]
    predicted = multilabel_spans(rows, predictions, tokenizer, entity_labels, settings=settings)
    result = strict_multihead_metrics(gold, predicted, entity_labels)
    result.update(gold_representation_metrics(rows, tokenizer, entity_labels, reconstructed))
    if include_tokens:
        result.update(multilabel_token_metrics(rows, predictions, settings))

    return result


def gold_representation_metrics(
    rows: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    reconstructed: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Expose original mentions that cannot survive word offsets, overlap policy or window cores."""
    if reconstructed is None:
        reconstructed = multihead_spans(categorical_rows(rows), tokenizer, entity_labels)
    original = rows.attrs.get("multihead_gold")
    gold = span_dataframe(original) if original is not None else reconstructed
    gold = gold[gold.filename.isin(rows.doc_id.astype(str))]
    identities = set(reconstructed[IDENTITY].itertuples(index=False, name=None))
    lost = sum(
        item[1] in entity_labels and item not in identities
        for item in gold[IDENTITY].itertuples(index=False, name=None)
    )
    if lost:
        warnings.warn(
            f"{lost} original gold entities differ from encoded BIO offsets/window cores; "
            "original gold remains in span scoring.",
            stacklevel=2,
        )

    return {
        "gold_unrepresentable_entities": lost,
        "gold_original_entities": len(gold),
        "gold_reconstructed_entities": len(reconstructed),
    }
