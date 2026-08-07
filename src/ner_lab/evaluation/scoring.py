"""SemEval span scoring via nervaluate — this library's canonical evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from nervaluate import Evaluator

from ner_lab.evaluation.spans import SPAN_COLUMNS, entity_tags

SCENARIO_PREFIXES = {
    "strict": "span_strict",
    "exact": "span_exact",
    "partial": "span_partial",
    "ent_type": "span_type",
}


def score_spans(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    tags: list[str],
    min_overlap_percentage: float = 40.0,
) -> dict[str, Any]:
    """
    Score predicted spans against gold ones, SemEval 2013 Task 9.1 style.

    Returns nervaluate's overall result across the strict, exact, partial and
    ent_type scenarios. Both frames use the canonical span columns.
    """
    gold_documents = group_by_document(normalize_spans(gold))
    predicted_documents = group_by_document(normalize_spans(predicted))

    doc_ids = sorted(set(gold_documents) | set(predicted_documents))

    evaluator = Evaluator(
        true=[to_nervaluate(gold_documents.get(doc_id, [])) for doc_id in doc_ids],
        pred=[to_nervaluate(predicted_documents.get(doc_id, [])) for doc_id in doc_ids],
        tags=tags,
        loader="dict",
        min_overlap_percentage=min_overlap_percentage,
    )

    return evaluator.evaluate()["overall"]


def span_metrics(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    id2label: dict | None = None,
    tags: list[str] | None = None,
    min_overlap_percentage: float = 40.0,
) -> dict[str, float | int]:
    """Score spans and flatten the result into a metric dict."""
    if tags is None:
        if id2label is None:
            raise ValueError("Pass either tags or id2label.")

        tags = entity_tags(id2label)

    return flatten(score_spans(gold, predicted, tags, min_overlap_percentage))


def flatten(results: dict[str, Any]) -> dict[str, float | int]:
    """Flatten nervaluate's per-scenario objects into scalar metrics."""
    flattened: dict[str, float | int] = {}

    for scenario, prefix in SCENARIO_PREFIXES.items():
        result = results.get(scenario)

        if result is None:
            continue

        flattened[f"{prefix}_precision"] = round(float(result.precision), 4)
        flattened[f"{prefix}_recall"] = round(float(result.recall), 4)
        flattened[f"{prefix}_f1"] = round(float(result.f1), 4)
        flattened[f"{prefix}_correct"] = int(result.correct)
        flattened[f"{prefix}_incorrect"] = int(result.incorrect)
        flattened[f"{prefix}_partial"] = int(result.partial)
        flattened[f"{prefix}_missed"] = int(result.missed)
        flattened[f"{prefix}_spurious"] = int(result.spurious)

    return flattened


def to_nervaluate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert half-open offsets to nervaluate's inclusive ones.

    Everything else in this library uses `text[start:end]`; nervaluate's overlap
    arithmetic treats `end` as the last character, so it is off by one without
    this.
    """
    return [{"label": row["label"], "start": row["off0"], "end": row["off1"] - 1} for row in rows]


def normalize_spans(spans: pd.DataFrame) -> pd.DataFrame:
    """Validate a canonical span frame and rename it to the scoring schema."""
    missing = [column for column in SPAN_COLUMNS if column not in spans.columns]

    if missing:
        raise ValueError(f"Span frame is missing columns {missing}. Required: {SPAN_COLUMNS}")

    normalized = spans[SPAN_COLUMNS].copy()
    normalized["filename"] = normalized["filename"].astype(str)
    normalized["label"] = normalized["label"].astype(str)
    normalized["start_span"] = pd.to_numeric(normalized["start_span"]).astype(int)
    normalized["end_span"] = pd.to_numeric(normalized["end_span"]).astype(int)
    normalized["text"] = normalized["text"].astype(str)

    if not normalized.empty and (normalized["end_span"] <= normalized["start_span"]).any():
        raise ValueError("Invalid spans detected: end_span <= start_span.")

    return normalized.rename(
        columns={"start_span": "off0", "end_span": "off1", "text": "span"}
    )[["filename", "label", "off0", "off1", "span"]]


def group_by_document(spans: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Group scoring-schema span rows by document."""
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in spans.itertuples(index=False):
        grouped.setdefault(str(row.filename), []).append(
            {
                "filename": str(row.filename),
                "off0": int(row.off0),
                "off1": int(row.off1),
                "label": str(row.label),
            }
        )

    return grouped


def safe_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Precision, recall and F1 from counts, with zero denominators giving zero."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
