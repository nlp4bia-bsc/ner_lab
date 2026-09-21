"""SemEval span scoring via nervaluate — this library's canonical evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from nervaluate import Evaluator

from lab.core.spans import SPAN_COLUMNS

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


def score_characters(gold: pd.DataFrame, predicted: pd.DataFrame) -> dict[str, float | int]:
    """
    Character-level precision, recall and F1, the character being the unit.

    A character counts as correct when a gold span and a predicted span of the
    same label both cover it. No span matching takes place, so overlapping spans
    on either side and the order they arrive in cannot change the result. Keys
    are `char_precision`, `char_recall`, `char_f1`, and the counts `char_correct`,
    `char_missed`, `char_spurious`.
    """
    gold_documents = group_by_document(normalize_spans(gold))
    predicted_documents = group_by_document(normalize_spans(predicted))

    correct = gold_total = predicted_total = 0

    for doc_id in set(gold_documents) | set(predicted_documents):
        gold_coverage = _character_coverage(gold_documents.get(doc_id, []))
        predicted_coverage = _character_coverage(predicted_documents.get(doc_id, []))

        gold_total += sum(_covered(ranges) for ranges in gold_coverage.values())
        predicted_total += sum(_covered(ranges) for ranges in predicted_coverage.values())
        correct += sum(
            _intersection(ranges, predicted_coverage.get(label, []))
            for label, ranges in gold_coverage.items()
        )

    scores = safe_f1(correct, predicted_total - correct, gold_total - correct)

    return {
        "char_precision": scores["precision"],
        "char_recall": scores["recall"],
        "char_f1": scores["f1"],
        "char_correct": scores["tp"],
        "char_missed": scores["fn"],
        "char_spurious": scores["fp"],
    }


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


def _character_coverage(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    by_label: dict[str, list[tuple[int, int]]] = {}

    for row in rows:
        by_label.setdefault(row["label"], []).append((row["off0"], row["off1"]))

    return {label: _merge(ranges) for label, ranges in by_label.items()}


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []

    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _covered(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def _intersection(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> int:
    total = i = j = 0

    while i < len(first) and j < len(second):
        total += max(0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))

        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1

    return total
