"""Strict events: the canonical scorer's strict accounting, one row per span it judged."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd

from lab.core.scoring import evaluate_spans, flatten, score_characters, score_dice

STRICT_OUTCOMES = ("TP", "FN", "FP")
EVENT_COLUMNS = [
    "filename",
    "strict_outcome",
    "gold_id",
    "pred_id",
    "gold_start",
    "gold_end",
    "gold_text",
    "pred_start",
    "pred_end",
    "pred_text",
    "score",
]


def strict_events(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    target_label: str,
    min_overlap_percentage: float = 40.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the canonical scorer and lay its strict accounting out as rows.

    A strict match is one `TP` row holding both spans; every other gold span is an
    `FN` row and every other prediction an `FP` row. `gold_id` and `pred_id` are row
    positions in `gold` and `predicted`, so every span appears exactly once.

    The metrics are the span and character metrics `predict_entities` reports, taken
    from the same nervaluate run the rows are read from, and the rows are checked
    against its strict counts before they are returned.
    """
    result = evaluate_spans(
        gold=gold,
        predicted=predicted,
        tags=[target_label],
        min_overlap_percentage=min_overlap_percentage,
    )
    metrics = {
        **flatten(result["overall"]),
        **score_characters(gold, predicted),
        **score_dice(gold, predicted),
    }

    matches = strict_matches(gold, predicted, result)
    events = _event_rows(gold, predicted, matches)
    _check_accounting(events, metrics)

    return events, metrics


def strict_matches(
    gold: pd.DataFrame, predicted: pd.DataFrame, result: dict[str, Any]
) -> list[tuple[int, int]]:
    """
    The `(gold_id, pred_id)` pairs nervaluate counted as strict matches.

    nervaluate names a match by its prediction only, so the gold side is the first
    unmatched gold span identical to it; identical gold spans are interchangeable.
    """
    rows_by_document: dict[str, list[int]] = defaultdict(list)

    for position, filename in enumerate(predicted["filename"].astype(str)):
        rows_by_document[filename].append(position)

    unmatched_gold: dict[tuple, deque[int]] = defaultdict(deque)

    for position, key in enumerate(_span_keys(gold)):
        unmatched_gold[key].append(position)

    predicted_keys = _span_keys(predicted)
    matches = []

    for document_index, row_index in sorted(result["overall_indices"]["strict"].correct_indices):
        pred_id = rows_by_document[result["doc_ids"][document_index]][row_index]
        candidates = unmatched_gold[predicted_keys[pred_id]]

        if not candidates:
            raise RuntimeError(
                f"nervaluate counted prediction {predicted_keys[pred_id]} as a strict match, "
                "but no unmatched gold span is identical to it."
            )

        matches.append((candidates.popleft(), pred_id))

    return matches


def _span_keys(spans: pd.DataFrame) -> list[tuple[str, str, int, int]]:
    return list(
        zip(
            spans["filename"].astype(str),
            spans["label"].astype(str),
            spans["start_span"].astype(int),
            spans["end_span"].astype(int),
        )
    )


def _event_rows(
    gold: pd.DataFrame, predicted: pd.DataFrame, matches: list[tuple[int, int]]
) -> pd.DataFrame:
    matched = pd.DataFrame(matches, columns=["gold_id", "pred_id"], dtype="Int64")
    matched_gold = set(matched["gold_id"].tolist())
    matched_predicted = set(matched["pred_id"].tolist())

    missed = [position for position in range(len(gold)) if position not in matched_gold]
    spurious = [position for position in range(len(predicted)) if position not in matched_predicted]

    events = pd.concat(
        [
            matched.assign(strict_outcome="TP"),
            pd.DataFrame({"gold_id": pd.array(missed, dtype="Int64"), "strict_outcome": "FN"}),
            pd.DataFrame({"pred_id": pd.array(spurious, dtype="Int64"), "strict_outcome": "FP"}),
        ],
        ignore_index=True,
    )
    events["gold_id"] = events["gold_id"].astype("Int64")
    events["pred_id"] = events["pred_id"].astype("Int64")

    gold_side = _side(gold, "gold")
    predicted_side = _side(predicted, "pred")
    events = events.merge(gold_side, on="gold_id", how="left").merge(
        predicted_side, on="pred_id", how="left", suffixes=("", "_pred")
    )
    events["filename"] = events["filename"].fillna(events["filename_pred"])
    events["position"] = events["gold_start"].fillna(events["pred_start"])

    return (
        events.sort_values(["filename", "position", "strict_outcome"], kind="stable")
        .reset_index(drop=True)
        .reindex(columns=EVENT_COLUMNS)
    )


def _side(spans: pd.DataFrame, prefix: str) -> pd.DataFrame:
    side = pd.DataFrame(
        {
            f"{prefix}_id": pd.array(range(len(spans)), dtype="Int64"),
            "filename": spans["filename"].astype(str).to_numpy(),
            f"{prefix}_start": pd.array(spans["start_span"].astype(int), dtype="Int64"),
            f"{prefix}_end": pd.array(spans["end_span"].astype(int), dtype="Int64"),
            f"{prefix}_text": spans["text"].astype(str).to_numpy(),
        }
    )

    if prefix == "pred":
        side["score"] = (
            pd.to_numeric(spans["score"], errors="coerce").to_numpy()
            if "score" in spans.columns
            else float("nan")
        )

    return side


def _check_accounting(events: pd.DataFrame, metrics: dict[str, Any]) -> None:
    counted = events["strict_outcome"].value_counts()
    unmatched = metrics["span_strict_incorrect"] + metrics["span_strict_partial"]
    expected = {
        "TP": metrics["span_strict_correct"],
        "FN": metrics["span_strict_missed"] + unmatched,
        "FP": metrics["span_strict_spurious"] + unmatched,
    }
    found = {outcome: int(counted.get(outcome, 0)) for outcome in STRICT_OUTCOMES}

    if found != expected:
        raise RuntimeError(
            f"Strict events {found} disagree with nervaluate's strict counts {expected}."
        )
