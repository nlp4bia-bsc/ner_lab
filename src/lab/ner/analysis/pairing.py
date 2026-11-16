"""Diagnostic pairing: what each strict error was, as units of gold spans and predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from lab.core.scoring import merge_intervals

ERROR_TYPES = ("CORRECT", "BOUNDARY_ERROR", "MISSED", "SPURIOUS", "UNCREDITED_EXACT")
BOUNDARY_KINDS = ("TOO_LONG", "TOO_SHORT", "SHIFTED_LEFT", "SHIFTED_RIGHT", "FRAGMENTED", "MERGED")
MATCHED_EDGES = ("START", "END", "NEITHER")
UNIT_COLUMNS = [
    "unit_id",
    "filename",
    "error_type",
    "boundary_kind",
    "matched_edge",
    "gold_ids",
    "pred_ids",
    "dice",
    "start_offset",
    "end_offset",
]


def pair_errors(events: pd.DataFrame, overlap_threshold: float = 0.60) -> pd.DataFrame:
    """
    Group a document's strict errors into diagnostic units, one row per unit.

    A strict match is a `CORRECT` unit. The gold spans and predictions left over are
    then grouped in this order. An identical pair is `UNCREDITED_EXACT`: the
    prediction is exact, but nervaluate credited the gold span to an earlier
    overlapping prediction. Two or more predictions inside one gold span whose union
    reaches `overlap_threshold` in character Dice are one `FRAGMENTED` boundary
    error; two or more gold spans inside one prediction, one `MERGED`. The rest are
    paired one to one, maximising total Dice, and a pair at or above the threshold
    is a boundary error, `TOO_LONG`, `TOO_SHORT`, `SHIFTED_LEFT` or `SHIFTED_RIGHT`,
    whose `matched_edge` says which edge it got right: `START`, `END` or `NEITHER`.
    What is still left is `MISSED` or `SPURIOUS`.

    Strict counts are never changed: every gold span and prediction keeps its strict
    outcome, and a unit only says what an error was. `gold_ids` and `pred_ids` list
    the spans in each unit; offsets are prediction minus gold, over the hull of a
    group.
    """
    units: list[dict[str, Any]] = []

    for row in events[events["strict_outcome"] == "TP"].itertuples(index=False):
        units.append(_unit(row.filename, "CORRECT", None, [row.gold_id], [row.pred_id], 1.0, 0, 0))

    errors = events[events["strict_outcome"] != "TP"]

    for filename, document in errors.groupby("filename", sort=True):
        gold = _spans(document, "gold")
        predicted = _spans(document, "pred")
        units.extend(_pair_document(str(filename), gold, predicted, overlap_threshold))

    frame = pd.DataFrame(units, columns=UNIT_COLUMNS[1:])
    frame.insert(0, "unit_id", range(len(frame)))

    return frame


def attach_units(events: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Add each event's `unit_id` and `error_type`, read off the unit holding its span."""
    by_gold = units[["unit_id", "error_type", "gold_ids"]].explode("gold_ids").dropna()
    by_predicted = units[["unit_id", "error_type", "pred_ids"]].explode("pred_ids").dropna()

    gold_unit = dict(
        zip(by_gold["gold_ids"].astype(int), zip(by_gold["unit_id"], by_gold["error_type"]))
    )
    predicted_unit = dict(
        zip(
            by_predicted["pred_ids"].astype(int),
            zip(by_predicted["unit_id"], by_predicted["error_type"]),
        )
    )

    assigned = [
        gold_unit[int(gold_id)] if pd.notna(gold_id) else predicted_unit[int(pred_id)]
        for gold_id, pred_id in zip(events["gold_id"], events["pred_id"])
    ]
    attached = events.copy()
    attached["unit_id"] = [unit_id for unit_id, _ in assigned]
    attached["error_type"] = [error_type for _, error_type in assigned]

    return attached


def span_dice(start: int, end: int, other_start: int, other_end: int) -> float:
    """Character Dice of two spans: twice their overlap over their summed lengths."""
    overlap = max(0, min(end, other_end) - max(start, other_start))

    return 2 * overlap / ((end - start) + (other_end - other_start))


def boundary_kind(start_offset: int, end_offset: int) -> str:
    """Name a one-to-one boundary error by where the prediction's edges fall."""
    if start_offset <= 0 <= end_offset:
        return "TOO_LONG"

    if start_offset >= 0 >= end_offset:
        return "TOO_SHORT"

    return "SHIFTED_LEFT" if start_offset < 0 else "SHIFTED_RIGHT"


def matched_edge(start_offset: int, end_offset: int) -> str:
    """
    Name the edge a one-to-one boundary error got right: `START`, `END` or `NEITHER`.

    Both cannot match, since the pair would then be exact. A shifted prediction always
    misses both.
    """
    if start_offset == 0:
        return "START"

    return "END" if end_offset == 0 else "NEITHER"


def _spans(document: pd.DataFrame, prefix: str) -> dict[int, tuple[int, int]]:
    present = document[document[f"{prefix}_id"].notna()]

    return {
        int(span_id): (int(start), int(end))
        for span_id, start, end in zip(
            present[f"{prefix}_id"], present[f"{prefix}_start"], present[f"{prefix}_end"]
        )
    }


def _pair_document(
    filename: str,
    gold: dict[int, tuple[int, int]],
    predicted: dict[int, tuple[int, int]],
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    units = _uncredited_exact(filename, gold, predicted)
    units += _contained_groups(filename, gold, predicted, overlap_threshold, "FRAGMENTED")
    units += _contained_groups(filename, predicted, gold, overlap_threshold, "MERGED")
    units += _one_to_one(filename, gold, predicted, overlap_threshold)

    units += [_unit(filename, "MISSED", None, [gold_id], [], None, None, None) for gold_id in gold]
    units += [
        _unit(filename, "SPURIOUS", None, [], [pred_id], None, None, None) for pred_id in predicted
    ]

    return units


def _uncredited_exact(
    filename: str, gold: dict[int, tuple[int, int]], predicted: dict[int, tuple[int, int]]
) -> list[dict[str, Any]]:
    units = []

    for gold_id, span in sorted(gold.items()):
        pred_id = next(
            (pred_id for pred_id, other in sorted(predicted.items()) if other == span), None
        )

        if pred_id is None:
            continue

        del gold[gold_id], predicted[pred_id]
        units.append(_unit(filename, "UNCREDITED_EXACT", None, [gold_id], [pred_id], 1.0, 0, 0))

    return units


def _contained_groups(
    filename: str,
    containers: dict[int, tuple[int, int]],
    members: dict[int, tuple[int, int]],
    overlap_threshold: float,
    kind: str,
) -> list[dict[str, Any]]:
    units = []
    by_length = sorted(containers.items(), key=lambda item: (item[1][1] - item[1][0], item[0]))

    for container_id, (start, end) in by_length:
        inside = sorted(
            member_id
            for member_id, (member_start, member_end) in members.items()
            if start <= member_start and member_end <= end
        )

        if len(inside) < 2:
            continue

        covered = sum(
            span_end - span_start
            for span_start, span_end in merge_intervals(
                [members[member_id] for member_id in inside]
            )
        )
        dice = 2 * covered / ((end - start) + covered)

        if dice < overlap_threshold:
            continue

        hull = (min(members[i][0] for i in inside), max(members[i][1] for i in inside))

        for member_id in inside:
            del members[member_id]

        del containers[container_id]

        if kind == "FRAGMENTED":
            units.append(
                _unit(
                    filename,
                    "BOUNDARY_ERROR",
                    kind,
                    [container_id],
                    inside,
                    dice,
                    hull[0] - start,
                    hull[1] - end,
                )
            )
        else:
            units.append(
                _unit(
                    filename,
                    "BOUNDARY_ERROR",
                    kind,
                    inside,
                    [container_id],
                    dice,
                    start - hull[0],
                    end - hull[1],
                )
            )

    return units


def _one_to_one(
    filename: str,
    gold: dict[int, tuple[int, int]],
    predicted: dict[int, tuple[int, int]],
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    if not gold or not predicted:
        return []

    gold_ids, predicted_ids = sorted(gold), sorted(predicted)
    dice = np.array(
        [
            [span_dice(*gold[gold_id], *predicted[pred_id]) for pred_id in predicted_ids]
            for gold_id in gold_ids
        ]
    )
    eligible = np.where(dice >= overlap_threshold, dice, 0.0)
    rows, columns = linear_sum_assignment(eligible, maximize=True)

    units = []

    for row, column in zip(rows, columns):
        if eligible[row, column] == 0.0:
            continue

        gold_id, pred_id = gold_ids[row], predicted_ids[column]
        start_offset = predicted[pred_id][0] - gold[gold_id][0]
        end_offset = predicted[pred_id][1] - gold[gold_id][1]
        units.append(
            _unit(
                filename,
                "BOUNDARY_ERROR",
                boundary_kind(start_offset, end_offset),
                [gold_id],
                [pred_id],
                float(dice[row, column]),
                start_offset,
                end_offset,
                matched=matched_edge(start_offset, end_offset),
            )
        )
        del gold[gold_id], predicted[pred_id]

    return units


def _unit(
    filename: str,
    error_type: str,
    kind: str | None,
    gold_ids: list,
    pred_ids: list,
    dice: float | None,
    start_offset: int | None,
    end_offset: int | None,
    matched: str | None = None,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "error_type": error_type,
        "boundary_kind": kind,
        "matched_edge": matched,
        "gold_ids": [int(gold_id) for gold_id in gold_ids],
        "pred_ids": [int(pred_id) for pred_id in pred_ids],
        "dice": dice,
        "start_offset": start_offset,
        "end_offset": end_offset,
    }
