"""What the errors were: taxonomy, boundary details, threshold sensitivity, oracle budget, surfaces."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

import pandas as pd

from lab.core.scoring import safe_f1
from lab.ner.analysis.exposure import normalize_mention
from lab.ner.analysis.pairing import BOUNDARY_KINDS, ERROR_TYPES, MATCHED_EDGES, pair_errors

ONE_TO_ONE_KINDS = ("TOO_LONG", "TOO_SHORT", "SHIFTED_LEFT", "SHIFTED_RIGHT")
BOUNDARY_ERROR_COLUMNS = [
    "unit_id",
    "filename",
    "boundary_kind",
    "matched_edge",
    "n_gold",
    "n_predicted",
    "gold_text",
    "pred_text",
    "dice",
    "start_offset",
    "end_offset",
    "off_by_one",
    "whitespace_edge",
    "punctuation_only",
    "exposure",
]
BOUNDARY_SUMMARY_COLUMNS = [
    "boundary_kind",
    "matched_edge",
    "count",
    "share",
    "median_abs_start_offset",
    "median_abs_end_offset",
    "off_by_one_share",
    "punctuation_only_share",
]
ORACLE_SCENARIOS = (
    ("fix_boundary_errors", ("BOUNDARY_ERROR",)),
    ("recover_missed", ("MISSED",)),
    ("remove_spurious", ("SPURIOUS",)),
    ("fix_uncredited_exact", ("UNCREDITED_EXACT",)),
    ("fix_all", ("BOUNDARY_ERROR", "MISSED", "SPURIOUS", "UNCREDITED_EXACT")),
)


def error_taxonomy(units: pd.DataFrame) -> pd.DataFrame:
    """
    How many diagnostic units of each type there are, correct ones first, errors by count.

    `share` is out of every unit; `error_share` and `cumulative_error_share` are out
    of the errors alone, so the rows read as a Pareto of what went wrong.
    """
    counts = units["error_type"].value_counts().reindex(ERROR_TYPES, fill_value=0)
    errors = counts.drop("CORRECT").sort_values(ascending=False, kind="stable")
    total_errors = errors.sum()

    taxonomy = pd.DataFrame(
        {
            "error_type": ["CORRECT", *errors.index],
            "count": [counts["CORRECT"], *errors.to_numpy()],
        }
    )
    taxonomy["share"] = taxonomy["count"] / max(len(units), 1)
    shares = errors / total_errors if total_errors else errors.astype(float)
    taxonomy["error_share"] = [float("nan"), *shares]
    taxonomy["cumulative_error_share"] = [float("nan"), *shares.cumsum()]

    return taxonomy


def boundary_errors(
    units: pd.DataFrame, gold: pd.DataFrame, predicted: pd.DataFrame, gold_exposure: pd.Series
) -> pd.DataFrame:
    """
    One row per boundary-error unit: its kind, both surfaces, Dice, offsets and flags.

    `matched_edge` is the edge a one-to-one error got right, `START`, `END` or
    `NEITHER`, and empty for a fragmentation or merge. A merge lists its gold surfaces and a fragmentation its predicted ones, joined by
    ` | `. `off_by_one` is a one-to-one error one character off at one edge;
    `whitespace_edge` a prediction starting or ending on whitespace;
    `punctuation_only` a one-to-one error whose surfaces differ only by punctuation
    or whitespace at their edges. `exposure` is the gold span's, the first one's in
    a merge.
    """
    boundary = units[units["error_type"] == "BOUNDARY_ERROR"]
    rows = []

    for unit in boundary.itertuples(index=False):
        gold_texts = [str(gold.at[gold_id, "text"]) for gold_id in unit.gold_ids]
        predicted_texts = [str(predicted.at[pred_id, "text"]) for pred_id in unit.pred_ids]
        one_to_one = unit.boundary_kind in ONE_TO_ONE_KINDS

        rows.append(
            {
                "unit_id": unit.unit_id,
                "filename": unit.filename,
                "boundary_kind": unit.boundary_kind,
                "matched_edge": unit.matched_edge,
                "n_gold": len(gold_texts),
                "n_predicted": len(predicted_texts),
                "gold_text": " | ".join(gold_texts),
                "pred_text": " | ".join(predicted_texts),
                "dice": unit.dice,
                "start_offset": unit.start_offset,
                "end_offset": unit.end_offset,
                "off_by_one": one_to_one and abs(unit.start_offset) + abs(unit.end_offset) == 1,
                "whitespace_edge": any(text != text.strip() for text in predicted_texts),
                "punctuation_only": one_to_one
                and _strip_edges(gold_texts[0]) == _strip_edges(predicted_texts[0]),
                "exposure": gold_exposure.at[unit.gold_ids[0]],
            }
        )

    return pd.DataFrame(rows, columns=BOUNDARY_ERROR_COLUMNS)


def boundary_summary(boundary: pd.DataFrame) -> pd.DataFrame:
    """
    Boundary errors per kind and matched edge: how many, and for one-to-one kinds how far off.

    A one-to-one kind has a row per edge it got right, `START`, `END` or `NEITHER`, so
    a prediction running past the gold's end is told apart from one starting before
    its start; a fragmentation or merge has one row, its `matched_edge` empty. `share`
    is out of every boundary error. Offsets are absolute, in characters; a
    fragmentation's or merge's hull offsets are not a single edge's error, so they
    are left out of the medians.
    """
    rows = []

    for kind in BOUNDARY_KINDS:
        one_to_one = kind in ONE_TO_ONE_KINDS

        for edge in MATCHED_EDGES if one_to_one else (None,):
            members = boundary[boundary["boundary_kind"] == kind]

            if one_to_one:
                members = members[members["matched_edge"] == edge]

            if members.empty:
                continue

            rows.append(
                {
                    "boundary_kind": kind,
                    "matched_edge": edge,
                    "count": len(members),
                    "share": len(members) / len(boundary),
                    **_offset_summary(members, one_to_one),
                }
            )

    return pd.DataFrame(rows, columns=BOUNDARY_SUMMARY_COLUMNS)


def _offset_summary(members: pd.DataFrame, one_to_one: bool) -> dict[str, float | None]:
    if not one_to_one:
        return dict.fromkeys(BOUNDARY_SUMMARY_COLUMNS[4:])

    return {
        "median_abs_start_offset": members["start_offset"].abs().median(),
        "median_abs_end_offset": members["end_offset"].abs().median(),
        "off_by_one_share": members["off_by_one"].mean(),
        "punctuation_only_share": members["punctuation_only"].mean(),
    }


def overlap_sensitivity(
    events: pd.DataFrame, thresholds: Sequence[float], chosen: float
) -> pd.DataFrame:
    """
    Diagnostic units of each type when pairing is rerun at each Dice threshold.

    Strict counts do not move; only how the strict errors are grouped does, and this
    table shows how much the taxonomy owes to the threshold chosen.
    """
    rows = []

    for threshold in sorted({*thresholds, chosen}):
        units = pair_errors(events, overlap_threshold=threshold)
        counts = units["error_type"].value_counts()
        kinds = units["boundary_kind"].value_counts()

        rows.append(
            {
                "threshold": threshold,
                "chosen": threshold == chosen,
                **{
                    error_type.lower(): int(counts.get(error_type, 0)) for error_type in ERROR_TYPES
                },
                "fragmented": int(kinds.get("FRAGMENTED", 0)),
                "merged": int(kinds.get("MERGED", 0)),
            }
        )

    return pd.DataFrame(rows)


def oracle_error_budget(units: pd.DataFrame) -> pd.DataFrame:
    """
    Strict F1 if each kind of error were fixed, one scenario at a time.

    Fixing a unit turns its gold spans into true positives and removes its
    predictions as false positives: a fragmentation of k predictions removes k false
    positives for one true positive, a merge of k gold spans gives k true positives
    for one false positive. These are ceilings, not forecasts, and the scenarios do
    not add up.
    """
    errors = units[units["error_type"] != "CORRECT"]
    n_gold = errors["gold_ids"].map(len)
    n_predicted = errors["pred_ids"].map(len)
    tp = int((units["error_type"] == "CORRECT").sum())
    fn, fp = int(n_gold.sum()), int(n_predicted.sum())
    observed = safe_f1(tp, fp, fn)["f1"]

    rows = []

    for scenario, error_types in ORACLE_SCENARIOS:
        fixed = errors["error_type"].isin(error_types)
        recovered, removed = int(n_gold[fixed].sum()), int(n_predicted[fixed].sum())
        oracle = safe_f1(tp + recovered, fp - removed, fn - recovered)

        rows.append(
            {
                "scenario": scenario,
                "units": int(fixed.sum()),
                "tp": oracle["tp"],
                "fp": oracle["fp"],
                "fn": oracle["fn"],
                "observed_f1": observed,
                "oracle_f1": oracle["f1"],
                "gain": round(oracle["f1"] - observed, 4),
            }
        )

    return pd.DataFrame(rows)


def error_surfaces(
    units: pd.DataFrame, gold: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    """
    How often each surface is in an error, by error type, most frequent first.

    A unit's surface is its first gold span's normalized text, or its prediction's
    when it has no gold span, so a mention and its case variants count together.
    """
    errors = units[units["error_type"] != "CORRECT"]
    error_types = [error_type for error_type in ERROR_TYPES if error_type != "CORRECT"]

    if errors.empty:
        return pd.DataFrame(
            columns=["surface", "total", *(error_type.lower() for error_type in error_types)]
        )

    surfaces = [
        normalize_mention(
            gold.at[unit.gold_ids[0], "text"]
            if unit.gold_ids
            else predicted.at[unit.pred_ids[0], "text"]
        )
        for unit in errors.itertuples(index=False)
    ]

    table = (
        pd.DataFrame({"surface": surfaces, "error_type": errors["error_type"].to_numpy()})
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=error_types, fill_value=0)
    )
    table.columns = [error_type.lower() for error_type in error_types]
    table.insert(0, "total", table.sum(axis=1))

    return (
        table.reset_index()
        .sort_values(["total", "surface"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )


def _strip_edges(text: str) -> str:
    start, end = 0, len(text)

    while start < end and _is_edge(text[start]):
        start += 1

    while end > start and _is_edge(text[end - 1]):
        end -= 1

    return text[start:end]


def _is_edge(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")
