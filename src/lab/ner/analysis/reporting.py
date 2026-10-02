"""Aggregate NER error/generalization tables built on the canonical evaluator."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from lab.core import safe_f1
from lab.ner.evaluation import span_metrics
from lab.ner.analysis.diagnostics import span_feature_frame
from lab.ner.analysis.exposure import ExposureIndex, describe_frame

GENERALIZATION_ORDER = ("SEEN", "FEW_SHOT", "ZERO_SHOT")


def score_subgroup(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    gold_mask,
    pred_mask,
    *,
    tags: list[str],
    min_overlap_percentage: float,
) -> dict[str, float | int]:
    """Evaluate complete gold/prediction subgroup populations with the official scorer."""
    gold_mask = pd.Series(gold_mask, index=gold.index).fillna(False).astype(bool)
    pred_mask = pd.Series(pred_mask, index=predicted.index).fillna(False).astype(bool)
    selected_gold = gold.loc[gold_mask].reset_index(drop=True)
    selected_pred = predicted.loc[pred_mask].reset_index(drop=True)
    metrics = span_metrics(
        selected_gold,
        selected_pred,
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )
    incorrect = int(metrics.get("span_strict_incorrect", 0)) + int(
        metrics.get("span_strict_partial", 0)
    )
    return {
        "support": int(len(selected_gold)),
        "predicted_support": int(len(selected_pred)),
        "strict_tp": int(metrics.get("span_strict_correct", 0)),
        "strict_fp": int(metrics.get("span_strict_spurious", 0)) + incorrect,
        "strict_fn": int(metrics.get("span_strict_missed", 0)) + incorrect,
        "strict_precision": float(metrics.get("span_strict_precision", 0.0)),
        "strict_recall": float(metrics.get("span_strict_recall", 0.0)),
        "strict_f1": float(metrics.get("span_strict_f1", 0.0)),
        "char_tp": int(metrics.get("char_correct", 0)),
        "char_fp": int(metrics.get("char_spurious", 0)),
        "char_fn": int(metrics.get("char_missed", 0)),
        "char_precision": float(metrics.get("char_precision", 0.0)),
        "char_recall": float(metrics.get("char_recall", 0.0)),
        "char_f1": float(metrics.get("char_f1", 0.0)),
    }


def build_analysis_tables(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    training: pd.DataFrame,
    events: pd.DataFrame,
    official_metrics: dict[str, float | int],
    exposure: ExposureIndex,
    *,
    tags: list[str],
    min_overlap_percentage: float = 40.0,
    bootstrap_samples: int = 2000,
    bootstrap_confidence: float = 0.95,
    random_state: int = 13,
) -> dict[str, pd.DataFrame]:
    """Create publication-ready Parquet tables without defining a second scorer."""
    gold_exposure = describe_frame(gold, exposure)
    pred_exposure = describe_frame(predicted, exposure)
    gold_features = span_feature_frame(gold)
    pred_features = span_feature_frame(predicted)

    overall = pd.DataFrame(
        [
            {
                "scope": "ALL",
                "support": int(len(gold)),
                "predicted_support": int(len(predicted)),
                **{key: value for key, value in official_metrics.items() if np.isscalar(value)},
            }
        ]
    )

    by_generalization = _categorical_metrics(
        gold,
        predicted,
        gold_exposure["generalization_class"],
        pred_exposure["generalization_class"],
        GENERALIZATION_ORDER,
        family="generalization",
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )

    by_similarity = _categorical_metrics(
        gold,
        predicted,
        gold_exposure["similarity_bin"],
        pred_exposure["similarity_bin"],
        _stable_values(gold_exposure["similarity_bin"], pred_exposure["similarity_bin"]),
        family="similarity_bin",
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )

    by_frequency = _categorical_metrics(
        gold,
        predicted,
        gold_exposure["train_frequency_bin"],
        pred_exposure["train_frequency_bin"],
        _stable_values(
            gold_exposure["train_frequency_bin"],
            pred_exposure["train_frequency_bin"],
        ),
        family="training_frequency",
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )

    by_label_rows = []
    by_label_generalization_rows = []
    for label in tags:
        gold_label = gold["label"].astype(str) == label
        pred_label = predicted["label"].astype(str) == label
        row = score_subgroup(
            gold,
            predicted,
            gold_label,
            pred_label,
            tags=[label],
            min_overlap_percentage=min_overlap_percentage,
        )
        by_label_rows.append({"label": label, **row})
        for class_name in GENERALIZATION_ORDER:
            row = score_subgroup(
                gold,
                predicted,
                gold_label & (gold_exposure["generalization_class"] == class_name),
                pred_label & (pred_exposure["generalization_class"] == class_name),
                tags=[label],
                min_overlap_percentage=min_overlap_percentage,
            )
            by_label_generalization_rows.append(
                {"label": label, "generalization_class": class_name, **row}
            )

    by_length = _categorical_metrics(
        gold,
        predicted,
        gold_features["entity_word_length_bin"],
        pred_features["entity_word_length_bin"],
        _ordered_length_values(
            gold_features["entity_word_length_bin"],
            pred_features["entity_word_length_bin"],
        ),
        family="entity_word_length",
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )

    by_acronym = _categorical_metrics(
        gold,
        predicted,
        gold_features["entity_acronym_like"].astype(str),
        pred_features["entity_acronym_like"].astype(str),
        ("False", "True"),
        family="acronym",
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )

    by_overlap = _categorical_metrics(
        gold,
        predicted,
        gold_features["gold_overlaps_other_gold"].astype(str),
        pred_features["gold_overlaps_other_gold"].astype(str),
        ("False", "True"),
        family="overlapping",
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )

    by_nested = _categorical_metrics(
        gold,
        predicted,
        gold_features["gold_nested"].astype(str),
        pred_features["gold_nested"].astype(str),
        ("False", "True"),
        family="nested",
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )

    gold_recall = gold_conditioned_generalization(events, gold_exposure)
    generalization_profile = generalization_error_profile(events)
    error_taxonomy = error_taxonomy_table(events)
    error_type_pareto_table = error_type_pareto(error_taxonomy)
    boundary = boundary_error_table(events)
    boundary_summary = boundary_summary_table(boundary)
    confusion_counts, confusion_normalized, confusion_extended = confusion_tables(
        events,
        tags,
    )
    documents = document_metrics(
        gold,
        predicted,
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )
    dataset_shift = dataset_shift_labels(training, gold)
    dataset_shift_summary_table = dataset_shift_summary(dataset_shift)
    oracle = oracle_error_budget(events, official_metrics)
    confidence = confidence_by_error(events)
    confidence_summary_table = confidence_summary(predicted)
    confidence_sweep = confidence_threshold_sweep(
        gold,
        predicted,
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
    )
    pareto = error_pareto(events)
    intersections = error_intersections(events)
    bootstrap = bootstrap_intervals(
        gold,
        predicted,
        gold_exposure,
        pred_exposure,
        tags=tags,
        min_overlap_percentage=min_overlap_percentage,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=random_state,
    )

    return {
        "overall_metrics": overall,
        "metrics_by_generalization": by_generalization,
        "gold_recall_by_generalization": gold_recall,
        "generalization_error_profile": generalization_profile,
        "metrics_by_similarity": by_similarity,
        "metrics_by_training_frequency": by_frequency,
        "metrics_by_label": pd.DataFrame(by_label_rows),
        "metrics_by_label_generalization": pd.DataFrame(by_label_generalization_rows),
        "metrics_by_entity_length": by_length,
        "metrics_by_acronym": by_acronym,
        "metrics_by_overlap": by_overlap,
        "metrics_by_nested": by_nested,
        "error_taxonomy": error_taxonomy,
        "error_type_pareto": error_type_pareto_table,
        "boundary_errors": boundary,
        "boundary_summary": boundary_summary,
        "label_confusion_counts": confusion_counts,
        "label_confusion_normalized": confusion_normalized,
        "label_confusion_extended": confusion_extended,
        "document_metrics": documents,
        "dataset_shift_labels": dataset_shift,
        "dataset_shift_summary": dataset_shift_summary_table,
        "oracle_error_budget": oracle,
        "confidence_by_error": confidence,
        "confidence_summary": confidence_summary_table,
        "confidence_threshold_sweep": confidence_sweep,
        "error_pareto": pareto,
        "error_intersections": intersections,
        "bootstrap_intervals": bootstrap,
    }


def gold_conditioned_generalization(
    events: pd.DataFrame,
    gold_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Gold-side strict recall/error decomposition by lexical exposure class."""
    rows = []
    gold_events = events.loc[events["gold_exists"]].copy()
    for class_name in GENERALIZATION_ORDER:
        ids = set(
            gold_exposure.index[
                gold_exposure["generalization_class"] == class_name
            ].astype(int)
        )
        subset = gold_events.loc[gold_events["gold_id"].isin(ids)]
        support = int(len(subset))
        correct = int(subset["strict_correct"].sum())
        rows.append(
            {
                "generalization_class": class_name,
                "support": support,
                "strict_correct": correct,
                "strict_errors": support - correct,
                "strict_recall": correct / support if support else 0.0,
                "boundary_errors": int(
                    (
                        (subset["error_pair_role"] == "GOLD")
                        & subset["error_primary"].isin(
                            ["BOUNDARY_ERROR", "BOUNDARY_AND_LABEL_ERROR"]
                        )
                    ).sum()
                ),
                "label_errors": int(
                    (
                        (subset["error_pair_role"] == "GOLD")
                        & subset["error_primary"].isin(
                            ["LABEL_ERROR", "BOUNDARY_AND_LABEL_ERROR"]
                        )
                    ).sum()
                ),
                "missed": int((subset["error_primary"] == "MISSED").sum()),
            }
        )
    return pd.DataFrame(rows)


def generalization_error_profile(events: pd.DataFrame) -> pd.DataFrame:
    """Mutually exclusive gold-conditioned outcomes by lexical exposure class.

    This table complements PRF by answering a more diagnostic question: when a
    gold mention is not strictly correct, is it missed, delimited incorrectly,
    assigned the wrong label, or affected by both boundary and label errors?
    Spurious predictions are intentionally excluded because they have no gold
    mention from which to define a gold-conditioned exposure class.
    """
    gold_events = events.loc[events["gold_exists"]].copy()
    rows = []
    error_order = (
        "CORRECT",
        "MISSED",
        "BOUNDARY_ERROR",
        "LABEL_ERROR",
        "BOUNDARY_AND_LABEL_ERROR",
    )
    for class_name in GENERALIZATION_ORDER:
        subset = gold_events.loc[
            gold_events["generalization_class"].astype(str) == class_name
        ].copy()
        support = int(len(subset))
        counts = subset["error_primary"].fillna("UNKNOWN").astype(str).value_counts()
        for error_type in error_order:
            count = int(counts.get(error_type, 0))
            rows.append(
                {
                    "generalization_class": class_name,
                    "error_type": error_type,
                    "count": count,
                    "support": support,
                    "proportion": count / support if support else 0.0,
                }
            )
    return pd.DataFrame(rows)


def error_type_pareto(error_taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Errors-only Pareto table ordered by diagnostic failure frequency."""
    if error_taxonomy.empty:
        return pd.DataFrame(
            columns=[
                "error_type",
                "count",
                "proportion",
                "overall_proportion",
                "cumulative_count",
                "cumulative_proportion",
            ]
        )
    work = error_taxonomy.loc[
        error_taxonomy["error_type"].astype(str) != "CORRECT"
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "error_type",
                "count",
                "proportion",
                "overall_proportion",
                "cumulative_count",
                "cumulative_proportion",
            ]
        )
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0).astype(int)
    total_errors = max(int(work["count"].sum()), 1)
    work = work.sort_values(["count", "error_type"], ascending=[False, True]).reset_index(drop=True)
    work["overall_proportion"] = pd.to_numeric(work["proportion"], errors="coerce").fillna(0.0)
    work["proportion"] = work["count"] / total_errors
    work["cumulative_count"] = work["count"].cumsum()
    work["cumulative_proportion"] = work["cumulative_count"] / total_errors
    return work[
        [
            "error_type",
            "count",
            "proportion",
            "overall_proportion",
            "cumulative_count",
            "cumulative_proportion",
        ]
    ]


def boundary_summary_table(boundary: pd.DataFrame) -> pd.DataFrame:
    """Compact descriptive statistics for paired boundary-related errors."""
    if boundary.empty:
        return pd.DataFrame()

    start = pd.to_numeric(boundary.get("span_start_delta"), errors="coerce").dropna()
    end = pd.to_numeric(boundary.get("span_end_delta"), errors="coerce").dropna()
    abs_start = pd.to_numeric(boundary.get("span_abs_start_delta"), errors="coerce").dropna()
    abs_end = pd.to_numeric(boundary.get("span_abs_end_delta"), errors="coerce").dropna()
    n = int(len(boundary))

    def median(values):
        return float(values.median()) if len(values) else np.nan

    def percentile(values, q):
        return float(values.quantile(q)) if len(values) else np.nan

    return pd.DataFrame(
        [
            {
                "boundary_pairs": n,
                "start_exact_proportion": float((start == 0).mean()) if len(start) else np.nan,
                "end_exact_proportion": float((end == 0).mean()) if len(end) else np.nan,
                "start_off_by_one_proportion": float((abs_start <= 1).mean()) if len(abs_start) else np.nan,
                "end_off_by_one_proportion": float((abs_end <= 1).mean()) if len(abs_end) else np.nan,
                "median_abs_start_delta": median(abs_start),
                "median_abs_end_delta": median(abs_end),
                "p90_abs_start_delta": percentile(abs_start, 0.90),
                "p90_abs_end_delta": percentile(abs_end, 0.90),
                "max_abs_start_delta": float(abs_start.max()) if len(abs_start) else np.nan,
                "max_abs_end_delta": float(abs_end.max()) if len(abs_end) else np.nan,
            }
        ]
    )


def error_taxonomy_table(events: pd.DataFrame) -> pd.DataFrame:
    canonical = _diagnostic_units(events)
    counts = canonical["error_primary"].fillna("UNKNOWN").value_counts()
    total = int(counts.sum())
    return pd.DataFrame(
        [
            {
                "error_type": name,
                "count": int(count),
                "proportion": float(count / total) if total else 0.0,
            }
            for name, count in counts.items()
        ]
    )


def boundary_error_table(events: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (events["error_pair_role"] == "GOLD")
        & events["error_primary"].isin(
            ["BOUNDARY_ERROR", "BOUNDARY_AND_LABEL_ERROR"]
        )
    )
    columns = [
        "document_id",
        "gold_text",
        "gold_label",
        "pred_text",
        "pred_label",
        "pred_confidence",
        "span_start_delta",
        "span_end_delta",
        "span_length_delta",
        "span_abs_start_delta",
        "span_abs_end_delta",
        "span_intersection",
        "span_iou",
        "span_boundary_error",
        "span_relation",
        "span_one_character_offset",
        "span_leading_whitespace",
        "span_trailing_whitespace",
        "span_punctuation_included",
        "span_punctuation_excluded",
        "error_primary",
        "generalization_class",
    ]
    present = [column for column in columns if column in events.columns]
    return events.loc[mask, present].reset_index(drop=True)


def confusion_tables(
    events: pd.DataFrame,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return exact-boundary label confusion plus missed/spurious flow table."""
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype="int64")

    # Strict TP diagonal.
    for row in events.loc[events["strict_is_tp"]].itertuples(index=False):
        if row.gold_label in matrix.index and row.pred_label in matrix.columns:
            matrix.loc[row.gold_label, row.pred_label] += 1

    # Exact-boundary label errors are represented twice; count gold side only.
    label_errors = events.loc[
        (events["error_pair_role"] == "GOLD")
        & (events["error_primary"] == "LABEL_ERROR")
    ]
    for row in label_errors.itertuples(index=False):
        if row.gold_label in matrix.index and row.pred_label in matrix.columns:
            matrix.loc[row.gold_label, row.pred_label] += 1

    counts = _matrix_to_long(matrix, "count")
    row_totals = matrix.sum(axis=1).replace(0, np.nan)
    normalized_matrix = matrix.div(row_totals, axis=0).fillna(0.0)
    normalized = _matrix_to_long(normalized_matrix, "proportion")

    extended_rows = []
    for row in _diagnostic_units(events).itertuples(index=False):
        extended_rows.append(
            {
                "gold_label": row.gold_label if bool(row.gold_exists) else "[SPURIOUS]",
                "pred_label": row.pred_label if bool(row.pred_exists) else "[MISSED]",
                "error_type": row.error_primary,
            }
        )
    extended = (
        pd.DataFrame(extended_rows)
        .value_counts(["gold_label", "pred_label", "error_type"])
        .rename("count")
        .reset_index()
        if extended_rows
        else pd.DataFrame(columns=["gold_label", "pred_label", "error_type", "count"])
    )
    return counts, normalized, extended


def document_metrics(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    tags: list[str],
    min_overlap_percentage: float,
) -> pd.DataFrame:
    rows = []
    doc_ids = sorted(set(gold["filename"].astype(str)) | set(predicted["filename"].astype(str)))
    for doc_id in doc_ids:
        result = score_subgroup(
            gold,
            predicted,
            gold["filename"].astype(str) == doc_id,
            predicted["filename"].astype(str) == doc_id,
            tags=tags,
            min_overlap_percentage=min_overlap_percentage,
        )
        rows.append({"document_id": doc_id, **result})
    return pd.DataFrame(rows)


def dataset_shift_labels(training: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(set(training["label"].astype(str)) | set(gold["label"].astype(str)))
    rows = []
    for dataset_name, frame in (("training", training), ("evaluation", gold)):
        counts = frame["label"].astype(str).value_counts()
        total = int(len(frame))
        for label in labels:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "dataset": dataset_name,
                    "label": label,
                    "count": count,
                    "proportion": count / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def dataset_shift_summary(shift: pd.DataFrame) -> pd.DataFrame:
    """Quantify label-distribution shift with Jensen-Shannon distance."""
    if shift.empty:
        return pd.DataFrame()
    pivot = (
        shift.pivot(index="label", columns="dataset", values="proportion")
        .fillna(0.0)
    )
    if not {"training", "evaluation"}.issubset(pivot.columns):
        return pd.DataFrame()
    p = pivot["training"].to_numpy(dtype=float)
    q = pivot["evaluation"].to_numpy(dtype=float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(left, right):
        mask = left > 0
        if not mask.any():
            return 0.0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    js_divergence = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    js_distance = float(np.sqrt(max(js_divergence, 0.0)))
    total_variation = float(0.5 * np.abs(p - q).sum())
    training_labels = int((p > 0).sum())
    evaluation_labels = int((q > 0).sum())
    shared_labels = int(((p > 0) & (q > 0)).sum())
    return pd.DataFrame(
        [
            {
                "jensen_shannon_divergence": js_divergence,
                "jensen_shannon_distance": js_distance,
                "total_variation_distance": total_variation,
                "training_labels": training_labels,
                "evaluation_labels": evaluation_labels,
                "shared_labels": shared_labels,
                "evaluation_label_coverage_of_training": (
                    shared_labels / training_labels if training_labels else np.nan
                ),
            }
        ]
    )


def confidence_summary(predicted: pd.DataFrame) -> pd.DataFrame:
    """Global confidence variability used to decide whether confidence plots are informative."""
    if "score" not in predicted.columns:
        return pd.DataFrame()
    values = pd.to_numeric(predicted["score"], errors="coerce").dropna()
    if values.empty:
        return pd.DataFrame()
    minimum = float(values.min())
    maximum = float(values.max())
    return pd.DataFrame(
        [
            {
                "support": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q3": float(values.quantile(0.75)),
                "min": minimum,
                "max": maximum,
                "range": maximum - minimum,
                "unique_scores": int(values.nunique()),
            }
        ]
    )


def oracle_error_budget(
    events: pd.DataFrame,
    official_metrics: dict[str, float | int],
) -> pd.DataFrame:
    """Counterfactual STRICT ceilings; these are diagnostics, not forecasts."""
    tp = int(events["strict_is_tp"].sum())
    fp = int(events["strict_is_fp"].sum())
    fn = int(events["strict_is_fn"].sum())
    baseline = safe_f1(tp, fp, fn)

    gold_pairs = events.loc[events["error_pair_role"] == "GOLD"]
    pair_counts = Counter(gold_pairs["error_primary"].astype(str))
    missed = int((events["error_primary"] == "MISSED").sum())
    spurious = int((events["error_primary"] == "SPURIOUS").sum())

    scenarios = [
        ("boundary_correction", int(pair_counts.get("BOUNDARY_ERROR", 0)), 0),
        ("label_correction", int(pair_counts.get("LABEL_ERROR", 0)), 0),
        (
            "boundary_and_label_correction",
            int(pair_counts.get("BOUNDARY_AND_LABEL_ERROR", 0)),
            0,
        ),
        ("recover_all_missed", missed, 1),
        ("remove_all_spurious", spurious, 2),
    ]

    rows = []
    for name, count, mode in scenarios:
        new_tp, new_fp, new_fn = tp, fp, fn
        if mode == 0:  # paired FP+FN -> TP
            new_tp += count
            new_fp -= count
            new_fn -= count
        elif mode == 1:  # FN -> TP
            new_tp += count
            new_fn -= count
        else:  # remove FP
            new_fp -= count
        scores = safe_f1(new_tp, new_fp, new_fn)
        rows.append(
            {
                "scenario": name,
                "affected": count,
                "baseline_f1": float(baseline["f1"]),
                "oracle_f1": float(scores["f1"]),
                "delta_f1": float(scores["f1"] - baseline["f1"]),
                "oracle_tp": int(scores["tp"]),
                "oracle_fp": int(scores["fp"]),
                "oracle_fn": int(scores["fn"]),
            }
        )

    all_pair = sum(
        int(pair_counts.get(name, 0))
        for name in ("BOUNDARY_ERROR", "LABEL_ERROR", "BOUNDARY_AND_LABEL_ERROR")
    )
    new_tp = tp + all_pair + missed
    new_fp = max(0, fp - all_pair - spurious)
    new_fn = max(0, fn - all_pair - missed)
    scores = safe_f1(new_tp, new_fp, new_fn)
    rows.append(
        {
            "scenario": "all_diagnostic_errors",
            "affected": all_pair + missed + spurious,
            "baseline_f1": float(baseline["f1"]),
            "oracle_f1": float(scores["f1"]),
            "delta_f1": float(scores["f1"] - baseline["f1"]),
            "oracle_tp": int(scores["tp"]),
            "oracle_fp": int(scores["fp"]),
            "oracle_fn": int(scores["fn"]),
        }
    )
    return pd.DataFrame(rows)


def confidence_by_error(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.loc[events["pred_exists"] & events["pred_confidence"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "error_type",
                "support",
                "mean",
                "std",
                "q1",
                "median",
                "q3",
                "min",
                "max",
            ]
        )
    rows = []
    for error_type, group in frame.groupby("error_primary", dropna=False):
        values = pd.to_numeric(group["pred_confidence"], errors="coerce").dropna()
        rows.append(
            {
                "error_type": str(error_type),
                "support": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "q1": float(values.quantile(0.25)) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "q3": float(values.quantile(0.75)) if len(values) else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confidence_threshold_sweep(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    tags: list[str],
    min_overlap_percentage: float,
) -> pd.DataFrame:
    if "score" not in predicted.columns or predicted.empty:
        return pd.DataFrame()
    scores = pd.to_numeric(predicted["score"], errors="coerce")
    # Fixed thresholds plus score quantiles make the table useful even for
    # narrowly distributed model confidence, as in the supplied example.
    thresholds = sorted(
        set(
            [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
            + [float(value) for value in scores.quantile([0.1, 0.25, 0.5, 0.75, 0.9])]
        )
    )
    rows = []
    for threshold in thresholds:
        result = score_subgroup(
            gold,
            predicted,
            pd.Series(True, index=gold.index),
            scores >= threshold,
            tags=tags,
            min_overlap_percentage=min_overlap_percentage,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "retained_predictions": int((scores >= threshold).sum()),
                "coverage": float((scores >= threshold).mean()),
                **result,
            }
        )
    return pd.DataFrame(rows)


def error_pareto(events: pd.DataFrame) -> pd.DataFrame:
    units = _diagnostic_units(events)
    errors = units.loc[units["error_primary"] != "CORRECT"].copy()
    if errors.empty:
        return pd.DataFrame(columns=["surface", "count", "cumulative_count", "cumulative_proportion"])
    errors["surface"] = errors["gold_text"].where(
        errors["gold_exists"], errors["pred_text"]
    ).astype(str)
    counts = errors["surface"].value_counts().rename_axis("surface").reset_index(name="count")
    counts["cumulative_count"] = counts["count"].cumsum()
    counts["cumulative_proportion"] = counts["cumulative_count"] / counts["count"].sum()
    return counts


def error_intersections(events: pd.DataFrame) -> pd.DataFrame:
    units = _diagnostic_units(events)
    features = [
        "error_fragmentation",
        "error_merging",
        "error_duplicate_prediction",
        "error_nested_entity",
        "error_overlapping_entity",
        "span_one_character_offset",
    ]
    present = [column for column in features if column in units.columns]
    rows = []
    for left_index, left in enumerate(present):
        for right in present[left_index + 1 :]:
            count = int(units[left].fillna(False).astype(bool).mul(units[right].fillna(False).astype(bool)).sum())
            if count:
                rows.append({"feature_a": left, "feature_b": right, "count": count})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "count"])


def bootstrap_intervals(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    gold_exposure: pd.DataFrame,
    pred_exposure: pd.DataFrame,
    *,
    tags: list[str],
    min_overlap_percentage: float,
    samples: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Document-level bootstrap CIs for overall and exposure-specific PRF."""
    if samples <= 0:
        return pd.DataFrame()
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap_confidence must be between 0 and 1.")

    doc_ids = sorted(set(gold["filename"].astype(str)) | set(predicted["filename"].astype(str)))
    if len(doc_ids) < 2:
        return pd.DataFrame(
            [
                {
                    "scope": "ALL",
                    "metric": metric,
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "samples": int(samples),
                    "documents": len(doc_ids),
                    "status": "insufficient_documents",
                }
                for metric in ("strict_precision", "strict_recall", "strict_f1", "char_precision", "char_recall", "char_f1")
            ]
        )

    scopes: list[tuple[str, pd.Series, pd.Series]] = [
        (
            "ALL",
            pd.Series(True, index=gold.index),
            pd.Series(True, index=predicted.index),
        )
    ]
    for class_name in GENERALIZATION_ORDER:
        scopes.append(
            (
                class_name,
                gold_exposure["generalization_class"] == class_name,
                pred_exposure["generalization_class"] == class_name,
            )
        )

    per_scope_doc: dict[str, dict[str, dict[str, int]]] = {}
    for scope, gold_scope, pred_scope in scopes:
        per_doc = {}
        for doc_id in doc_ids:
            result = score_subgroup(
                gold,
                predicted,
                gold_scope & (gold["filename"].astype(str) == doc_id),
                pred_scope & (predicted["filename"].astype(str) == doc_id),
                tags=tags,
                min_overlap_percentage=min_overlap_percentage,
            )
            per_doc[doc_id] = {
                key: int(result[key])
                for key in ("strict_tp", "strict_fp", "strict_fn", "char_tp", "char_fp", "char_fn")
            }
        per_scope_doc[scope] = per_doc

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    rows = []
    for scope, _, _ in scopes:
        distributions = {
            metric: []
            for metric in (
                "strict_precision",
                "strict_recall",
                "strict_f1",
                "char_precision",
                "char_recall",
                "char_f1",
            )
        }
        for _ in range(int(samples)):
            sampled = rng.choice(doc_ids, size=len(doc_ids), replace=True)
            counts = Counter()
            for doc_id in sampled:
                counts.update(per_scope_doc[scope][str(doc_id)])
            strict = safe_f1(counts["strict_tp"], counts["strict_fp"], counts["strict_fn"])
            char = safe_f1(counts["char_tp"], counts["char_fp"], counts["char_fn"])
            distributions["strict_precision"].append(strict["precision"])
            distributions["strict_recall"].append(strict["recall"])
            distributions["strict_f1"].append(strict["f1"])
            distributions["char_precision"].append(char["precision"])
            distributions["char_recall"].append(char["recall"])
            distributions["char_f1"].append(char["f1"])

        # Point estimate from complete corpus for the same scope.
        scope_tuple = next(item for item in scopes if item[0] == scope)
        estimate_counts = score_subgroup(
            gold,
            predicted,
            scope_tuple[1],
            scope_tuple[2],
            tags=tags,
            min_overlap_percentage=min_overlap_percentage,
        )
        for metric, values in distributions.items():
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "estimate": float(estimate_counts[metric]),
                    "ci_low": float(np.quantile(values, alpha)),
                    "ci_high": float(np.quantile(values, 1.0 - alpha)),
                    "samples": int(samples),
                    "documents": len(doc_ids),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _categorical_metrics(
    gold,
    predicted,
    gold_values,
    pred_values,
    values: Iterable,
    *,
    family: str,
    tags: list[str],
    min_overlap_percentage: float,
) -> pd.DataFrame:
    rows = []
    for value in values:
        result = score_subgroup(
            gold,
            predicted,
            gold_values.astype(str) == str(value),
            pred_values.astype(str) == str(value),
            tags=tags,
            min_overlap_percentage=min_overlap_percentage,
        )
        rows.append(
            {
                "family": family,
                "value": str(value),
                **result,
                "low_support": int(result["support"]) < 10,
            }
        )
    return pd.DataFrame(rows)


def _stable_values(first: pd.Series, second: pd.Series) -> tuple[str, ...]:
    values = [str(value) for value in pd.concat([first, second]).dropna().unique()]
    preferred = ["EXACT", "0.95-<1.00", "0.90-<0.95", "0.85-<0.90", "0.80-<0.85", "<0.80", "0", "1", "2-5", "6-10", "11-20", ">20"]
    ordered = [value for value in preferred if value in values]
    ordered.extend(sorted(value for value in values if value not in ordered))
    return tuple(ordered)


def _ordered_length_values(first: pd.Series, second: pd.Series) -> tuple[str, ...]:
    values = {str(value) for value in pd.concat([first, second]).dropna().unique()}
    return tuple(value for value in ("0", "1", "2", "3", "4", "5+") if value in values)


def _matrix_to_long(matrix: pd.DataFrame, value_name: str) -> pd.DataFrame:
    rows = []
    for gold_label in matrix.index:
        for pred_label in matrix.columns:
            rows.append(
                {
                    "gold_label": str(gold_label),
                    "pred_label": str(pred_label),
                    value_name: matrix.loc[gold_label, pred_label],
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_units(events: pd.DataFrame) -> pd.DataFrame:
    """One row per diagnostic entity/relation, avoiding FP/FN pair double-counting."""
    pair_gold = events["error_pair_role"] == "GOLD"
    unpaired = events["error_pair_id"].isna()
    return events.loc[pair_gold | unpaired].copy().reset_index(drop=True)
