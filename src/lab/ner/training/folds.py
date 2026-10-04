"""Per-fold metrics of a training run and their aggregate across folds."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

FOLD_METRICS_FILENAME = "fold_metrics.parquet"


def fold_row(
    validation_index: int | None,
    summary: dict[str, Any],
    epoch_metrics: pd.DataFrame,
    greater_is_better: bool = True,
) -> dict[str, Any]:
    """
    One row of `fold_metrics.parquet`: the selection metric, the epoch behind it, and cost.

    `best_metric` is stored under its own fixed key rather than looked up by
    name, so aggregating across folds never has to guess the Trainer's prefix.
    """
    best = summary.get("best", {})
    metric = best.get("metric") or ""

    row: dict[str, Any] = {
        "validation_fold": validation_index,
        "best_metric": best.get("value"),
        "best_epoch": best.get("epoch"),
    }
    row.update(best_epoch_metrics(epoch_metrics, metric, greater_is_better))
    row.update(summary.get("resources") or {})

    return row


def best_epoch_metrics(
    epoch_metrics: pd.DataFrame,
    metric: str,
    greater_is_better: bool = True,
) -> dict[str, Any]:
    """
    The evaluation row of whichever epoch scored best on `metric`.

    `EpochMetricsLogger` strips the `eval_` prefix the Trainer adds, so a
    `metric_for_best_model` given either way resolves to the same column.
    """
    if epoch_metrics.empty:
        return {}

    evaluated = epoch_metrics
    if "split" in epoch_metrics.columns:
        evaluated = epoch_metrics[epoch_metrics["split"] == "eval"]

    column = metric.removeprefix("eval_")

    if evaluated.empty or column not in evaluated.columns:
        return {}

    index = evaluated[column].idxmax() if greater_is_better else evaluated[column].idxmin()

    return {
        name: value for name, value in evaluated.loc[index].to_dict().items() if name != "split"
    }


def aggregate_metrics(
    frame: pd.DataFrame,
    exclude: Sequence[str] = (),
) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of every numeric column, across runs."""
    if frame.empty:
        return {}

    excluded = set(exclude)
    numeric = frame.select_dtypes(include="number")

    return {
        column: {
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std()) if len(numeric) > 1 else 0.0,
        }
        for column in numeric.columns
        if column not in excluded
    }


def write_fold_metrics(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the per-run metric table to parquet."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(metrics_path, index=False)

    return metrics_path
