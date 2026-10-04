"""A finished sweep's trials: their directories, the per-trial table, and the best one."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

TRIALS_SUMMARY_FILENAME = "trials_summary.parquet"

RESOURCE_COLUMNS = (
    "duration_sec",
    "energy_kwh",
    "emissions_kg_co2",
    "cpu_utilization_percent",
    "gpu_utilization_percent",
    "ram_used_gb",
    "gpu_peak_vram_allocated_gb",
    "gpu_peak_vram_reserved_gb",
)


def trial_directory_name(trial: Any) -> str:
    """A readable per-trial directory: the id plus the dimensions that vary most."""
    configuration = trial.config

    return (
        f"trial{trial.trial_id}"
        f"_lr{configuration.get('learning_rate', 0):.1e}"
        f"_bs{configuration.get('effective_train_batch_size', 0)}"
        f"_wd{configuration.get('weight_decay', 0):.3f}"
    )


def trials_table(result_grid: Any, metric_key: str) -> pd.DataFrame:
    """
    One row per trial: sampled values, score and spread, batch resolution, resources.

    Everything needed to compare trials without opening each trial's own
    artifacts individually.
    """
    reported_columns = [
        metric_key,
        f"{metric_key}_std",
        "n_seeds",
        "per_seed_scores",
        "oom",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        *RESOURCE_COLUMNS,
    ]

    rows: list[dict[str, Any]] = []

    for result in result_grid:
        metrics = result.metrics or {}
        row: dict[str, Any] = {
            "trial_id": metrics.get("trial_id"),
            "trial_path": result.path,
            "error": str(result.error) if result.error else None,
        }
        row.update(result.config or {})

        for column in reported_columns:
            row[column] = metrics.get(column)

        rows.append(row)

    return pd.DataFrame(rows)


def write_trials_table(trials: pd.DataFrame, path: str | Path) -> Path:
    """Write the per-trial table to parquet."""
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    trials.to_parquet(table_path, index=False)

    return table_path


def best_trial(result_grid: Any, metric_key: str, greater_is_better: bool = True) -> Any | None:
    """The finished trial with the best finite score, or None when there is none."""
    scored = []

    for result in result_grid:
        value = (result.metrics or {}).get(metric_key)

        if value is not None and math.isfinite(value):
            scored.append((value, result))

    if not scored:
        return None

    scored.sort(key=lambda pair: pair[0], reverse=greater_is_better)

    return scored[0][1]


def optional_int(value: Any) -> int | None:
    """Coerce a metric Ray reported back to a plain `int`, leaving a missing one alone."""
    return None if value is None else int(value)
