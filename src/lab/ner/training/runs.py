"""The run directory and the split it trains on: claiming it, naming it, reading the split."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from lab.core.dataset import DATA_MANIFEST_FILENAME
from lab.core.provenance import read_manifest
from lab.ner.models.registry import Architecture, architecture_name

RUN_MANIFEST_FILENAME = "run_manifest.json"

SPLIT_MODES = ("train_validation", "fixed_holdout_kfold")


def read_data_manifest(split_dir: str | Path) -> dict[str, Any]:
    """Read the `data_manifest.json` that `prepare_dataset` wrote beside a split."""
    manifest_path = Path(split_dir) / DATA_MANIFEST_FILENAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No {DATA_MANIFEST_FILENAME} in {split_dir} — expected the output directory of "
            "lab.core.prepare_dataset."
        )

    return read_manifest(manifest_path)


def fold_rotations(
    data_manifest: dict[str, Any],
    folds: Sequence[int] | None = None,
) -> list[int | None]:
    """
    Which validation folds a split calls for: `[None]` for a plain train/validation split.

    With a fixed holdout, every non-holdout fold takes a turn as validation
    unless `folds` narrows it.
    """
    split = data_manifest.get("split", {})
    mode = split.get("mode")

    if mode == "train_validation":
        if folds is not None:
            raise ValueError(
                "folds only applies to a fixed_holdout_kfold split; this one is "
                "train_validation, which has a single rotation."
            )

        return [None]

    if mode != "fixed_holdout_kfold":
        raise ValueError(f"Unsupported split mode {mode!r}; expected one of {SPLIT_MODES}.")

    n_splits = int(split["n_splits"])
    holdout_fold = int(split["holdout_fold"])
    rotatable = [index for index in range(n_splits) if index != holdout_fold]

    if folds is None:
        return rotatable

    requested = [int(fold) for fold in folds]
    unknown = sorted(set(requested) - set(rotatable))

    if unknown:
        raise ValueError(
            f"folds={unknown} are not rotatable validation folds; this split has "
            f"{n_splits} folds with {holdout_fold} reserved as the fixed holdout."
        )

    return requested


def claim_run_dir(output_dir: str | Path, overwrite: bool = False) -> Path:
    """
    Resolve `output_dir` as the run directory, refusing one a previous run owns.

    `output_dir` is written into directly, so a second run pointed at it would
    replace the first one's manifest, metrics and weights. A `run_manifest.json`
    already there means a run claimed this directory, whether it finished or died,
    and `overwrite` is what says to take it anyway.

    Call this before creating anything, and exactly once per run. It reports a
    directory a run already owns, and every run fills its own directory as it goes
    — a second call partway through would refuse the run's own output.
    """
    run_dir = Path(output_dir).resolve()
    claimed = run_dir / RUN_MANIFEST_FILENAME

    if claimed.exists() and not overwrite:
        stamp = datetime.fromtimestamp(claimed.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        raise FileExistsError(
            f"{run_dir} already holds a run from {stamp}. output_dir is the run "
            f"directory itself, so continuing would overwrite its manifest, metrics "
            f"and weights. Point output_dir at a new directory, or pass "
            f"overwrite=True to replace what is there."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def run_directory_name(
    target_label: str,
    architecture: str | Architecture,
    base_model: str,
    timestamp: str | None = None,
) -> str:
    """
    A directory name describing one run: what was trained, on what, when.

    Nothing calls this on your behalf — `output_dir` is the run directory. Join it
    yourself for a timestamped tree:
    `train_model(output_dir=root / run_directory_name(label, architecture, model))`.
    """
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{target_label}__{architecture_name(architecture)}__{Path(base_model).name}__{stamp}"


def split_provenance(
    validation_index: int | None,
    partitions: dict[str, Path | list[Path]],
) -> dict[str, Any]:
    """What this run actually trained and validated on, for its `training_summary.json`."""
    train = partitions["train"]

    return {
        "validation_fold": validation_index,
        "train_parquet": ([str(path) for path in train] if isinstance(train, list) else str(train)),
        "validation_parquet": str(partitions["validation"]),
    }
