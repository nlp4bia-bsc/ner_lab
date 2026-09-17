"""Applying a split: partitioning a corpus, reporting its balance, and persisting it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.core.assignments import (
    assert_partition_integrity,
    assign_partitions,
    manifest_holdout_index,
    manifest_n_partitions,
    manifest_partition_names,
    manifest_ratios,
    validate_assignments,
)
from lab.core.corpus import DOCUMENT_COLUMNS, validate_corpus
from lab.core.io import DEFAULT_PARQUET_COMPRESSION, read_corpus, write_corpus
from lab.core.stratification import DEFAULT_RANDOM_STATE, parse_document_label_counts

ASSIGNMENTS_FILENAME = "split_assignments.parquet"


@dataclass(frozen=True)
class SplitResult:
    assignments: pd.DataFrame
    summary: pd.DataFrame
    balance_report: pd.DataFrame
    paths: dict[str, Path]


def split_documents(
    documents_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Partition a corpus in memory, keyed by partition name."""
    assert_partition_integrity(assignments_df)

    unassigned = set(documents_df["doc_id"].astype(str)) - set(
        assignments_df["doc_id"].astype(str)
    )

    if unassigned:
        raise ValueError(
            f"{len(unassigned)} document(s) have no partition assignment: "
            f"{sorted(unassigned)[:10]}"
        )

    fold_by_doc_id = assignments_df.set_index(assignments_df["doc_id"].astype(str))["fold"]
    folds = documents_df["doc_id"].astype(str).map(fold_by_doc_id).astype("int16")

    return {
        name: documents_df.loc[folds == fold_id, DOCUMENT_COLUMNS].reset_index(drop=True)
        for fold_id, name in manifest_partition_names(assignments_df).items()
    }


def build_balance_report(
    documents_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-partition document, entity and label counts against their expected values.

    Counts reflect the corpus as annotated. Overlapping entities are resolved at
    encoding time, so these are not the counts any particular training run sees.
    """
    n_partitions = manifest_n_partitions(assignments_df)
    holdout_index = manifest_holdout_index(assignments_df)
    ratios = manifest_ratios(assignments_df)

    labels, label_counts = parse_document_label_counts(documents_df)

    work_df = documents_df[["doc_id", "n_entities"]].merge(
        assignments_df[["doc_id", "fold"]],
        on="doc_id",
        how="left",
        validate="one_to_one",
    )

    if work_df["fold"].isna().any():
        raise ValueError("Some documents do not have a partition assignment.")

    work_df["fold"] = work_df["fold"].astype(int)
    fold_index = range(n_partitions)
    fold_documents = work_df.groupby("fold")["doc_id"].count().reindex(fold_index, fill_value=0)
    fold_entities = work_df.groupby("fold")["n_entities"].sum().reindex(fold_index, fill_value=0)

    fold_label_counts = np.zeros((n_partitions, len(labels)), dtype=np.int64)

    for document_index, fold_id in enumerate(work_df["fold"]):
        fold_label_counts[fold_id] += label_counts[document_index]

    total_documents = len(work_df)
    metrics: dict[str, np.ndarray] = {
        "n_documents": fold_documents.to_numpy(dtype=np.int64),
        "n_entities": fold_entities.to_numpy(dtype=np.int64),
    }
    metrics.update(
        {f"label:{label}": fold_label_counts[:, index] for index, label in enumerate(labels)}
    )

    rows: list[dict[str, Any]] = []

    for metric, counts in metrics.items():
        total_count = int(np.sum(counts))

        for fold_id, count in enumerate(counts):
            n_documents = int(fold_documents.iloc[fold_id])
            expected = (
                total_documents * ratios[fold_id]
                if metric == "n_documents"
                else total_count * n_documents / total_documents
            )
            absolute_difference = float(count - expected)

            rows.append(
                {
                    "fold": fold_id,
                    "partition": _partition_label(fold_id, holdout_index),
                    "metric": metric,
                    "count": int(count),
                    "expected_count": float(expected),
                    "absolute_difference": absolute_difference,
                    "relative_difference": absolute_difference / expected if expected else 0.0,
                }
            )

    return pd.DataFrame(rows).sort_values(["metric", "fold"], kind="stable").reset_index(drop=True)


def create_split(
    corpus: pd.DataFrame | str | Path,
    output_dir: str | Path,
    ratios: Sequence[float],
    holdout_index: int | None = None,
    partition_names: Sequence[str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_restarts: int = 64,
    reuse_assignments: bool = True,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> SplitResult:
    """
    Split a corpus and write the partitions, manifest and balance report.

    `ratios=[0.8, 0.2]` with no holdout writes train and validation. `ratios` of
    `[1/n] * n` with `holdout_index` set writes a permanently reserved test
    partition plus rotatable `fold_XX` partitions.

    An existing manifest in `output_dir` is reused unless `reuse_assignments` is
    False, and is validated against the corpus before being trusted.
    """
    documents_df = validate_corpus(corpus) if isinstance(corpus, pd.DataFrame) else read_corpus(corpus)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / ASSIGNMENTS_FILENAME

    if reuse_assignments and assignments_path.exists():
        assignments_df = validate_assignments(
            documents_df=documents_df,
            assignments_df=pd.read_parquet(assignments_path),
            ratios=ratios,
            holdout_index=holdout_index,
        )
        recorded_names = list(manifest_partition_names(assignments_df).values())

        if partition_names is not None and list(partition_names) != recorded_names:
            raise ValueError(
                "Requested partition_names do not match the persisted manifest "
                f"(source of truth): requested={list(partition_names)}, recorded={recorded_names}."
            )
    else:
        assignments_df = assign_partitions(
            documents_df=documents_df,
            ratios=ratios,
            holdout_index=holdout_index,
            partition_names=partition_names,
            random_state=random_state,
            n_restarts=n_restarts,
        )
        assignments_df.to_parquet(
            assignments_path, engine="pyarrow", compression=compression, index=False
        )

    partitions = split_documents(documents_df, assignments_df)
    resolved_holdout_index = manifest_holdout_index(assignments_df)

    paths: dict[str, Path] = {}
    summary_rows: list[dict[str, Any]] = []

    for fold_id, name in manifest_partition_names(assignments_df).items():
        partition_df = partitions[name]
        paths[name] = write_corpus(partition_df, output_dir / f"{name}.parquet", compression)

        summary_rows.append(
            {
                "fold": fold_id,
                "partition_name": name,
                "partition": _partition_label(fold_id, resolved_holdout_index, in_folds=True),
                "n_documents": len(partition_df),
                "n_entities": int(partition_df["n_entities"].sum()),
                "output_file": str(paths[name]),
            }
        )

    balance_report = build_balance_report(documents_df, assignments_df)
    balance_report.to_parquet(
        output_dir / "split_balance_report.parquet",
        engine="pyarrow",
        compression=compression,
        index=False,
    )
    balance_report.to_csv(output_dir / "split_balance_report.tsv", sep="\t", index=False)

    return SplitResult(
        assignments=assignments_df,
        summary=pd.DataFrame(summary_rows),
        balance_report=balance_report,
        paths=paths,
    )


def split_paths(
    output_dir: str | Path,
    validation_index: int | None = None,
    holdout_index: int | None = None,
) -> dict[str, Path | list[Path]]:
    """
    Resolve the partition files for one experiment from the persisted manifest.

    With a fixed holdout, `validation_index` picks which rotatable fold is
    validation and train becomes every remaining non-holdout fold.
    """
    output_dir = Path(output_dir)
    assignments_path = output_dir / ASSIGNMENTS_FILENAME

    if not assignments_path.exists():
        raise FileNotFoundError(f"No split manifest found at: {assignments_path}")

    assignments_df = pd.read_parquet(assignments_path)
    n_partitions = len(manifest_ratios(assignments_df))
    recorded_holdout_index = manifest_holdout_index(assignments_df)
    names = manifest_partition_names(assignments_df)

    if holdout_index is not None and holdout_index != recorded_holdout_index:
        raise ValueError(
            "Requested holdout_index does not match the persisted manifest "
            f"(source of truth): requested={holdout_index}, recorded={recorded_holdout_index}."
        )

    if recorded_holdout_index is None:
        if validation_index is not None:
            raise ValueError(
                "validation_index is not meaningful for a split with no fixed "
                "holdout_index; there is no fold rotation to select."
            )

        return {name: output_dir / f"{name}.parquet" for name in names.values()}

    if validation_index is None:
        raise ValueError("validation_index is required when the split has a fixed holdout_index.")

    if not 0 <= validation_index < n_partitions:
        raise ValueError(f"validation_index must be between 0 and {n_partitions - 1}.")

    if validation_index == recorded_holdout_index:
        raise ValueError("validation_index cannot be the fixed holdout index.")

    train_folds = [
        fold_id
        for fold_id in range(n_partitions)
        if fold_id not in {recorded_holdout_index, validation_index}
    ]

    return {
        "train": [output_dir / f"{names[fold_id]}.parquet" for fold_id in train_folds],
        "validation": output_dir / f"{names[validation_index]}.parquet",
        "test": output_dir / f"{names[recorded_holdout_index]}.parquet",
    }


def read_split(
    output_dir: str | Path,
    validation_index: int | None = None,
    holdout_index: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load one experiment's partitions as DataFrames, asserting they stay disjoint."""
    paths = split_paths(output_dir, validation_index=validation_index, holdout_index=holdout_index)
    loaded: dict[str, pd.DataFrame] = {}

    for name, path_or_paths in paths.items():
        if isinstance(path_or_paths, list):
            loaded[name] = pd.concat(
                [pd.read_parquet(path) for path in path_or_paths], ignore_index=True
            )
        else:
            loaded[name] = pd.read_parquet(path_or_paths)

    id_sets = {name: set(df["doc_id"].astype(str)) for name, df in loaded.items()}
    names = list(id_sets)

    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if id_sets[first] & id_sets[second]:
                raise AssertionError(f"{first} and {second} are not disjoint.")

    return loaded


def _partition_label(fold_id: int, holdout_index: int | None, in_folds: bool = False) -> str:
    if holdout_index is None:
        return "split"

    if fold_id == holdout_index:
        return "fixed_test"

    return "cross_validation_fold" if in_folds else "cross_validation"
