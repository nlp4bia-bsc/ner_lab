"""The split assignment manifest: which document belongs to which partition."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd

from ner_lab.data.corpus import document_fingerprints
from ner_lab.data.stratification import (
    DEFAULT_RANDOM_STATE,
    normalize_ratios,
    parse_document_label_counts,
    stratify_documents,
)

ASSIGNMENT_DTYPES = {
    "doc_id": "string",
    "document_fingerprint": "string",
    "n_entities": "int32",
    "fold": "int16",
    "strategy": "string",
    "ratios": "string",
    "n_partitions": "int16",
    "random_state": "int64",
    "n_restarts": "int16",
    "stratification_objective": "float64",
    "holdout_index": "Int16",
    "partition_name": "string",
}


def assign_partitions(
    documents_df: pd.DataFrame,
    ratios: Sequence[float],
    holdout_index: int | None = None,
    partition_names: Sequence[str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_restarts: int = 64,
) -> pd.DataFrame:
    """
    Build the assignment manifest for a corpus, stratified by entity label counts.

    The manifest is self-describing: it carries the ratios, holdout index,
    partition names and per-document fingerprints needed to validate or reuse it
    later, so nothing downstream has to be told them again.
    """
    normalized_ratios = normalize_ratios(ratios)
    n_partitions = len(normalized_ratios)

    validate_holdout_index(holdout_index, n_partitions)
    resolved_names = resolve_partition_names(n_partitions, holdout_index, partition_names)

    _, label_counts = parse_document_label_counts(documents_df)
    folds, objective = stratify_documents(
        label_counts=label_counts,
        ratios=normalized_ratios,
        random_state=random_state,
        n_restarts=n_restarts,
    )

    manifest = documents_df[["doc_id", "n_entities"]].copy()
    manifest["document_fingerprint"] = document_fingerprints(documents_df)
    manifest["fold"] = folds
    manifest["strategy"] = "entity_count_stratified_partition"
    manifest["ratios"] = json.dumps(normalized_ratios)
    manifest["n_partitions"] = n_partitions
    manifest["random_state"] = int(random_state)
    manifest["n_restarts"] = int(n_restarts)
    manifest["stratification_objective"] = objective
    manifest["holdout_index"] = pd.NA if holdout_index is None else int(holdout_index)
    manifest["partition_name"] = manifest["fold"].map(dict(enumerate(resolved_names)))

    return manifest[list(ASSIGNMENT_DTYPES)].astype(ASSIGNMENT_DTYPES).reset_index(drop=True)


def validate_assignments(
    documents_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    ratios: Sequence[float] | None = None,
    holdout_index: int | None = None,
) -> pd.DataFrame:
    """
    Check a persisted manifest against the corpus it claims to describe.

    The manifest is the source of truth: `ratios` and `holdout_index`, when given,
    are checked against what it recorded rather than overriding it. Raises if any
    document has been added, removed or edited since the split was made.
    """
    missing_columns = set(ASSIGNMENT_DTYPES) - set(assignments_df.columns)

    if missing_columns:
        raise ValueError(f"Assignment manifest misses required columns: {sorted(missing_columns)}")

    manifest = assignments_df.copy()
    manifest["doc_id"] = manifest["doc_id"].astype("string")
    manifest["document_fingerprint"] = manifest["document_fingerprint"].astype("string")

    if manifest["doc_id"].duplicated().any():
        raise ValueError("Assignment manifest contains duplicated doc_id values.")

    source = dict(
        zip(documents_df["doc_id"].astype(str), document_fingerprints(documents_df).astype(str))
    )
    assigned = dict(
        zip(manifest["doc_id"].astype(str), manifest["document_fingerprint"].astype(str))
    )

    missing_ids = sorted(set(source) - set(assigned))
    unknown_ids = sorted(set(assigned) - set(source))
    changed_ids = sorted(
        doc_id for doc_id in set(source) & set(assigned) if source[doc_id] != assigned[doc_id]
    )

    if missing_ids or unknown_ids or changed_ids:
        raise ValueError(
            "Persisted assignments do not match the corpus. "
            f"Missing: {missing_ids[:5]} | Unknown: {unknown_ids[:5]} | Changed: {changed_ids[:5]}"
        )

    recorded_ratios = manifest_ratios(manifest)

    if ratios is not None:
        expected_ratios = normalize_ratios(ratios)

        if len(recorded_ratios) != len(expected_ratios) or not all(
            abs(recorded - expected) < 1e-9
            for recorded, expected in zip(recorded_ratios, expected_ratios)
        ):
            raise ValueError(
                "Existing assignments use different ratios: "
                f"recorded={recorded_ratios}, requested={expected_ratios}."
            )

    observed_folds = set(manifest["fold"].astype(int))
    expected_folds = set(range(len(recorded_ratios)))

    if observed_folds != expected_folds:
        raise ValueError(f"Expected folds {sorted(expected_folds)}, found {sorted(observed_folds)}.")

    recorded_holdout_index = manifest_holdout_index(manifest)

    if holdout_index is not None and holdout_index != recorded_holdout_index:
        raise ValueError(
            "Requested holdout_index does not match the persisted manifest "
            f"(source of truth): requested={holdout_index}, recorded={recorded_holdout_index}."
        )

    if recorded_holdout_index is not None and recorded_holdout_index not in observed_folds:
        raise ValueError(f"holdout_index={recorded_holdout_index} is unavailable.")

    return manifest


def assert_partition_integrity(assignments_df: pd.DataFrame) -> None:
    """
    Verify the partitions are disjoint and together cover every document.

    With a fixed holdout, every possible choice of validation fold among the
    rotatable partitions is checked against the test fold and the rest.
    """
    if assignments_df["doc_id"].duplicated().any():
        raise AssertionError("A document has more than one partition assignment.")

    n_partitions = manifest_n_partitions(assignments_df)
    holdout_index = manifest_holdout_index(assignments_df)
    all_ids = set(assignments_df["doc_id"].astype(str))

    def ids_where(mask: pd.Series) -> set[str]:
        return set(assignments_df.loc[mask, "doc_id"].astype(str))

    if holdout_index is None:
        seen: set[str] = set()

        for fold_id in range(n_partitions):
            fold_ids = ids_where(assignments_df["fold"] == fold_id)

            if fold_ids & seen:
                raise AssertionError(f"Overlap detected involving fold {fold_id}.")

            seen |= fold_ids

        if seen != all_ids:
            raise AssertionError("Incomplete coverage across partitions.")

        return

    test_ids = ids_where(assignments_df["fold"] == holdout_index)

    for validation_fold in range(n_partitions):
        if validation_fold == holdout_index:
            continue

        validation_ids = ids_where(assignments_df["fold"] == validation_fold)
        train_ids = ids_where(~assignments_df["fold"].isin([holdout_index, validation_fold]))

        if test_ids & validation_ids or test_ids & train_ids or validation_ids & train_ids:
            raise AssertionError(f"Overlap detected for validation fold {validation_fold}.")

        if test_ids | validation_ids | train_ids != all_ids:
            raise AssertionError(f"Incomplete coverage for validation fold {validation_fold}.")


def validate_holdout_index(holdout_index: int | None, n_partitions: int) -> None:
    """Check a holdout index is in range and leaves at least two rotatable folds."""
    if holdout_index is None:
        return

    if not 0 <= holdout_index < n_partitions:
        raise ValueError(f"holdout_index must be between 0 and {n_partitions - 1}.")

    if n_partitions < 3:
        raise ValueError(
            "At least 3 partitions are required when using a fixed holdout_index "
            "(1 test fold + at least 2 rotatable folds)."
        )


def resolve_partition_names(
    n_partitions: int,
    holdout_index: int | None,
    partition_names: Sequence[str] | None = None,
) -> list[str]:
    """Resolve the output name of each partition index."""
    if partition_names is not None:
        if len(partition_names) != n_partitions:
            raise ValueError(
                f"partition_names must have length {n_partitions}, received {len(partition_names)}."
            )

        if len(set(partition_names)) != len(partition_names):
            raise ValueError("partition_names must be unique.")

        return list(partition_names)

    if holdout_index is not None:
        return [
            "test" if index == holdout_index else f"fold_{index:02d}"
            for index in range(n_partitions)
        ]

    if n_partitions == 2:
        return ["train", "validation"]

    raise ValueError(
        "partition_names must be provided explicitly when holdout_index is None "
        "and there are more than 2 partitions."
    )


def manifest_ratios(assignments_df: pd.DataFrame) -> list[float]:
    """The normalized ratios the manifest was built with."""
    return json.loads(str(assignments_df["ratios"].iloc[0]))


def manifest_n_partitions(assignments_df: pd.DataFrame) -> int:
    """The number of partitions the manifest describes."""
    observed = set(assignments_df["n_partitions"].astype(int))

    if len(observed) != 1:
        raise ValueError(f"Assignment manifest has inconsistent n_partitions: {sorted(observed)}")

    return observed.pop()


def manifest_holdout_index(assignments_df: pd.DataFrame) -> int | None:
    """The fixed holdout fold index, or None for a plain split."""
    recorded = assignments_df["holdout_index"].iloc[0]

    return None if pd.isna(recorded) else int(recorded)


def manifest_partition_names(assignments_df: pd.DataFrame) -> dict[int, str]:
    """Partition name by fold index, in fold order."""
    pairs = assignments_df[["fold", "partition_name"]].drop_duplicates().sort_values("fold")

    return {int(fold): str(name) for fold, name in pairs.itertuples(index=False)}
