"""Preparing a dataset end to end: source corpus, canonical parquet, split, manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ner_lab.data.corpus import build_corpus
from ner_lab.data.io import DEFAULT_PARQUET_COMPRESSION, read_corpus, write_corpus
from ner_lab.data.split import ASSIGNMENTS_FILENAME, SplitResult, create_split
from ner_lab.data.stratification import DEFAULT_RANDOM_STATE
from ner_lab.provenance import file_sha256, read_manifest, write_manifest

GENERIC_DOCUMENT_DIRNAMES = {"txt", "text", "documents"}

SOURCE_MANIFEST_FILENAME = "source_manifest.json"
DATA_MANIFEST_FILENAME = "data_manifest.json"


@dataclass(frozen=True)
class PreparedDataset:
    corpus: pd.DataFrame
    corpus_path: Path
    dataset_root: Path
    source_manifest: dict[str, Any]
    split: SplitResult | None = None
    split_dir: Path | None = None
    data_manifest: dict[str, Any] | None = None


def read_dataset_metadata(source_parquet: str | Path) -> dict[str, Any] | None:
    """Read the `metadata.json` sitting next to a parquet corpus, if there is one."""
    metadata_path = Path(source_parquet).resolve().parent / "metadata.json"

    return read_manifest(metadata_path) if metadata_path.exists() else None


def find_split_metadata(metadata: dict[str, Any], filename: str) -> dict[str, Any] | None:
    """Find the `splits` entry in a dataset metadata file that describes `filename`."""
    for split_info in metadata.get("splits", {}).values():
        if split_info.get("file") == filename:
            return split_info

    return None


def derive_dataset_name(
    documents: str | Path | None = None,
    source_parquet: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Derive a dataset name from whatever identifies the source.

    Prefers the name recorded in a sibling `metadata.json`, then the documents
    directory (its parent when the directory is a generic `txt/`-style holder),
    then the parquet filename.
    """
    if metadata is not None and metadata.get("dataset_name"):
        return str(metadata["dataset_name"])

    if documents is not None:
        documents_path = Path(documents).resolve()

        if documents_path.name.lower() in GENERIC_DOCUMENT_DIRNAMES:
            return documents_path.parent.name

        return documents_path.name

    if source_parquet is not None:
        return Path(source_parquet).stem

    raise ValueError("Cannot derive a dataset name without documents or source_parquet.")


def split_descriptor(
    validation_size: float = 0.2,
    kfolds: int | None = None,
    holdout_fold: int = 0,
) -> str:
    """Directory name describing one split variant."""
    if kfolds:
        return f"kfold_{kfolds}_holdout_{holdout_fold}"

    validation_percentage = round(validation_size * 100)

    return f"train_val_{100 - validation_percentage}_{validation_percentage}"


def prepare_dataset(
    output_dir: str | Path,
    documents: str | Path | None = None,
    annotations: pd.DataFrame | str | Path | None = None,
    source_parquet: str | Path | None = None,
    dataset_name: str | None = None,
    normalize_labels: bool = True,
    split: bool = True,
    validation_size: float = 0.2,
    kfolds: int | None = None,
    holdout_fold: int = 0,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_restarts: int = 64,
    reuse_assignments: bool = True,
    corpus_filename: str = "documents.parquet",
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> PreparedDataset:
    """
    Convert a source corpus to the canonical parquet, split it, and record provenance.

    Give either `documents` plus `annotations` (BRAT-style input) or an existing
    canonical `source_parquet`. Output goes to
    `<output_dir>/<dataset_name>/<split_descriptor>/`, the layout the training
    stage expects; pass `dataset_name` to override the derived one.

    Set `split=False` to write only the canonical corpus. Set `kfolds` for
    fixed-holdout cross-validation instead of a train/validation split.
    """
    _validate_inputs(documents, annotations, source_parquet, kfolds, holdout_fold)

    metadata = read_dataset_metadata(source_parquet) if source_parquet is not None else None

    if dataset_name is None:
        dataset_name = derive_dataset_name(documents, source_parquet, metadata)

    dataset_root = Path(output_dir) / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    if source_parquet is not None:
        corpus_path = Path(source_parquet)
        corpus = read_corpus(corpus_path)
    else:
        corpus = build_corpus(documents, annotations, normalize_labels=normalize_labels)
        corpus_path = write_corpus(corpus, dataset_root / corpus_filename, compression)

    source_manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "documents": str(documents) if documents is not None else None,
        "annotations": str(annotations) if isinstance(annotations, (str, Path)) else None,
        "source_parquet": str(corpus_path),
        "source_parquet_sha256": file_sha256(corpus_path),
        "normalize_labels": normalize_labels if source_parquet is None else None,
        "n_documents": len(corpus),
        "n_entities": int(corpus["n_entities"].sum()),
        "upstream_metadata": _upstream_metadata(source_parquet, metadata),
    }
    write_manifest(source_manifest, dataset_root / SOURCE_MANIFEST_FILENAME)

    if not split:
        return PreparedDataset(
            corpus=corpus,
            corpus_path=corpus_path,
            dataset_root=dataset_root,
            source_manifest=source_manifest,
        )

    split_dir = dataset_root / split_descriptor(validation_size, kfolds, holdout_fold)
    reused = reuse_assignments and (split_dir / ASSIGNMENTS_FILENAME).exists()

    split_result = create_split(
        corpus=corpus,
        output_dir=split_dir,
        ratios=[1.0 / kfolds] * kfolds if kfolds else [1.0 - validation_size, validation_size],
        holdout_index=holdout_fold if kfolds else None,
        random_state=random_state,
        n_restarts=n_restarts,
        reuse_assignments=reuse_assignments,
        compression=compression,
    )

    data_manifest: dict[str, Any] = {
        "source_manifest": str(dataset_root / SOURCE_MANIFEST_FILENAME),
        "random_state": random_state,
        "reused_existing_split_assignments": reused,
        "split": {
            "mode": "fixed_holdout_kfold" if kfolds else "train_validation",
            "n_splits": kfolds,
            "holdout_fold": holdout_fold if kfolds else None,
            "validation_size": None if kfolds else validation_size,
            "partitions": split_result.summary.to_dict(orient="records"),
        },
    }
    write_manifest(data_manifest, split_dir / DATA_MANIFEST_FILENAME)

    return PreparedDataset(
        corpus=corpus,
        corpus_path=corpus_path,
        dataset_root=dataset_root,
        source_manifest=source_manifest,
        split=split_result,
        split_dir=split_dir,
        data_manifest=data_manifest,
    )


def _validate_inputs(
    documents: str | Path | None,
    annotations: pd.DataFrame | str | Path | None,
    source_parquet: str | Path | None,
    kfolds: int | None,
    holdout_fold: int,
) -> None:
    if (documents is None) == (source_parquet is None):
        raise ValueError("Give either documents (with annotations) or source_parquet, not both.")

    if documents is not None and annotations is None:
        raise ValueError("annotations is required when documents is given.")

    if source_parquet is not None and annotations is not None:
        raise ValueError("annotations only applies to documents input.")

    if kfolds is not None and kfolds < 2:
        raise ValueError("kfolds must be at least 2; omit it for a train/validation split.")

    if kfolds is None and holdout_fold:
        raise ValueError("holdout_fold only applies with kfolds.")


def _upstream_metadata(
    source_parquet: str | Path | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if metadata is None or source_parquet is None:
        return None

    source_parquet = Path(source_parquet)

    return {
        "metadata_json_path": str(source_parquet.resolve().parent / "metadata.json"),
        "dataset_name": metadata.get("dataset_name"),
        "language": metadata.get("language"),
        "matched_split": find_split_metadata(metadata, source_parquet.name),
    }
