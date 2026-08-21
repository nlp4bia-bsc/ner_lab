"""Corpus-level data handling: source formats, the canonical schema, and splitting."""

from __future__ import annotations

from ner_lab.data.assignments import (
    assert_partition_integrity,
    assign_partitions,
    manifest_holdout_index,
    manifest_partition_names,
    manifest_ratios,
    validate_assignments,
)
from ner_lab.data.brat import read_ann, read_annotation_tsv, read_annotations, resolve_documents
from ner_lab.data.corpus import (
    ConflictPolicy,
    DOCUMENT_COLUMNS,
    MismatchPolicy,
    SourceConflict,
    SourceMismatch,
    build_corpus,
    count_labels,
    document_fingerprints,
    resolve_conflicts,
    resolve_mismatch,
    validate_corpus,
)
from ner_lab.data.dataset import (
    PreparedDataset,
    derive_dataset_name,
    prepare_dataset,
    read_dataset_metadata,
    split_descriptor,
)
from ner_lab.data.io import read_corpus, write_corpus
from ner_lab.data.labels import (
    CANONICAL_LABELS,
    LABEL_ALIASES,
    normalize_annotation_labels,
    normalize_entity_labels,
    normalize_label,
)
from ner_lab.data.split import (
    SplitResult,
    build_balance_report,
    create_split,
    read_split,
    split_documents,
    split_paths,
)
from ner_lab.data.stats import compute_annotation_stats, compute_text_stats
from ner_lab.data.stratification import parse_document_label_counts

__all__ = [
    "CANONICAL_LABELS",
    "DOCUMENT_COLUMNS",
    "LABEL_ALIASES",
    "ConflictPolicy",
    "MismatchPolicy",
    "PreparedDataset",
    "SourceConflict",
    "SourceMismatch",
    "SplitResult",
    "assert_partition_integrity",
    "assign_partitions",
    "build_balance_report",
    "build_corpus",
    "compute_annotation_stats",
    "compute_text_stats",
    "count_labels",
    "create_split",
    "derive_dataset_name",
    "document_fingerprints",
    "manifest_holdout_index",
    "manifest_partition_names",
    "manifest_ratios",
    "normalize_annotation_labels",
    "normalize_entity_labels",
    "normalize_label",
    "parse_document_label_counts",
    "prepare_dataset",
    "read_ann",
    "read_annotation_tsv",
    "read_annotations",
    "read_corpus",
    "read_dataset_metadata",
    "read_split",
    "resolve_documents",
    "resolve_conflicts",
    "resolve_mismatch",
    "split_descriptor",
    "split_documents",
    "split_paths",
    "validate_assignments",
    "validate_corpus",
    "write_corpus",
]
