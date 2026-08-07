"""Training a token-classification model on encoded rows."""

from __future__ import annotations

from ner_lab.training.arguments import DEFAULTS, for_inference, training_arguments
from ner_lab.training.assessment import (
    AssessmentResult,
    aggregate_metrics,
    best_epoch_metrics,
    encode_partition,
    fold_rotations,
    read_data_manifest,
    resolve_training_arguments,
    run_directory_name,
    train_model,
)
from ner_lab.training.dataset import ensure_int_list, is_encoded, to_dataset, validate_rows
from ner_lab.training.tracking import EpochMetricsLogger, ResourceTracker, gpu_hardware_info
from ner_lab.training.trainer import (
    CRFTrainer,
    TrainingResult,
    build_summary,
    dataset_summary,
    remove_checkpoints,
    resolve_trainer_class,
    train,
)

__all__ = [
    "DEFAULTS",
    "AssessmentResult",
    "CRFTrainer",
    "EpochMetricsLogger",
    "ResourceTracker",
    "TrainingResult",
    "aggregate_metrics",
    "best_epoch_metrics",
    "build_summary",
    "dataset_summary",
    "encode_partition",
    "ensure_int_list",
    "fold_rotations",
    "for_inference",
    "gpu_hardware_info",
    "is_encoded",
    "read_data_manifest",
    "remove_checkpoints",
    "resolve_trainer_class",
    "resolve_training_arguments",
    "run_directory_name",
    "to_dataset",
    "train",
    "train_model",
    "training_arguments",
    "validate_rows",
]
