"""Training a token-classification model on encoded rows."""

from __future__ import annotations

from ner_lab.training.arguments import DEFAULTS, for_inference, training_arguments
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
    "CRFTrainer",
    "EpochMetricsLogger",
    "ResourceTracker",
    "TrainingResult",
    "build_summary",
    "dataset_summary",
    "ensure_int_list",
    "for_inference",
    "gpu_hardware_info",
    "is_encoded",
    "remove_checkpoints",
    "resolve_trainer_class",
    "to_dataset",
    "train",
    "training_arguments",
    "validate_rows",
]
