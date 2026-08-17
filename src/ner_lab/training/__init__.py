"""Training a token-classification model on encoded rows."""

from __future__ import annotations

from ner_lab.training.arguments import (
    DEFAULTS,
    default_precision,
    for_inference,
    training_arguments,
)
from ner_lab.training.assessment import (
    AssessmentResult,
    aggregate_metrics,
    best_epoch_metrics,
    claim_run_dir,
    encode_partition,
    fold_rotations,
    model_encoding,
    read_data_manifest,
    resolve_training_arguments,
    run_directory_name,
    train_model,
)
from ner_lab.training.dataset import ensure_int_list, is_encoded, to_dataset, validate_rows
from ner_lab.training.devices import (
    DevicePolicy,
    apply_device_policy,
    effective_train_batch_size,
    multi_device_message,
    require_single_device,
)
from ner_lab.training.tracking import EpochMetricsLogger, ResourceTracker, gpu_hardware_info
from ner_lab.training.trainer import (
    CRFTrainer,
    TrainingResult,
    build_summary,
    dataset_summary,
    read_model_encoding,
    remove_checkpoints,
    resolve_trainer_class,
    train,
    write_model_encoding,
)

__all__ = [
    "DEFAULTS",
    "AssessmentResult",
    "CRFTrainer",
    "DevicePolicy",
    "EpochMetricsLogger",
    "ResourceTracker",
    "TrainingResult",
    "aggregate_metrics",
    "apply_device_policy",
    "best_epoch_metrics",
    "build_summary",
    "claim_run_dir",
    "dataset_summary",
    "default_precision",
    "effective_train_batch_size",
    "encode_partition",
    "ensure_int_list",
    "fold_rotations",
    "for_inference",
    "gpu_hardware_info",
    "is_encoded",
    "model_encoding",
    "multi_device_message",
    "read_data_manifest",
    "read_model_encoding",
    "remove_checkpoints",
    "require_single_device",
    "resolve_trainer_class",
    "resolve_training_arguments",
    "run_directory_name",
    "to_dataset",
    "train",
    "train_model",
    "training_arguments",
    "validate_rows",
    "write_model_encoding",
]
