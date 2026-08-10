"""End-to-end library for clinical Named Entity Recognition."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_EXPORTS: dict[str, str] = {
    "Encoder": "ner_lab.encoding",
    "build_compute_metrics": "ner_lab.evaluation",
    "build_model": "ner_lab.models",
    "evaluate_predictions": "ner_lab.evaluation",
    "predict_entities": "ner_lab.inference",
    "prepare_dataset": "ner_lab.data",
    "search_hyperparameters": "ner_lab.hpo",
    "train": "ner_lab.training",
    "train_model": "ner_lab.training",
    "training_arguments": "ner_lab.training",
}

if TYPE_CHECKING:
    from ner_lab.data import prepare_dataset
    from ner_lab.encoding import Encoder
    from ner_lab.evaluation import build_compute_metrics, evaluate_predictions
    from ner_lab.hpo import search_hyperparameters
    from ner_lab.inference import predict_entities
    from ner_lab.models import build_model
    from ner_lab.training import train, train_model, training_arguments

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
