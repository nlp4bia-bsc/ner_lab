"""Clinical Named Entity Recognition: encoding, models, training, search, inference, analysis."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "Encoder": "lab.ner.encoding",
    "analyze_predictions": "lab.ner.analysis",
    "build_compute_metrics": "lab.ner.evaluation",
    "build_model": "lab.ner.models",
    "evaluate_predictions": "lab.ner.evaluation",
    "fit_ensemble": "lab.ner.ensemble",
    "predict_ensemble": "lab.ner.ensemble",
    "predict_entities": "lab.ner.inference.predict",
    "predict_multihead": "lab.ner.multihead",
    "search_hyperparameters": "lab.ner.hpo",
    "search_multihead": "lab.ner.multihead",
    "train": "lab.ner.training",
    "train_model": "lab.ner.training",
    "train_multihead": "lab.ner.multihead",
    "training_arguments": "lab.ner.training",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)

    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
