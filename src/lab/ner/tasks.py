"""NER's task table, strings only so importing it never loads torch."""

from __future__ import annotations

TASKS: dict[str, tuple[str, str]] = {
    "train_model": ("lab.ner.training.assessment", "train_model"),
    "search_hyperparameters": ("lab.ner.hpo.search", "search_hyperparameters"),
    "predict_entities": ("lab.ner.inference.predict", "predict_entities"),
    "fit_ensemble": ("lab.ner.ensemble.fit", "fit_ensemble"),
    "predict_ensemble": ("lab.ner.ensemble.predict", "predict_ensemble"),
}

REQUIRES = ("torch", "datasets", "ray")
