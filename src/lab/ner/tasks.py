"""NER's task table, strings only so importing it never loads torch."""

from __future__ import annotations

TASKS: dict[str, tuple[str, str]] = {
    "analyze_evaluation": ("lab.ner.analysis.api", "analyze_evaluation"),
    "train_model": ("lab.ner.training.assessment", "train_model"),
    "search_hyperparameters": ("lab.ner.hpo.search", "search_hyperparameters"),
    "predict_entities": ("lab.ner.inference", "predict_entities"),
}

REQUIRES = ("torch", "datasets", "ray")
