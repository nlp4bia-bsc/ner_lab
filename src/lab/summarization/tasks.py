"""Summarization's task table, strings only so importing it never loads its dependencies."""

from __future__ import annotations

TASKS: dict[str, tuple[str, str]] = {
    "evaluate_summaries": ("lab.summarization.evaluation.evaluate", "evaluate_summaries"),
}

REQUIRES = ("torch", "rouge_score", "bert_score")
