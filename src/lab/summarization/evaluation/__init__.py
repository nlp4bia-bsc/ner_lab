"""Evaluating generated summaries: ROUGE, BERTScore and LLM judges, each reported on its own."""

from __future__ import annotations

from lab.summarization.evaluation.bertscore import score_bertscore
from lab.summarization.evaluation.evaluate import SummaryEvaluation, evaluate_summaries
from lab.summarization.evaluation.judge_model import JudgeModel, load_judge_model
from lab.summarization.evaluation.judges import (
    CARE_DIMENSIONS,
    JUDGES,
    build_prompt,
    parse_judgement,
)
from lab.summarization.evaluation.judging import judge_summaries
from lab.summarization.evaluation.rouge import score_rouge
from lab.summarization.evaluation.scores import describe_scores, merge_scores

__all__ = [
    "CARE_DIMENSIONS",
    "JUDGES",
    "JudgeModel",
    "SummaryEvaluation",
    "build_prompt",
    "describe_scores",
    "evaluate_summaries",
    "judge_summaries",
    "load_judge_model",
    "merge_scores",
    "parse_judgement",
    "score_bertscore",
    "score_rouge",
]
