"""Running the LLM judges over a summaries table: one judgement per summary and judge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from lab.summarization.evaluation.judges import JUDGES, Judge, build_prompt, parse_judgement

if TYPE_CHECKING:
    from lab.summarization.evaluation.judge_model import JudgeModel

JUDGEMENT_COLUMNS = (
    "doc_id",
    "language",
    "judge",
    "raw_score",
    "score",
    "confidence",
    "reasoning",
    "flags_json",
    "dimensions_json",
    "truncated",
    "response",
)


def judge_summaries(
    summaries_df: pd.DataFrame,
    judge_model: JudgeModel,
    judges: Sequence[Judge] = JUDGES,
) -> pd.DataFrame:
    """
    Ask every judge about every summary, and read each answer into a judgement row.

    All prompts — summaries times judges — go to `judge_model` in one call, which
    is what lets vLLM batch them. `judge_model` is a `JudgeModel`, or anything with
    its `prepare(prompt)` and `generate(prompt_token_ids)`.

    A row is one summary and one judge: the raw and normalised score, the
    confidence, the reasoning, the flags and, for completeness, the six CARE
    dimensions as JSON, whether the prompt was cut to fit (`truncated`), and the
    model's full `response`. An unreadable answer scores 0 and is flagged
    `llm_output_malformed`, as `parse_judgement` describes.
    """
    unknown = [judge for judge in judges if judge not in JUDGES]

    if unknown:
        raise ValueError(f"Unknown judges {unknown}. Available judges: {JUDGES}.")

    requests = [
        (
            row.doc_id,
            row.language,
            judge,
            build_prompt(judge, row.text, row.generated_summary, row.language),
        )
        for row in summaries_df.itertuples(index=False)
        for judge in judges
    ]
    prepared = [judge_model.prepare(prompt) for *_, prompt in requests]
    responses = judge_model.generate([token_ids for token_ids, _ in prepared])
    rows = []

    for (doc_id, language, judge, _), (_, truncated), response in zip(
        requests, prepared, responses
    ):
        judgement = parse_judgement(judge, response)
        rows.append(
            {
                "doc_id": doc_id,
                "language": language,
                "judge": judge,
                "raw_score": judgement["raw_score"],
                "score": judgement["score"],
                "confidence": judgement["confidence"],
                "reasoning": judgement["reasoning"],
                "flags_json": json.dumps(judgement["flags"], ensure_ascii=False),
                "dimensions_json": json.dumps(judgement["dimensions"]),
                "truncated": truncated,
                "response": response,
            }
        )

    return pd.DataFrame(rows, columns=list(JUDGEMENT_COLUMNS))
