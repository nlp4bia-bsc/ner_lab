"""Per-summary score tables and their per-language statistics, one independent column per metric."""

from __future__ import annotations

import json
import statistics
from typing import Any

import pandas as pd

from lab.summarization.evaluation.judges import CARE_DIMENSIONS, MALFORMED_FLAG

KEY_COLUMNS = ("doc_id", "language")


def merge_scores(classic_df: pd.DataFrame, judgements_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    One row per summary, one column per metric: the classic metrics, then each judge.

    Each judge's normalised score becomes a column named after it, and
    completeness also gets one `completeness_<dimension>` column per CARE
    dimension. No column combines metrics; each is reported on its own.
    """
    if judgements_df is None or judgements_df.empty:
        return classic_df.reset_index(drop=True)

    judge_scores = judgements_df.pivot(index=list(KEY_COLUMNS), columns="judge", values="score")
    judge_scores = judge_scores.loc[:, list(judgements_df["judge"].unique())].reset_index()
    judge_scores.columns.name = None
    scores_df = classic_df.merge(judge_scores, on=list(KEY_COLUMNS), how="left", validate="1:1")

    completeness = judgements_df.loc[judgements_df["judge"] == "completeness"]

    if completeness.empty:
        return scores_df

    rows = []

    for row in completeness.itertuples(index=False):
        values = json.loads(row.dimensions_json) or {}
        rows.append(
            {
                "doc_id": row.doc_id,
                "language": row.language,
                **{f"completeness_{name}": values.get(name) for name in CARE_DIMENSIONS},
            }
        )

    dimensions = pd.DataFrame(rows)

    return scores_df.merge(dimensions, on=list(KEY_COLUMNS), how="left", validate="1:1")


def describe_scores(scores_df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """
    Mean, standard deviation, minimum and maximum of every metric, per language.

    The deviation is the population one, since the evaluation set is fixed, not
    a sample of a larger one. A missing value, such as a CARE dimension of an
    unreadable completeness answer, is left out of its metric's statistics.
    """
    metrics = [column for column in scores_df.columns if column not in KEY_COLUMNS]

    return {
        str(language): {
            metric: describe_values(group[metric].dropna().astype(float).tolist())
            for metric in metrics
        }
        for language, group in scores_df.groupby("language", sort=True)
    }


def describe_values(values: list[float]) -> dict[str, float]:
    """Mean, population standard deviation, minimum and maximum, rounded to six decimals."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": round(statistics.mean(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def count_judgement_issues(judgements_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per language and judge, how many answers were unreadable and how many prompts were cut."""
    malformed = judgements_df["flags_json"].map(lambda flags: MALFORMED_FLAG in json.loads(flags))
    counts = judgements_df.assign(malformed=malformed).groupby(["language", "judge"], sort=True)

    issues: dict[str, dict[str, Any]] = {}

    for (language, judge), group in counts:
        issues.setdefault(str(language), {})[str(judge)] = {
            "n_judged": int(len(group)),
            "n_malformed": int(group["malformed"].sum()),
            "n_truncated": int(group["truncated"].sum()),
        }

    return issues
