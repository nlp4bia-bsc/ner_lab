"""What each LLM judge asks about a summary, and how its answer is read back into a score."""

from __future__ import annotations

import json
import re
from functools import cache
from importlib import resources
from typing import Any, Literal

Judge = Literal["faithfulness", "completeness", "fluency"]

JUDGES = ("faithfulness", "completeness", "fluency")
CARE_DIMENSIONS = (
    "patient_demographics",
    "clinical_presentation",
    "diagnosis",
    "intervention",
    "outcome",
    "follow_up",
)
SCORE_SCALE = 10.0
PROMPTS_DIR = "prompts"
MALFORMED_FLAG = "llm_output_malformed"
JSON_MALFORMED_FLAG = "json_malformed"
SCORE_PATTERN = re.compile(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)')


def build_prompt(judge: Judge, text: str, generated_summary: str, language: str) -> str:
    """
    The prompt `judge` asks about one generated summary.

    Faithfulness and completeness read the summary against its source `text`;
    fluency reads the summary alone. `language` is written into the prompt as
    given, so it is whatever code the summaries table carries.
    """
    if judge not in JUDGES:
        raise ValueError(f"Unknown judge {judge!r}. Available judges: {JUDGES}.")

    return (
        read_prompt_template(judge)
        .replace("{source_text}", text)
        .replace("{generated_summary}", generated_summary)
        .replace("{language}", language)
    )


@cache
def read_prompt_template(judge: Judge) -> str:
    """The prompt template of one judge, as shipped in `lab.summarization.evaluation`."""
    template = resources.files("lab.summarization.evaluation") / PROMPTS_DIR / f"{judge}.txt"

    return template.read_text(encoding="utf-8")


def parse_judgement(judge: Judge, response: str) -> dict[str, Any]:
    """
    Read a judge's answer into `raw_score`, `score`, `confidence`, `reasoning`, `flags`
    and `dimensions`.

    The answer is the first complete JSON object in `response`; prose before or
    after it is ignored. Faithfulness and fluency answer one `score` in [0, 10],
    and when no JSON object parses but a `"score": n` can still be found, that
    score is kept with the flag `json_malformed`. Completeness answers the six CARE
    dimensions in [0, 10]; its `raw_score` is their mean and `dimensions` holds
    each one normalised. `score` is `raw_score / 10`, in [0, 1].

    A confidence above 1 is read as given on a 0–10 scale and divided by 10;
    completeness is not asked for one, so it is 0.

    An answer that cannot be read — no score, no reasoning, a score outside
    [0, 10] — never raises: it scores 0 with the flag `llm_output_malformed`, so
    one bad generation does not stop a run, and the flag is what tells it apart
    from a real 0.
    """
    try:
        if judge == "completeness":
            return _parse_care(response)

        return _parse_score(response)
    except (ValueError, TypeError):
        return _judgement(0.0, 0.0, "parsing_failed", [MALFORMED_FLAG], 0.0)


def _parse_score(response: str) -> dict[str, Any]:
    data = _first_json_object(response)

    if data is None:
        match = SCORE_PATTERN.search(response)

        if match is None:
            raise ValueError("No JSON object and no score in the response.")

        raw_score = float(match.group(1))

        return _judgement(
            raw_score,
            _normalise(raw_score),
            "json_malformed_score_extracted",
            [JSON_MALFORMED_FLAG],
            0.0,
        )

    if "score" not in data or "reasoning" not in data:
        raise ValueError("The response's JSON has no score or no reasoning.")

    raw_score = float(data["score"])

    return _judgement(
        raw_score,
        _normalise(raw_score),
        str(data["reasoning"]),
        list(data.get("flags", [])),
        _normalise_confidence(data.get("confidence", 0.0)),
    )


def _parse_care(response: str) -> dict[str, Any]:
    data = _first_json_object(response)

    if data is None:
        raise ValueError("No JSON object in the response.")

    if any(dimension not in data for dimension in CARE_DIMENSIONS) or "reasoning" not in data:
        raise ValueError("The response's JSON lacks a CARE dimension or the reasoning.")

    raw_dimensions = {dimension: float(data[dimension]) for dimension in CARE_DIMENSIONS}
    overall = sum(raw_dimensions.values()) / len(CARE_DIMENSIONS)

    return _judgement(
        round(overall, 6),
        _normalise(overall),
        str(data["reasoning"]),
        list(data.get("flags", [])),
        _normalise_confidence(data.get("confidence", 0.0)),
        {dimension: _normalise(value) for dimension, value in raw_dimensions.items()},
    )


def _first_json_object(response: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    index = response.find("{")

    while index != -1:
        try:
            value, _ = decoder.raw_decode(response, index)
        except json.JSONDecodeError:
            value = None

        if isinstance(value, dict):
            return value

        index = response.find("{", index + 1)

    return None


def _judgement(
    raw_score: float,
    score: float,
    reasoning: str,
    flags: list[Any],
    confidence: float,
    dimensions: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not 0.0 <= raw_score <= SCORE_SCALE:
        raise ValueError(f"Score {raw_score} is outside [0, {SCORE_SCALE}].")

    if not all(isinstance(flag, str) for flag in flags):
        raise ValueError("Flags must be strings.")

    return {
        "raw_score": raw_score,
        "score": score,
        "confidence": confidence,
        "reasoning": reasoning,
        "flags": flags,
        "dimensions": dimensions,
    }


def _normalise(raw_score: float) -> float:
    return max(0.0, min(1.0, round(raw_score / SCORE_SCALE, 6)))


def _normalise_confidence(value: Any) -> float:
    confidence = float(value) if value is not None else 0.0

    if confidence > 1.0:
        confidence = confidence / SCORE_SCALE

    return max(0.0, min(1.0, round(confidence, 6)))
