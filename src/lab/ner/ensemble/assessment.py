"""Comparable ensemble baselines on one cached set of member predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from lab.ner.ensemble.entities import (
    EntityVotes,
    VotingParameters,
    decode_entities,
    score_ensemble,
    select_specialists,
)
from lab.ner.ensemble.spans import SPAN_METHODS, combine_spans
from lab.ner.ensemble.tables import MemberPredictions


def specialist_choices(
    predictions: Sequence[MemberPredictions],
    gold: pd.DataFrame,
    entity_labels: Sequence[str],
    min_gold_entities: int,
) -> tuple[dict[str, int], dict[str, str]]:
    """Select the first best expert per type, falling back deterministically with sparse gold."""
    selected, fallbacks = {}, {}
    for label in entity_labels:
        eligible = [i for i, member in enumerate(predictions) if label in member.entity_labels]
        reference = gold[gold["label"] == label]
        if (
            len(reference.drop_duplicates(["filename", "start_span", "end_span"]))
            < min_gold_entities
        ):
            selected[label] = eligible[0]
            fallbacks[label] = "insufficient_gold_first_eligible"
            continue
        scores = [
            score_ensemble(reference, member.spans[member.spans["label"] == label], [label])[
                "span_strict_f1"
            ]
            for member in (predictions[i] for i in eligible)
        ]
        selected[label] = eligible[max(range(len(scores)), key=scores.__getitem__)]

    return selected, fallbacks


def evaluate_baselines(
    predictions: Sequence[MemberPredictions],
    aligned: dict[str, EntityVotes],
    gold: pd.DataFrame,
    texts: dict[str, str],
    legacy: bool,
    missing_policy: str,
    min_gold_entities: int,
    min_overlap_percentage: float,
) -> dict[str, Any]:
    """Score individuals on the full task and separately on their supported specialist tasks."""
    labels = list(aligned)

    def score(spans: pd.DataFrame) -> dict:
        return score_ensemble(gold, spans, labels, legacy, min_overlap_percentage)

    member_metrics, specialist_metrics = [], []
    for member in predictions:
        spans = member.spans[member.spans["label"].isin(labels)]
        member_metrics.append(score(spans))
        supported = [label for label in labels if label in member.entity_labels]
        specialist_metrics.append(
            score_ensemble(
                gold[gold["label"].isin(supported)],
                spans,
                supported,
                legacy,
                min_overlap_percentage,
            )
        )
    selected, fallbacks = specialist_choices(predictions, gold, labels, min_gold_entities)
    best_spans = select_specialists(predictions, selected)
    parameters = VotingParameters([1 / len(predictions)] * len(predictions))
    baselines = {
        name: score(decode_entities(aligned, parameters, votes, texts, missing_policy))
        for name, votes in (("majority", "hard"), ("soft", "soft"))
    }
    for method in SPAN_METHODS:
        baselines[method] = score(combine_spans(predictions, labels, method, texts))
    baselines["best_per_entity"] = score(best_spans)

    return {
        "members": member_metrics,
        "specialists": specialist_metrics,
        "selected_members": selected,
        "fallbacks": fallbacks,
        "baselines": baselines,
    }
