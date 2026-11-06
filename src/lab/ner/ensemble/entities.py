"""Eligibility-aware collections of the existing single-entity word votes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from lab.core.segmentation import split_into_words
from lab.core.spans import span_dataframe
from lab.ner.ensemble.tables import MemberPredictions
from lab.ner.ensemble.voting import WordVotes, align_votes, ensemble_spans
from lab.ner.evaluation.multihead import strict_multihead_metrics
from lab.ner.evaluation.scoring import span_metrics

Scope = Literal["global", "entity"]
MissingPolicy = Literal["skip", "break"]


@dataclass(frozen=True)
class EntityVotes:
    """One label's votes, with local columns mapped to eligible member indices."""

    votes: WordVotes
    member_indices: tuple[int, ...]


@dataclass(frozen=True)
class VotingParameters:
    """Global values and optional per-entity overrides in eligible member order."""

    weights: list[float]
    threshold: float = 0.5
    entity_weights: dict[str, list[float]] = field(default_factory=dict)
    entity_thresholds: dict[str, float] = field(default_factory=dict)


def align_entities(
    predictions: Sequence[MemberPredictions],
    entity_labels: Sequence[str],
) -> dict[str, EntityVotes]:
    """Allocate word/member arrays only for members supporting each entity."""
    aligned = {}
    for label in entity_labels:
        indices = tuple(i for i, member in enumerate(predictions) if label in member.entity_labels)
        if not indices:
            raise ValueError(
                f"No member supports entity {label!r}; select metadata-supported labels."
            )
        tables = [predictions[index].words.query("entity_label == @label") for index in indices]
        aligned[label] = EntityVotes(align_votes(tables), indices)

    return aligned


def decode_entities(
    aligned: dict[str, EntityVotes],
    parameters: VotingParameters,
    votes: str,
    texts: dict[str, str],
    missing_policy: MissingPolicy = "break",
    word_spans: dict[str, list[dict]] | None = None,
) -> pd.DataFrame:
    """Decode types independently, normalizing only eligible and present evidence."""
    boundaries = (
        word_spans
        if word_spans is not None
        else {doc_id: split_into_words(text) for doc_id, text in texts.items()}
    )
    frames = []
    for label, entity in aligned.items():
        weights = parameters.entity_weights.get(label)
        if weights is None:
            weights = [parameters.weights[index] for index in entity.member_indices]
        threshold = parameters.entity_thresholds.get(label, parameters.threshold)
        frames.append(
            ensemble_spans(
                entity.votes,
                weights=weights,
                threshold=threshold,
                votes=votes,
                texts=texts,
                target_label=label,
                missing_policy=missing_policy,
                word_spans=boundaries,
            )
        )

    return span_dataframe(
        [row for frame in frames for row in frame.to_dict("records")], scored=True
    )


def score_ensemble(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    entity_labels: Sequence[str],
    legacy: bool = False,
    min_overlap_percentage: float = 40.0,
) -> dict[str, float | int]:
    """Keep legacy SemEval semantics or reuse Multi-Head exact set scoring."""
    metrics = strict_multihead_metrics(gold, predicted, list(entity_labels))
    if legacy:
        metrics.update(
            span_metrics(
                gold,
                predicted,
                tags=list(entity_labels),
                min_overlap_percentage=min_overlap_percentage,
            )
        )

    return metrics


def select_specialists(
    predictions: Sequence[MemberPredictions],
    selected: dict[str, int],
) -> pd.DataFrame:
    """Collect each selected expert's original spans for its assigned label."""
    records = []
    for label, member in selected.items():
        spans = predictions[member].spans
        records.extend(spans[spans["label"] == label].to_dict("records"))

    return span_dataframe(records, scored=True)
