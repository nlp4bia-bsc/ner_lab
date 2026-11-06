"""Apply fitted word voting, specialist selection or exact span consensus."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from lab.ner.ensemble.entities import (
    EntityVotes,
    VotingParameters,
    decode_entities,
    select_specialists,
)
from lab.ner.ensemble.spans import SPAN_METHODS, combine_spans
from lab.ner.ensemble.tables import MemberPredictions


def combine_predictions(
    predictions: Sequence[MemberPredictions],
    aligned: dict[str, EntityVotes],
    method: str,
    parameters: VotingParameters,
    votes: str,
    texts: dict[str, str],
    selected_members: dict[str, int] | None = None,
    min_support: int | None = None,
    missing_policy: str = "break",
    entity_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Use cached evidence without loading a checkpoint or changing fitted parameters."""
    if method in SPAN_METHODS:
        labels = list(aligned) if entity_labels is None else entity_labels
        return combine_spans(predictions, labels, method, texts, min_support)
    if method == "best":
        return select_specialists(predictions, selected_members or {})

    return decode_entities(aligned, parameters, votes, texts, missing_policy)
