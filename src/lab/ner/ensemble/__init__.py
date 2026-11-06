"""Ensembling saved NER models: word-level voting across backbones, fitting and predicting."""

from __future__ import annotations

from lab.ner.ensemble.entities import (
    EntityVotes,
    VotingParameters,
    align_entities,
    decode_entities,
)
from lab.ner.ensemble.fit import (
    ENSEMBLE_MANIFEST_FILENAME,
    ENSEMBLE_METRICS_FILENAME,
    ENSEMBLE_TRIALS_FILENAME,
    METHODS,
    EnsembleFit,
    fit_ensemble,
)
from lab.ner.ensemble.io import (
    ENSEMBLE_FILENAME,
    check_member_weights,
    describe_member,
    read_ensemble,
    weights_sha256,
    write_ensemble,
)
from lab.ner.ensemble.predict import EnsembleInference, predict_ensemble
from lab.ner.ensemble.search import (
    EntitySearchResult,
    SearchResult,
    search_ensemble,
    search_entity_parameters,
)
from lab.ner.ensemble.spans import combine_spans
from lab.ner.ensemble.tables import (
    WORD_TABLE_COLUMNS,
    MarginalCRFTrainer,
    MemberPredictions,
    predict_member,
    predict_members,
    predict_word_table,
    predict_word_tables,
    word_table,
)
from lab.ner.ensemble.voting import (
    VOTES,
    CombinedWords,
    WordVotes,
    align_votes,
    combine_votes,
    decode_combined,
    ensemble_spans,
    normalize_weights,
)

__all__ = [
    "ENSEMBLE_FILENAME",
    "ENSEMBLE_MANIFEST_FILENAME",
    "ENSEMBLE_METRICS_FILENAME",
    "ENSEMBLE_TRIALS_FILENAME",
    "METHODS",
    "VOTES",
    "WORD_TABLE_COLUMNS",
    "CombinedWords",
    "EntityVotes",
    "VotingParameters",
    "MemberPredictions",
    "EntitySearchResult",
    "align_entities",
    "decode_entities",
    "combine_spans",
    "predict_member",
    "predict_members",
    "search_entity_parameters",
    "EnsembleFit",
    "EnsembleInference",
    "MarginalCRFTrainer",
    "SearchResult",
    "WordVotes",
    "align_votes",
    "check_member_weights",
    "combine_votes",
    "decode_combined",
    "describe_member",
    "ensemble_spans",
    "fit_ensemble",
    "normalize_weights",
    "predict_ensemble",
    "predict_word_table",
    "predict_word_tables",
    "read_ensemble",
    "search_ensemble",
    "weights_sha256",
    "word_table",
    "write_ensemble",
]
