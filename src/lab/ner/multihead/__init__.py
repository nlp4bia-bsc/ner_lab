"""One shared Transformer with an O/B/I head per entity type: training, search, inference."""

from __future__ import annotations

from lab.ner.multihead.crf import MultiHeadCRFForTokenClassification
from lab.ner.multihead.encoding import (
    MultiHeadEncoder,
    describe_multihead_encoder,
    entity_labels_from_partitions,
    multihead_encoder_from_description,
    resolve_entities_by_type,
    validate_entity_labels,
)
from lab.ner.multihead.loading import load_multihead_model, multihead_model_encoding
from lab.ner.multihead.model import MultiHeadForTokenClassification, head_balanced_loss
from lab.ner.multihead.predict import predict_multihead
from lab.ner.multihead.scoring import (
    build_multihead_compute_metrics,
    evaluate_multihead_predictions,
    head_rows,
    multihead_gold_spans,
    multihead_predicted_spans,
    multihead_token_diagnostics,
    score_heads,
)
from lab.ner.multihead.search import encode_multihead_variants, search_multihead
from lab.ner.multihead.training import train_multihead

__all__ = [
    "MultiHeadCRFForTokenClassification",
    "MultiHeadEncoder",
    "MultiHeadForTokenClassification",
    "build_multihead_compute_metrics",
    "describe_multihead_encoder",
    "encode_multihead_variants",
    "entity_labels_from_partitions",
    "evaluate_multihead_predictions",
    "head_balanced_loss",
    "head_rows",
    "load_multihead_model",
    "multihead_encoder_from_description",
    "multihead_gold_spans",
    "multihead_model_encoding",
    "multihead_predicted_spans",
    "multihead_token_diagnostics",
    "predict_multihead",
    "resolve_entities_by_type",
    "score_heads",
    "search_multihead",
    "train_multihead",
    "validate_entity_labels",
]
