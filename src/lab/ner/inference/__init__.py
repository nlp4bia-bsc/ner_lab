"""Running a saved model over documents: loading it, predicting, decoding, scoring."""

from __future__ import annotations

from lab.ner.inference.io import (
    has_gold,
    read_documents,
    read_reference,
    write_gold,
    write_predictions,
)
from lab.ner.inference.loading import load_model, load_state_dict, resolve_device
from lab.ner.inference.predict import (
    GOLD_FILENAME,
    INFERENCE_MANIFEST_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    InferenceResult,
    predict_entities,
)
from lab.ner.inference.prediction import decode_spans, predict_logits

__all__ = [
    "GOLD_FILENAME",
    "INFERENCE_MANIFEST_FILENAME",
    "METRICS_FILENAME",
    "PREDICTIONS_FILENAME",
    "InferenceResult",
    "decode_spans",
    "has_gold",
    "load_model",
    "load_state_dict",
    "predict_entities",
    "predict_logits",
    "read_documents",
    "read_reference",
    "resolve_device",
    "write_gold",
    "write_predictions",
]
