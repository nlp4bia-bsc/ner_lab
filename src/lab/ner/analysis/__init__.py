"""NER error analysis: strict errors diagnosed, set against lexical exposure, with intervals and figures."""

from __future__ import annotations

from lab.ner.analysis.analyze import (
    ANALYSIS_MANIFEST_FILENAME,
    EVENTS_FILENAME,
    ErrorAnalysis,
    analyze_predictions,
    write_figures,
)
from lab.ner.analysis.exposure import (
    EXPOSURE_CLASSES,
    FREQUENCY_BINS,
    SIMILARITY_BINS,
    normalize_mention,
)
from lab.ner.analysis.figures import FIGURE_FORMATS
from lab.ner.analysis.pairing import BOUNDARY_KINDS, ERROR_TYPES, MATCHED_EDGES

__all__ = [
    "ANALYSIS_MANIFEST_FILENAME",
    "BOUNDARY_KINDS",
    "ERROR_TYPES",
    "EVENTS_FILENAME",
    "EXPOSURE_CLASSES",
    "FIGURE_FORMATS",
    "FREQUENCY_BINS",
    "MATCHED_EDGES",
    "SIMILARITY_BINS",
    "ErrorAnalysis",
    "analyze_predictions",
    "normalize_mention",
    "write_figures",
]
