"""NER error, boundary, label and lexical-generalization analysis."""

from __future__ import annotations

from lab.ner.analysis.api import (
    AnalysisResult,
    analyze_evaluation,
    inspect_analysis,
    regenerate_report,
)
from lab.ner.analysis.diagnostics import DiagnosticMatchConfig
from lab.ner.analysis.exposure import ExposureConfig, ExposureIndex, NormalizationConfig

__all__ = [
    "AnalysisResult",
    "DiagnosticMatchConfig",
    "ExposureConfig",
    "ExposureIndex",
    "NormalizationConfig",
    "analyze_evaluation",
    "inspect_analysis",
    "regenerate_report",
]
