"""The `ner.analyze_predictions` task: predictions, gold and training mentions to an error analysis."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from lab.core.corpus import DOCUMENT_COLUMNS
from lab.core.io import DEFAULT_PARQUET_COMPRESSION
from lab.core.provenance import file_sha256, write_manifest
from lab.core.spans import SPAN_COLUMNS, spans_from_corpus
from lab.core.stratification import DEFAULT_RANDOM_STATE
from lab.ner.analysis.confidence import (
    calibration,
    confidence_by_outcome,
    confidence_sweep,
    has_varying_scores,
)
from lab.ner.analysis.errors import (
    boundary_errors,
    boundary_summary,
    error_surfaces,
    error_taxonomy,
    oracle_error_budget,
    overlap_sensitivity,
)
from lab.ner.analysis.events import strict_events
from lab.ner.analysis.exposure import describe_exposure
from lab.ner.analysis.features import gold_structure, mention_features
from lab.ner.analysis.figures import FIGURE_FORMATS, draw_figures
from lab.ner.analysis.pairing import attach_units, pair_errors
from lab.ner.analysis.subgroups import covered_characters, gold_subgroups, predicted_subgroups
from lab.ner.analysis.variability import bootstrap_draws, bootstrap_intervals, document_scores
from lab.ner.inference.io import read_predictions, read_reference

FigureFormat = Literal["svg", "pdf", "png"]

ANALYSIS_MANIFEST_FILENAME = "analysis_manifest.json"
EVENTS_FILENAME = "events.parquet"
TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"
DEFAULT_SENSITIVITY_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80)


@dataclass(frozen=True)
class ErrorAnalysis:
    """What one error analysis produced: the metrics, the event table and every table, and where they landed."""

    output_dir: Path
    metrics: dict[str, Any]
    events: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    paths: dict[str, Any]

    def summary(self) -> str:
        """A one-line account: strict F1 with its interval, then how many errors of each type."""
        interval = self.tables["bootstrap_intervals"].set_index("metric").loc["span_strict_f1"]
        counts = self.tables["error_taxonomy"].set_index("error_type")["count"]

        return (
            f"strict F1 = {self.metrics['span_strict_f1']} [{interval['low']:.4f}, {interval['high']:.4f}]"
            f" | {counts['BOUNDARY_ERROR']} boundary, {counts['MISSED']} missed,"
            f" {counts['SPURIOUS']} spurious, {counts['UNCREDITED_EXACT']} uncredited exact"
        )


def analyze_predictions(
    predictions: pd.DataFrame | str | Path,
    reference: pd.DataFrame | str | Path,
    training: pd.DataFrame | str | Path,
    target_label: str,
    output_dir: str | Path,
    similarity_threshold: float = 0.80,
    overlap_threshold: float = 0.60,
    sensitivity_thresholds: Sequence[float] = DEFAULT_SENSITIVITY_THRESHOLDS,
    min_overlap_percentage: float = 40.0,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    random_state: int = DEFAULT_RANDOM_STATE,
    figures: bool = True,
    figure_format: FigureFormat = "svg",
) -> ErrorAnalysis:
    """
    Analyse predicted entities against gold: what the errors are, and where they fall.

    `predictions` is the span table `predict_entities` or `predict_ensemble` wrote, as
    parquet or TSV, or the frame itself. `reference` is the gold and `training` the
    mentions the model learned from, each a corpus (frame or parquet) or an annotation
    TSV. All three are restricted to `target_label`; a prediction of any other label
    raises.

    The strict scores are those `predict_entities` reports, from one nervaluate run.
    Each strict error is then put in a diagnostic unit: a boundary error, where a
    prediction and a gold span reach `overlap_threshold` in character Dice, or a
    fragmentation or merge of several; an exact prediction nervaluate did not credit;
    or a plain miss or spurious prediction. `sensitivity_thresholds` reruns this
    grouping to show how much it owes to the threshold.

    Every mention gets its lexical exposure to the training mentions — `SEEN`,
    `LEXICALLY_SIMILAR` from `similarity_threshold` in normalized Levenshtein
    similarity, or `NOVEL` — and its surface features. Gold subgroups report recall
    and how the rest failed; prediction subgroups report precision. Every rate has a
    `confidence` interval from `bootstrap_samples` document resamples seeded by
    `random_state`. Predictions carrying a varying `score` also get confidence tables.

    `output_dir` receives `events.parquet`, `tables/*.parquet`, the figures in
    `figures/` as `figure_format` when `figures` is set, and `analysis_manifest.json`.
    """
    _check_arguments(
        similarity_threshold,
        overlap_threshold,
        sensitivity_thresholds,
        bootstrap_samples,
        confidence,
        figure_format,
    )
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = read_mentions(reference, target_label, "reference")
    training_mentions = read_mentions(training, target_label, "training")
    predicted = read_predicted(predictions, target_label)

    events, metrics = strict_events(
        gold=gold,
        predicted=predicted,
        target_label=target_label,
        min_overlap_percentage=min_overlap_percentage,
    )
    units = pair_errors(events, overlap_threshold=overlap_threshold)
    events = attach_units(events, units)

    gold_mentions = describe_mentions(
        spans=gold,
        events=events,
        side="gold",
        other=predicted,
        training=training_mentions,
        similarity_threshold=similarity_threshold,
    )
    predicted_mentions = describe_mentions(
        spans=predicted,
        events=events,
        side="pred",
        other=gold,
        training=training_mentions,
        similarity_threshold=similarity_threshold,
    )

    tables = build_tables(
        gold=gold,
        predicted=predicted,
        events=events,
        units=units,
        gold_mentions=gold_mentions,
        predicted_mentions=predicted_mentions,
        target_label=target_label,
        overlap_threshold=overlap_threshold,
        sensitivity_thresholds=sensitivity_thresholds,
        min_overlap_percentage=min_overlap_percentage,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        random_state=random_state,
    )
    paths = write_tables(events, tables, output_dir)
    paths["figures"] = (
        draw_figures(tables, output_dir / FIGURES_DIRNAME, figure_format) if figures else []
    )

    manifest = {
        "target_label": target_label,
        "output_dir": str(output_dir),
        "inputs": {
            "predictions": _describe_source(predictions, predicted),
            "reference": _describe_source(reference, gold),
            "training": _describe_source(training, training_mentions),
        },
        "counts": {
            "gold": len(gold),
            "predicted": len(predicted),
            "training": len(training_mentions),
            "documents": len(tables["documents"]),
            "duplicate_gold": int(gold.duplicated(SPAN_COLUMNS).sum()),
            "duplicate_predicted": int(predicted.duplicated(SPAN_COLUMNS).sum()),
        },
        "parameters": {
            "similarity_threshold": similarity_threshold,
            "overlap_threshold": overlap_threshold,
            "sensitivity_thresholds": list(sensitivity_thresholds),
            "min_overlap_percentage": min_overlap_percentage,
            "bootstrap_samples": bootstrap_samples,
            "confidence": confidence,
            "random_state": random_state,
            "figure_format": figure_format if figures else None,
        },
        "metrics": metrics,
        "outputs": {
            "events": str(paths["events"]),
            "tables": {name: str(path) for name, path in paths["tables"].items()},
            "figures": [str(path) for path in paths["figures"]],
        },
        "duration_sec": round(time.perf_counter() - started, 3),
    }
    paths["analysis_manifest"] = write_manifest(manifest, output_dir / ANALYSIS_MANIFEST_FILENAME)

    return ErrorAnalysis(
        output_dir=output_dir,
        metrics=metrics,
        events=events,
        tables=tables,
        manifest=manifest,
        paths=paths,
    )


def write_figures(output_dir: str | Path, figure_format: FigureFormat = "svg") -> list[Path]:
    """
    Draw the figures again from the tables an analysis wrote to `output_dir`, without rescoring.

    The figures go to `output_dir/figures`, replacing any of the same name; the
    manifest is left as the analysis wrote it.
    """
    if figure_format not in FIGURE_FORMATS:
        raise ValueError(f"figure_format must be one of {FIGURE_FORMATS}, got {figure_format!r}.")

    tables_dir = Path(output_dir) / TABLES_DIRNAME

    if not tables_dir.is_dir():
        raise FileNotFoundError(
            f"No analysis tables in {tables_dir}; run analyze_predictions first."
        )

    tables = {path.stem: pd.read_parquet(path) for path in sorted(tables_dir.glob("*.parquet"))}

    return draw_figures(tables, Path(output_dir) / FIGURES_DIRNAME, figure_format)


def read_mentions(source: pd.DataFrame | str | Path, target_label: str, name: str) -> pd.DataFrame:
    """
    Read gold or training mentions of `target_label` as a span table, one row per mention.

    `source` is a corpus frame or parquet, whose entities are taken as annotated, a
    span table frame, or an annotation TSV. Rows are renumbered from 0, which is
    what the analysis's `gold_id`s refer to.
    """
    if isinstance(source, pd.DataFrame) and set(DOCUMENT_COLUMNS) <= set(source.columns):
        spans = spans_from_corpus(source, label=target_label)
    elif isinstance(source, pd.DataFrame):
        spans = source[source["label"].astype(str) == target_label]
    else:
        spans = read_reference(source, target_label)

    if spans.empty:
        raise ValueError(f"{name} holds no mention labelled target_label={target_label!r}.")

    return spans[SPAN_COLUMNS].reset_index(drop=True)


def read_predicted(predictions: pd.DataFrame | str | Path, target_label: str) -> pd.DataFrame:
    """
    Read the predictions to analyse, keeping `score` when they carry it.

    The analysis covers one label, so a prediction labelled anything but
    `target_label` raises rather than being dropped unseen.
    """
    spans = predictions if isinstance(predictions, pd.DataFrame) else read_predictions(predictions)
    other_labels = sorted(set(spans["label"].astype(str)) - {target_label})

    if other_labels:
        raise ValueError(
            f"predictions carry labels other than target_label={target_label!r}: {other_labels[:10]}. "
            "The analysis covers one label; pass the predictions of that label only."
        )

    columns = [*SPAN_COLUMNS, "score"] if "score" in spans.columns else SPAN_COLUMNS

    return spans[columns].reset_index(drop=True)


def describe_mentions(
    spans: pd.DataFrame,
    events: pd.DataFrame,
    side: Literal["gold", "pred"],
    other: pd.DataFrame,
    training: pd.DataFrame,
    similarity_threshold: float,
) -> pd.DataFrame:
    """
    One row per gold span or prediction: the span, its outcome, exposure and features.

    `outcome` is the error type of the unit holding the span. `characters` and
    `covered_characters` give its length and how much of it `other` covers, the
    character-level counterpart of the outcome. Gold spans also say whether they
    overlap or nest with another gold span.
    """
    holding = events[events[f"{side}_id"].notna()]
    outcome = pd.Series(
        holding["error_type"].to_numpy(), index=holding[f"{side}_id"].astype(int).to_numpy()
    )

    mentions = spans.copy()
    mentions["outcome"] = outcome.reindex(mentions.index).to_numpy()
    mentions["characters"] = (mentions["end_span"] - mentions["start_span"]).astype(int)
    mentions["covered_characters"] = covered_characters(mentions, other)

    parts = [
        mentions,
        describe_exposure(
            texts=spans["text"],
            training_texts=training["text"],
            similarity_threshold=similarity_threshold,
        ),
        mention_features(spans["text"]),
    ]

    if side == "gold":
        parts.append(gold_structure(spans))

    return pd.concat(parts, axis=1)


def build_tables(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    events: pd.DataFrame,
    units: pd.DataFrame,
    gold_mentions: pd.DataFrame,
    predicted_mentions: pd.DataFrame,
    target_label: str,
    overlap_threshold: float,
    sensitivity_thresholds: Sequence[float],
    min_overlap_percentage: float,
    bootstrap_samples: int,
    confidence: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """
    Every table the analysis writes, by name.

    The confidence tables are there only when the predictions carry a `score` that
    varies; every other table always is, empty when there is nothing to put in it.
    """
    documents = document_scores(events=events, gold=gold, predicted=predicted)
    draws = bootstrap_draws(
        n_documents=len(documents), samples=bootstrap_samples, random_state=random_state
    )
    resampling = {
        "documents": pd.Index(documents["filename"]),
        "draws": draws,
        "confidence": confidence,
    }
    boundary = boundary_errors(
        units=units, gold=gold, predicted=predicted, gold_exposure=gold_mentions["exposure"]
    )

    tables = {
        "gold_mentions": gold_mentions,
        "predicted_mentions": predicted_mentions,
        "units": units,
        "gold_subgroups": gold_subgroups(gold_mentions, **resampling),
        "predicted_subgroups": predicted_subgroups(predicted_mentions, **resampling),
        "error_taxonomy": error_taxonomy(units),
        "boundary_errors": boundary,
        "boundary_summary": boundary_summary(boundary),
        "overlap_sensitivity": overlap_sensitivity(
            events, thresholds=sensitivity_thresholds, chosen=overlap_threshold
        ),
        "oracle_error_budget": oracle_error_budget(units),
        "error_surfaces": error_surfaces(units=units, gold=gold, predicted=predicted),
        "documents": documents,
        "bootstrap_intervals": bootstrap_intervals(documents, draws=draws, confidence=confidence),
    }

    if has_varying_scores(predicted):
        tables["confidence_by_outcome"] = confidence_by_outcome(predicted_mentions)
        tables["calibration"] = calibration(predicted_mentions)
        tables["confidence_sweep"] = confidence_sweep(
            gold=gold,
            predicted=predicted,
            target_label=target_label,
            min_overlap_percentage=min_overlap_percentage,
        )

    return tables


def write_tables(
    events: pd.DataFrame, tables: dict[str, pd.DataFrame], output_dir: Path
) -> dict[str, Any]:
    """Write the event table and every table as parquet; returns their paths."""
    tables_dir = output_dir / TABLES_DIRNAME
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {"events": output_dir / EVENTS_FILENAME, "tables": {}}

    events.to_parquet(paths["events"], compression=DEFAULT_PARQUET_COMPRESSION, index=False)

    for name, table in tables.items():
        paths["tables"][name] = tables_dir / f"{name}.parquet"
        table.to_parquet(
            paths["tables"][name], compression=DEFAULT_PARQUET_COMPRESSION, index=False
        )

    return paths


def _check_arguments(
    similarity_threshold: float,
    overlap_threshold: float,
    sensitivity_thresholds: Sequence[float],
    bootstrap_samples: int,
    confidence: float,
    figure_format: str,
) -> None:
    if not 0.0 < similarity_threshold < 1.0:
        raise ValueError(
            f"similarity_threshold must lie strictly between 0 and 1, got {similarity_threshold}; "
            "1.0 would leave no mention LEXICALLY_SIMILAR."
        )

    for name, threshold in [
        ("overlap_threshold", overlap_threshold),
        *(("sensitivity_thresholds", t) for t in sensitivity_thresholds),
    ]:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"{name} values are character Dice in (0, 1], got {threshold}.")

    if bootstrap_samples < 1:
        raise ValueError(f"bootstrap_samples must be at least 1, got {bootstrap_samples}.")

    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between 0 and 1, got {confidence}.")

    if figure_format not in FIGURE_FORMATS:
        raise ValueError(f"figure_format must be one of {FIGURE_FORMATS}, got {figure_format!r}.")


def _describe_source(source: Any, frame: pd.DataFrame) -> dict[str, Any]:
    path = Path(source).resolve() if isinstance(source, (str, Path)) else None

    return {
        "path": None if path is None else str(path),
        "sha256": file_sha256(path) if path is not None and path.is_file() else None,
        "rows": len(frame),
    }
