"""Public orchestration API for NER error and generalization analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core import (
    DOCUMENT_COLUMNS,
    file_sha256,
    read_corpus,
    read_spans,
    spans_from_corpus,
    validate_corpus,
    validate_spans,
    write_manifest,
)
from lab.ner.analysis.diagnostics import (
    DiagnosticMatchConfig,
    audit_annotations,
    enrich_error_diagnostics,
    entity_features,
    evaluate_to_events,
)
from lab.ner.analysis.exposure import ExposureConfig, ExposureIndex
from lab.ner.analysis.plotting import write_analysis_figures, write_figure_captions
from lab.ner.analysis.reporting import build_analysis_tables


@dataclass(frozen=True)
class AnalysisResult:
    """Paths and top-level metadata produced by one NER analysis run."""

    output_dir: Path
    evaluation_path: Path
    manifest_path: Path
    table_paths: dict[str, Path]
    figure_paths: tuple[Path, ...]
    metrics: dict[str, float | int]
    multiclass: bool


def analyze_evaluation(
    predictions: pd.DataFrame | str | Path,
    gold: pd.DataFrame | str | Path,
    training: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    run_id: str = "analysis",
    levenshtein_threshold: float = 0.80,
    lexical_similarity_mode: str = "hybrid",
    min_overlap_percentage: float = 40.0,
    diagnostic_overlap_threshold: float = 0.60,
    diagnostic_sensitivity_thresholds: tuple[float, ...] = (0.40, 0.50, 0.60, 0.70, 0.80),
    bootstrap_samples: int = 2000,
    bootstrap_confidence: float = 0.95,
    partition_size: int = 50,
    random_state: int = 13,
    verbose: int = 0,
    evaluation_documents: pd.DataFrame | str | Path | None = None,
    training_documents: pd.DataFrame | str | Path | None = None,
) -> AnalysisResult:
    """Analyze NER predictions beyond aggregate STRICT/character metrics.

    Parameters accept the repository's canonical span tables in TSV or Parquet.
    ``training`` may additionally be a canonical ``documents.parquet`` corpus;
    its entities are converted through :func:`lab.core.spans_from_corpus`.

    ``SEEN`` / ``LEXICALLY_SIMILAR`` / ``NOVEL`` are lexical exposure classes.
    Exact raw mention surfaces are ``SEEN``. Otherwise lexical similarity is
    compared with ``levenshtein_threshold``. In the default ``hybrid`` mode the
    similarity is the conservative minimum of normalized Levenshtein and
    Jaro-Winkler similarity. This operational definition is persisted in the
    manifest.
    """
    if int(verbose) not in (0, 1):
        raise ValueError("verbose must be 0 or 1.")
    if int(bootstrap_samples) < 0:
        raise ValueError("bootstrap_samples must be >= 0.")
    if not 0.0 < float(diagnostic_overlap_threshold) <= 1.0:
        raise ValueError("diagnostic_overlap_threshold must be in (0, 1].")
    if int(partition_size) < 0:
        raise ValueError("partition_size must be >= 0; 0 disables partition stability analysis.")

    output_root = Path(output_dir)
    tables_root = output_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    gold_frame = read_spans(gold, scored=False)
    predicted_frame = read_spans(predictions, scored=None)
    training_frame, inferred_training_documents = _load_training(training)

    tags = sorted(
        set(gold_frame["label"].astype(str))
        | set(predicted_frame["label"].astype(str))
    )
    multiclass = len(tags) > 1

    exposure_config = ExposureConfig(
        levenshtein_threshold=float(levenshtein_threshold),
        similarity_mode=str(lexical_similarity_mode),
    )
    exposure = ExposureIndex(training_frame, exposure_config)

    events, metrics = evaluate_to_events(
        gold_frame,
        predicted_frame,
        run_id=run_id,
        tags=tags,
        min_overlap_percentage=float(min_overlap_percentage),
    )
    events = enrich_error_diagnostics(
        events,
        gold_frame,
        predicted_frame,
        match_config=DiagnosticMatchConfig(
            overlap_threshold=float(diagnostic_overlap_threshold)
        ),
        multiclass=multiclass,
    )
    events = _attach_exposure(events, gold_frame, predicted_frame, exposure)
    events = _attach_entity_features(events)

    evaluation_document_map = _document_map(evaluation_documents)
    training_document_map = _document_map(training_documents)
    if not training_document_map and inferred_training_documents:
        training_document_map = inferred_training_documents

    gold_audit = audit_annotations(gold_frame, evaluation_document_map)
    pred_audit = audit_annotations(predicted_frame, evaluation_document_map)
    training_audit = audit_annotations(training_frame, training_document_map)
    events = _attach_audit(events, gold_audit, pred_audit)

    tables = build_analysis_tables(
        gold_frame,
        predicted_frame,
        training_frame,
        events,
        metrics,
        exposure,
        tags=tags,
        min_overlap_percentage=float(min_overlap_percentage),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_confidence=float(bootstrap_confidence),
        random_state=int(random_state),
        diagnostic_overlap_threshold=float(diagnostic_overlap_threshold),
        diagnostic_sensitivity_thresholds=tuple(
            float(value) for value in diagnostic_sensitivity_thresholds
        ),
        partition_size=int(partition_size),
    )

    # Annotation-integrity tables are persisted independently from model errors.
    tables["gold_annotation_audit"] = _audit_table(gold_frame, gold_audit)
    tables["prediction_annotation_audit"] = _audit_table(predicted_frame, pred_audit)
    tables["training_annotation_audit"] = _audit_table(training_frame, training_audit)

    evaluation_path = output_root / "evaluation.parquet"
    events.to_parquet(
        evaluation_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    table_paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = tables_root / f"{name}.parquet"
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        table_paths[name] = path

    figure_paths: tuple[Path, ...] = ()
    if int(verbose) == 1:
        figure_paths = tuple(
            write_analysis_figures(
                tables,
                output_root,
                multiclass=multiclass,
            )
        )
        write_figure_captions(output_root, figure_paths)

    manifest = {
        "artifact": "ner_error_analysis",
        "schema_version": "2.1.0",
        "run_id": str(run_id),
        "official_evaluator": "lab.ner.evaluation.span_metrics",
        "official_metrics": metrics,
        "labels": tags,
        "multiclass": multiclass,
        "inputs": {
            "predictions": _source_metadata(predictions),
            "gold": _source_metadata(gold),
            "training": _source_metadata(training),
        },
        "counts": {
            "gold_entities": int(len(gold_frame)),
            "predicted_entities": int(len(predicted_frame)),
            "training_entities": int(len(training_frame)),
            "gold_duplicates": int(gold_frame.duplicated().sum()),
            "prediction_duplicates": int(predicted_frame.duplicated().sum()),
            "training_duplicates": int(training_frame.duplicated().sum()),
        },
        "generalization": {
            "classes": ["SEEN", "LEXICALLY_SIMILAR", "NOVEL"],
            "legacy_class_mapping": {
                "SEEN": "SEEN",
                "LEXICALLY_SIMILAR": "FEW_SHOT",
                "NOVEL": "ZERO_SHOT",
            },
            "seen_definition": "raw mention surface occurs exactly in training",
            "lexically_similar_definition": (
                "not exact-seen and configured lexical similarity >= "
                f"{float(levenshtein_threshold):.6g}"
            ),
            "novel_definition": (
                "not exact-seen and configured lexical similarity < "
                f"{float(levenshtein_threshold):.6g}, or no training neighbour"
            ),
            "lexical_similarity_threshold": float(levenshtein_threshold),
            "levenshtein_threshold_legacy_name": float(levenshtein_threshold),
            "similarity_mode": str(lexical_similarity_mode),
            "similarity_backend": exposure.backend,
            "hybrid_rule": (
                "minimum of normalized Levenshtein and Jaro-Winkler"
                if str(lexical_similarity_mode) == "hybrid"
                else "normalized Levenshtein"
            ),
            "normalization": exposure_config.normalization.__dict__,
            "unique_training_mentions": exposure.unique_training_mentions,
            "cache_hits": exposure.cache_hits,
            "references": [
                {
                    "method": "Jaro similarity",
                    "citation": "Jaro MA. J Am Stat Assoc. 1989;84(406):414-420.",
                    "doi": "10.1080/01621459.1989.10478785",
                }
            ],
        },
        "diagnostics": {
            "strict_pairing": (
                "diagnostic FP/FN association never changes official STRICT credit"
            ),
            "error_types": [
                "CORRECT",
                "MISSED",
                "SPURIOUS",
                "LABEL_ERROR",
                "BOUNDARY_ERROR",
                "BOUNDARY_AND_LABEL_ERROR",
            ],
            "label_error_diagnostics_enabled": bool(multiclass),
            "diagnostic_overlap_metric": "character-level Sorensen-Dice",
            "diagnostic_overlap_threshold": float(diagnostic_overlap_threshold),
            "diagnostic_sensitivity_thresholds": [
                float(value) for value in diagnostic_sensitivity_thresholds
            ],
            "diagnostic_overlap_threshold_is_repository_default": True,
            "fragmentation_policy": (
                "one gold mention split across multiple unmatched predictions is counted "
                "as one boundary-error diagnostic unit when the fragment union passes "
                "the same normalized Dice threshold and no fragment is already an eligible "
                "one-to-one pair"
            ),
            "merging_analysis": "disabled by design",
            "official_min_overlap_percentage": float(min_overlap_percentage),
            "references": [
                {
                    "method": "Sorensen-Dice overlap coefficient",
                    "citation": "Dice LR. Ecology. 1945;26(3):297-302.",
                    "doi": "10.2307/1932409",
                    "note": (
                        "The coefficient motivates the normalized symmetric overlap; "
                        "the 0.60 default is a repository diagnostic heuristic, not a "
                        "threshold prescribed by the original paper."
                    ),
                }
            ],
        },
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "confidence": float(bootstrap_confidence),
            "seed": int(random_state),
            "unit": "document",
        },
        "document_partition_stability": {
            "enabled": int(partition_size) > 0,
            "partition_size": int(partition_size),
            "sampling": "without replacement",
            "seed": int(random_state),
            "interpretation": "descriptive stability analysis; not a bootstrap confidence interval",
        },
        "visualization": {
            "profile": "publication_svg_v2",
            "vector_format": "SVG",
            "color_palette": "Okabe-Ito",
            "category_tick_rotation_degrees": 0,
            "top_spine": False,
            "right_spine": False,
            "default_grid": False,
            "support_annotations": True,
            "bootstrap_ci_on_generalization_f1": False,
            "uninformative_plots_are_omitted": True,
        },
        "verbose": int(verbose),
        "outputs": {
            "evaluation": str(evaluation_path),
            "tables": {name: str(path) for name, path in table_paths.items()},
            "figures": [str(path) for path in figure_paths],
            "figure_captions": (
                str(output_root / "captions.md") if int(verbose) == 1 else None
            ),
            "plot_manifest": (
                str(output_root / "figures" / "plot_manifest.json")
                if int(verbose) == 1
                else None
            ),
        },
    }
    manifest_path = output_root / "manifest.json"
    write_manifest(manifest, manifest_path)

    return AnalysisResult(
        output_dir=output_root,
        evaluation_path=evaluation_path,
        manifest_path=manifest_path,
        table_paths=table_paths,
        figure_paths=figure_paths,
        metrics=metrics,
        multiclass=multiclass,
    )


def inspect_analysis(path: str | Path) -> dict[str, Any]:
    """Load compact summary information from an analysis directory/manifest."""
    root = Path(path)
    manifest_path = root / "manifest.json" if root.is_dir() else root
    if manifest_path.name == "evaluation.parquet":
        manifest_path = manifest_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Analysis manifest does not exist: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = data.get("outputs", {}).get("tables", {})
    summary = {
        "run_id": data.get("run_id"),
        "multiclass": data.get("multiclass"),
        "labels": data.get("labels", []),
        "counts": data.get("counts", {}),
        "official_metrics": data.get("official_metrics", {}),
        "generalization": data.get("generalization", {}),
        "diagnostics": data.get("diagnostics", {}),
        "bootstrap": data.get("bootstrap", {}),
        "document_partition_stability": data.get("document_partition_stability", {}),
        "tables": tables,
        "figures": data.get("outputs", {}).get("figures", []),
    }
    generalization_path = tables.get("metrics_by_generalization")
    if generalization_path:
        candidate = Path(generalization_path)
        if not candidate.exists():
            candidate = manifest_path.parent / "tables" / "metrics_by_generalization.parquet"
        if candidate.exists():
            summary["metrics_by_generalization"] = pd.read_parquet(
                candidate
            ).to_dict(orient="records")
    return summary


def regenerate_report(
    analysis_dir: str | Path,
    *,
    verbose: int = 1,
) -> tuple[Path, ...]:
    """Regenerate SVG figures from persisted Parquet tables without rescoring."""
    if int(verbose) not in (0, 1):
        raise ValueError("verbose must be 0 or 1.")
    root = Path(analysis_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Analysis manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_paths = manifest.get("outputs", {}).get("tables", {})
    tables: dict[str, pd.DataFrame] = {}
    for name, path in table_paths.items():
        candidate = Path(path)
        if not candidate.exists():
            candidate = root / "tables" / f"{name}.parquet"
        if candidate.exists():
            tables[name] = pd.read_parquet(candidate)
    if int(verbose) == 0:
        return ()
    figures = tuple(
        write_analysis_figures(
            tables,
            root,
            multiclass=bool(manifest.get("multiclass", False)),
        )
    )
    write_figure_captions(root, figures)
    manifest["outputs"]["figures"] = [str(path) for path in figures]
    manifest["outputs"]["figure_captions"] = str(root / "captions.md")
    manifest["outputs"]["plot_manifest"] = str(root / "figures" / "plot_manifest.json")
    manifest.setdefault("visualization", {}).update(
        {
            "profile": "publication_svg_v2",
            "category_tick_rotation_degrees": 0,
            "uninformative_plots_are_omitted": True,
        }
    )
    write_manifest(manifest, manifest_path)
    return figures


def _load_training(
    training: pd.DataFrame | str | Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if isinstance(training, pd.DataFrame):
        if set(DOCUMENT_COLUMNS).issubset(training.columns):
            corpus = validate_corpus(training)
            return spans_from_corpus(corpus), {
                str(row.doc_id): str(row.text) for row in corpus.itertuples(index=False)
            }
        return validate_spans(training, scored=False), {}

    path = Path(training)
    if not path.exists():
        raise FileNotFoundError(f"Training input does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        if set(DOCUMENT_COLUMNS).issubset(frame.columns):
            corpus = read_corpus(path)
            return spans_from_corpus(corpus), {
                str(row.doc_id): str(row.text) for row in corpus.itertuples(index=False)
            }
    return read_spans(path, scored=False), {}


def _document_map(value) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, pd.DataFrame):
        frame = value
    else:
        path = Path(value)
        if path.suffix.lower() != ".parquet":
            raise ValueError("Document text input must be a canonical Parquet or DataFrame.")
        frame = pd.read_parquet(path)
    if not {"doc_id", "text"}.issubset(frame.columns):
        raise ValueError("Document text input requires doc_id and text columns.")
    return {str(row.doc_id): str(row.text) for row in frame.itertuples(index=False)}


def _attach_exposure(events, gold, predicted, exposure):
    enriched = events.copy()
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for side, frame in (("gold", gold), ("pred", predicted)):
        for index, row in frame.iterrows():
            cache[(side, int(index))] = exposure.describe(str(row.text), str(row.label))

    keys = list(exposure.describe("", "").keys())
    for prefix in ("gold_", "pred_"):
        for key in keys:
            enriched[f"{prefix}{key}"] = None
    enriched["generalization_class"] = None
    enriched["train_nearest_similarity"] = None
    enriched["similarity_bin"] = None
    enriched["train_frequency_bin"] = None

    for event_index, row in enriched.iterrows():
        gold_description = None
        pred_description = None
        if bool(row.gold_exists) and pd.notna(row.gold_id):
            gold_description = cache[("gold", int(row.gold_id))]
            for key, value in gold_description.items():
                enriched.at[event_index, f"gold_{key}"] = value
        if bool(row.pred_exists) and pd.notna(row.pred_id):
            pred_description = cache[("pred", int(row.pred_id))]
            for key, value in pred_description.items():
                enriched.at[event_index, f"pred_{key}"] = value

        chosen = gold_description or pred_description
        if chosen:
            enriched.at[event_index, "generalization_class"] = chosen[
                "generalization_class"
            ]
            enriched.at[event_index, "train_nearest_similarity"] = chosen[
                "train_nearest_similarity"
            ]
            enriched.at[event_index, "similarity_bin"] = chosen["similarity_bin"]
            enriched.at[event_index, "train_frequency_bin"] = chosen[
                "train_frequency_bin"
            ]
    return enriched


def _attach_entity_features(events):
    enriched = events.copy()
    feature_names = list(entity_features("").keys())
    for name in feature_names:
        enriched[name] = None
    for index, row in enriched.iterrows():
        text = row.gold_text if bool(row.gold_exists) else row.pred_text
        for name, value in entity_features(str(text or "")).items():
            enriched.at[index, name] = value
    return enriched


def _attach_audit(events, gold_audit, pred_audit):
    enriched = events.copy()
    enriched["annotation_gold_valid"] = None
    enriched["annotation_gold_issues"] = None
    enriched["annotation_pred_valid"] = None
    enriched["annotation_pred_issues"] = None
    for index, row in enriched.iterrows():
        if bool(row.gold_exists) and pd.notna(row.gold_id):
            audit = gold_audit.iloc[int(row.gold_id)]
            enriched.at[index, "annotation_gold_valid"] = bool(audit.annotation_valid)
            enriched.at[index, "annotation_gold_issues"] = audit.annotation_issues
        if bool(row.pred_exists) and pd.notna(row.pred_id):
            audit = pred_audit.iloc[int(row.pred_id)]
            enriched.at[index, "annotation_pred_valid"] = bool(audit.annotation_valid)
            enriched.at[index, "annotation_pred_issues"] = audit.annotation_issues
    return enriched


def _audit_table(spans, audit):
    return pd.concat([spans.reset_index(drop=True), audit.reset_index(drop=True)], axis=1)


def _source_metadata(value) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return {"kind": "dataframe", "rows": int(len(value))}
    path = Path(value)
    return {
        "kind": "file",
        "path": str(path),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
    }


def _write_captions(output_root: Path, figures: tuple[Path, ...] | list[Path]) -> None:
    """Backward-compatible delegate to the publication-v2 caption writer."""
    write_figure_captions(output_root, figures)
