"""The `ner.predict_ensemble` task: a fitted ensemble over documents, to a scored span table."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.provenance import write_manifest
from lab.ner.ensemble.combination import combine_predictions
from lab.ner.ensemble.entities import align_entities, score_ensemble
from lab.ner.ensemble.io import check_member_weights, parameters_from_ensemble, read_ensemble
from lab.ner.ensemble.tables import predict_members, predict_word_tables
from lab.ner.ensemble.voting import align_votes, ensemble_spans
from lab.ner.evaluation.scoring import span_metrics
from lab.ner.inference.io import read_documents, resolve_gold, write_gold, write_predictions
from lab.ner.inference.predict import (
    GOLD_FILENAME,
    INFERENCE_MANIFEST_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    PREDICTIONS_PARQUET_FILENAME,
)


@dataclass(frozen=True)
class EnsembleInference:
    """What one ensemble inference run produced: the entities, any scores, and where they landed."""

    output_dir: Path
    spans: pd.DataFrame
    metrics: dict[str, Any] | None
    manifest: dict[str, Any]
    paths: dict[str, Path]

    def summary(self) -> str:
        """A one-line account of the run, with the headline scores when there are any."""
        line = (
            f"{len(self.spans)} entities from {len(self.manifest['ensemble']['members'])} members"
        )

        if self.metrics is None:
            return line

        return (
            f"{line} | strict P/R/F1 = {self.metrics['span_strict_precision']}"
            f"/{self.metrics['span_strict_recall']}/{self.metrics['span_strict_f1']}"
            f" | char F1 = {self.metrics['char_f1']}"
        )


def predict_ensemble(
    ensemble_dir: str | Path,
    documents: pd.DataFrame | dict[str, str] | str | Path,
    output_dir: str | Path,
    reference: str | Path | None = None,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    batch_size: int = 16,
    device: str = "auto",
    min_score: float = 0.0,
    min_overlap_percentage: float = 40.0,
    pad_to_multiple_of: int | None = 8,
) -> EnsembleInference:
    """
    Predict entities with the ensemble `fit_ensemble` wrote to `ensemble_dir`.

    Its `ensemble.json` names the members, their weights, the threshold and the
    kind of vote; nothing about the combination is restated here. A member whose
    weights changed since the fit raises before anything runs.

    `documents`, `reference`, `min_score` and `min_overlap_percentage` mean what
    they mean for `predict_entities`, and the same files are written: the span
    table as `predictions.tsv` and `predictions.parquet`, and with gold, the gold
    and its span and character metrics. Token diagnostics are not computed, since
    the members do not share token rows. A span's score is the mean combined
    probability of the tags its words took.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    timings = {}
    ensemble = read_ensemble(ensemble_dir)
    check_member_weights(ensemble)
    members = ensemble["members"]
    target_label = (
        ensemble["target_label"]
        if ensemble["format_version"] == 1 or ensemble["legacy_semantics"]
        else None
    )

    corpus = read_documents(documents, pattern=pattern, encoding=encoding)
    if ensemble["format_version"] == 1:
        tables = predict_word_tables(
            [member["model_dir"] for member in members],
            corpus,
            batch_size=batch_size,
            device=device,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        word_votes = align_votes(tables)

        spans = ensemble_spans(
            word_votes,
            weights=[member["weight"] for member in members],
            threshold=ensemble["threshold"],
            votes=ensemble["votes"],
            texts=dict(zip(corpus["doc_id"], corpus["text"])),
            target_label=target_label,
        )
        n_words = len(word_votes.words)
    else:
        predictions = predict_members(
            [member["model_dir"] for member in members],
            corpus,
            batch_size,
            device,
            pad_to_multiple_of,
            use_annotations=ensemble["legacy_semantics"],
        )
        aligned_started = time.perf_counter()
        aligned = (
            align_entities(predictions, ensemble["entity_labels"])
            if ensemble["method"] in ("majority", "soft", "weighted")
            else {}
        )
        timings["alignment_sec"] = time.perf_counter() - aligned_started
        combined_started = time.perf_counter()
        spans = combine_predictions(
            predictions,
            aligned,
            ensemble["method"],
            parameters_from_ensemble(ensemble),
            ensemble["votes"],
            dict(zip(corpus["doc_id"], corpus["text"])),
            ensemble["selected_members"],
            ensemble["min_support"],
            ensemble["missing_policy"],
            entity_labels=ensemble["entity_labels"],
        )
        timings["combine_decode_sec"] = time.perf_counter() - combined_started
        timings["members"] = [prediction.timings for prediction in predictions]
        n_words = len(
            pd.concat(
                [member.words[["doc_id", "word_id"]] for member in predictions]
            ).drop_duplicates()
        )

    if min_score > 0:
        spans = spans[spans["score"] >= min_score].reset_index(drop=True)

    paths = {
        "predictions": write_predictions(spans, output_dir / PREDICTIONS_FILENAME),
        "predictions_parquet": write_predictions(spans, output_dir / PREDICTIONS_PARQUET_FILENAME),
    }
    metrics: dict[str, Any] | None = None

    gold = resolve_gold(reference, corpus, target_label)

    if gold is not None:
        paths["gold"] = write_gold(gold, output_dir / GOLD_FILENAME)
        if ensemble["format_version"] == 1:
            metrics = span_metrics(
                gold=gold,
                predicted=spans,
                tags=[target_label],
                min_overlap_percentage=min_overlap_percentage,
            )
        else:
            if ensemble["restrict_gold"]:
                gold = gold[gold["label"].isin(ensemble["entity_labels"])].reset_index(drop=True)
                paths["gold"] = write_gold(gold, output_dir / GOLD_FILENAME)
            metrics = score_ensemble(
                gold,
                spans,
                ensemble["entity_labels"],
                ensemble["legacy_semantics"],
                min_overlap_percentage,
            )
        paths["metrics"] = write_manifest(metrics, output_dir / METRICS_FILENAME)

    manifest = {
        "ensemble_dir": str(Path(ensemble_dir).resolve()),
        "ensemble": ensemble,
        "output_dir": str(output_dir),
        "n_documents": int(corpus["doc_id"].nunique()),
        "n_words": int(n_words),
        "n_predicted_entities": int(len(spans)),
        "min_score": float(min_score),
        "timings": timings,
        "duration_sec": time.perf_counter() - started,
        "scored_against_gold": metrics is not None,
        "reference": None if reference is None else str(Path(reference).resolve()),
    }
    paths["inference_manifest"] = write_manifest(manifest, output_dir / INFERENCE_MANIFEST_FILENAME)

    return EnsembleInference(
        output_dir=output_dir,
        spans=spans,
        metrics=metrics,
        manifest=manifest,
        paths=paths,
    )
