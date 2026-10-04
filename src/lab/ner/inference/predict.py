"""The `ner.predict_entities` task: a saved model over documents, to a scored span table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.provenance import write_manifest
from lab.core.spans import spans_from_corpus
from lab.ner.encoding.encoder import WindowStrategy, encoder_from_description
from lab.ner.evaluation.metrics import token_diagnostics
from lab.ner.evaluation.scoring import span_metrics
from lab.ner.inference.io import (
    has_gold,
    read_documents,
    read_reference,
    write_gold,
    write_predictions,
)
from lab.ner.inference.loading import load_model
from lab.ner.inference.prediction import decode_spans, predict_logits

PREDICTIONS_FILENAME = "predictions.tsv"
GOLD_FILENAME = "gold.tsv"
METRICS_FILENAME = "prediction_metrics.json"
INFERENCE_MANIFEST_FILENAME = "inference_manifest.json"


@dataclass(frozen=True)
class InferenceResult:
    """What one inference run produced: the entities, any scores, and where they landed."""

    output_dir: Path
    spans: pd.DataFrame
    metrics: dict[str, Any] | None
    manifest: dict[str, Any]
    paths: dict[str, Path]

    def summary(self) -> str:
        """A one-line account of the run, with the headline scores when there are any."""
        line = f"{len(self.spans)} entities from {self.manifest['n_windows']} windows"

        if self.metrics is None:
            return line

        return (
            f"{line} | strict P/R/F1 = {self.metrics['span_strict_precision']}"
            f"/{self.metrics['span_strict_recall']}/{self.metrics['span_strict_f1']}"
            f" | char F1 = {self.metrics['char_f1']}"
        )


def predict_entities(
    model_dir: str | Path,
    documents: pd.DataFrame | dict[str, str] | str | Path,
    output_dir: str | Path,
    reference: str | Path | None = None,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    batch_size: int = 16,
    device: str = "auto",
    min_score: float = 0.0,
    min_overlap_percentage: float = 40.0,
    include_confusion: bool = False,
    strategy: str | WindowStrategy | None = None,
    pad_to_multiple_of: int | None = 8,
) -> InferenceResult:
    """
    Predict entities with the model in `model_dir` and write them to `output_dir`.

    `model_dir` holds saved weights and the `encoding.json` written beside them,
    which supplies the windowing the model was trained under — nothing about the
    encoding is restated here, so inference cannot silently disagree with
    training. Pass `strategy` only if the model was trained with a callable one.

    `documents` is a canonical corpus (frame or parquet), a directory of `.txt`
    files, or a `{doc_id: text}` mapping. Predictions are always written.
    Documents carrying gold entities are also scored; a directory or mapping
    has no gold, so scoring those needs `reference`, which also takes precedence
    over the corpus's own entities when both are given.

    Span and character metrics score the predicted spans against the gold as
    annotated, restricted to the model's `target_label` — not against the gold
    reconstructed from windows, which is what training sees. Token diagnostics
    need the encoded rows, so they are added only when `documents` itself
    carries the gold.

    `min_score` drops predicted entities below a mean token probability. Where
    two overlapping windows predict the same span, the higher-scoring copy is
    kept.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, description = load_model(model_dir, device=device)
    encoder = encoder_from_description(
        description["encoding"], tokenizer, strategy=strategy, require_target_label=False
    )

    corpus = read_documents(documents, pattern=pattern, encoding=encoding)
    rows = encoder.encode(corpus)

    if rows.empty:
        raise ValueError(
            f"No windows were produced from {len(corpus)} document(s) — nothing to predict on."
        )

    predictions = predict_logits(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=batch_size,
        device=device,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    spans = decode_spans(
        rows=rows,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=encoder.id2label,
        texts=dict(zip(corpus["doc_id"], corpus["text"])),
        min_score=min_score,
    )

    paths = {"predictions": write_predictions(spans, output_dir / PREDICTIONS_FILENAME)}
    metrics: dict[str, Any] | None = None

    gold = _gold_spans(reference, corpus, encoder.target_label)

    if gold is not None:
        paths["gold"] = write_gold(gold, output_dir / GOLD_FILENAME)
        metrics = span_metrics(
            gold=gold,
            predicted=spans,
            tags=[encoder.target_label],
            min_overlap_percentage=min_overlap_percentage,
        )

        if has_gold(corpus):
            metrics.update(
                token_diagnostics(
                    rows=rows,
                    predictions=predictions,
                    id2label=encoder.id2label,
                    include_confusion=include_confusion,
                )
            )

        paths["metrics"] = write_manifest(metrics, output_dir / METRICS_FILENAME)

    manifest = {
        "model_dir": str(Path(model_dir).resolve()),
        "model": description["model"],
        "encoding": description["encoding"],
        "output_dir": str(output_dir),
        "n_documents": int(corpus["doc_id"].nunique()),
        "n_windows": int(len(rows)),
        "n_predicted_entities": int(len(spans)),
        "min_score": float(min_score),
        "scored_against_gold": metrics is not None,
        "reference": None if reference is None else str(Path(reference).resolve()),
    }
    paths["inference_manifest"] = write_manifest(manifest, output_dir / INFERENCE_MANIFEST_FILENAME)

    return InferenceResult(
        output_dir=output_dir,
        spans=spans,
        metrics=metrics,
        manifest=manifest,
        paths=paths,
    )


def _gold_spans(
    reference: str | Path | None, corpus: pd.DataFrame, target_label: str
) -> pd.DataFrame | None:
    if reference is not None:
        return read_reference(reference, target_label)

    if has_gold(corpus):
        return spans_from_corpus(corpus, label=target_label)

    return None
