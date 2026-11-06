"""The `ner.predict_entities` task: a saved model over documents, to a scored span table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lab.core.provenance import write_manifest
from lab.ner.encoding.encoder import WindowStrategy, encoder_from_description
from lab.ner.evaluation.metrics import token_diagnostics
from lab.ner.evaluation.scoring import span_metrics
from lab.ner.inference.io import (
    has_gold,
    read_documents,
    resolve_gold,
    write_gold,
    write_predictions,
)
from lab.ner.inference.loading import load_model
from lab.ner.inference.prediction import decode_spans, predict_logits

PREDICTIONS_FILENAME = "predictions.tsv"
PREDICTIONS_PARQUET_FILENAME = "predictions.parquet"
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
    decoder_path: str | Path | None = None,
) -> InferenceResult:
    """
    Predict entities with the model in `model_dir` and write them to `output_dir`.

    `model_dir` holds saved weights and the `encoding.json` written beside them,
    which supplies the windowing the model was trained under — nothing about the
    encoding is restated here, so inference cannot silently disagree with
    training. Pass `strategy` only if the model was trained with a callable one.

    `documents` is a canonical corpus (frame or parquet), a directory of `.txt`
    files, or a `{doc_id: text}` mapping. Predictions are always written, as both
    `predictions.tsv` and `predictions.parquet`.
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
    if decoder_path is not None:
        from lab.ner.inference.binary_settings import read_binary_decoding

        if getattr(model, "label_schema", None) != "multilabel":
            raise ValueError("decoder_path requires a multilabel checkpoint.")
        model.decoding = read_binary_decoding(decoder_path, model_dir, model.entity_labels)
    encoder = encoder_from_description(
        description["encoding"], tokenizer, strategy=strategy, require_target_label=False
    )

    corpus = read_documents(documents, pattern=pattern, encoding=encoding)
    binary = getattr(model, "label_schema", None) == "multilabel"
    structured = hasattr(model, "structured_from_emissions")
    inputs = corpus.assign(entities_json="[]", n_entities=0) if structured else corpus
    rows = encoder.encode(inputs)
    if binary:
        from dataclasses import asdict

        rows.attrs["multilabel_decoding"] = asdict(model.decoding)

    if rows.empty:
        raise ValueError(
            f"No windows were produced from {len(corpus)} document(s) — nothing to predict on."
        )

    from lab.ner.training.trainer import WordMarginalTrainer

    predictions = predict_logits(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=batch_size,
        device=device,
        pad_to_multiple_of=pad_to_multiple_of,
        trainer_class=WordMarginalTrainer if structured else None,
    )

    probabilities = None
    if structured:
        log_marginals, predictions = np.split(predictions, 2, axis=-1)
        probabilities = np.exp(log_marginals)

    spans = decode_spans(
        rows=rows,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=encoder.id2label,
        texts=dict(zip(corpus["doc_id"], corpus["text"])),
        min_score=min_score,
        probabilities=probabilities,
    )

    paths = {
        "predictions": write_predictions(spans, output_dir / PREDICTIONS_FILENAME),
        "predictions_parquet": write_predictions(spans, output_dir / PREDICTIONS_PARQUET_FILENAME),
    }
    metrics: dict[str, Any] | None = None

    gold = resolve_gold(reference, corpus, encoder.target_label)

    if gold is not None:
        paths["gold"] = write_gold(gold, output_dir / GOLD_FILENAME)
        if encoder.target_label is None:
            from lab.ner.evaluation.multihead import strict_multihead_metrics

            metrics = strict_multihead_metrics(gold, spans, encoder.entity_labels)
        else:
            metrics = span_metrics(
                gold=gold,
                predicted=spans,
                tags=[encoder.target_label],
                min_overlap_percentage=min_overlap_percentage,
            )

        if has_gold(corpus):
            diagnostic_rows = encoder.encode(corpus) if structured else rows
            if binary:
                diagnostic_rows.attrs["multilabel_decoding"] = rows.attrs["multilabel_decoding"]
                from lab.ner.evaluation.multilabel import gold_representation_metrics

                metrics.update(
                    gold_representation_metrics(diagnostic_rows, tokenizer, model.entity_labels)
                )
            metrics.update(
                token_diagnostics(
                    rows=diagnostic_rows,
                    predictions=predictions,
                    id2label=encoder.id2label,
                    include_confusion=include_confusion,
                )
            )

        paths["metrics"] = write_manifest(metrics, output_dir / METRICS_FILENAME)

    manifest = {
        "model_dir": str(Path(model_dir).resolve()),
        "model": description["model"],
        **(
            {
                "decoding": asdict(model.decoding),
                "decoder_path": str(decoder_path) if decoder_path else None,
            }
            if binary
            else {}
        ),
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
