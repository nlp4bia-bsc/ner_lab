"""The `ner.predict_multihead` task: a saved multihead model over documents, to a span table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.provenance import write_manifest
from lab.ner.encoding.encoder import WindowStrategy
from lab.ner.inference.io import (
    has_gold,
    read_documents,
    resolve_gold,
    write_gold,
    write_predictions,
)
from lab.ner.inference.predict import (
    GOLD_FILENAME,
    INFERENCE_MANIFEST_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    PREDICTIONS_PARQUET_FILENAME,
    InferenceResult,
)
from lab.ner.inference.prediction import predict_logits
from lab.ner.multihead.encoding import multihead_encoder_from_description
from lab.ner.multihead.loading import load_multihead_model
from lab.ner.multihead.scoring import (
    multihead_predicted_spans,
    multihead_token_diagnostics,
    score_heads,
)


def predict_multihead(
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
    Predict entities of every head's type with the multihead model in `model_dir`.

    The `predict_entities` counterpart, taking the same documents, reference and
    options and writing the same files. Every head is decoded on its own, so the
    span table can hold entities of different types over the same text.

    Gold, from `reference` or the documents themselves, is scored per head, micro
    and macro (see `score_heads`), restricted to the types the model has heads for;
    token diagnostics are added per head when `documents` itself carries the gold.
    `min_score` drops predicted entities below a mean token probability.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, description = load_multihead_model(model_dir, device=device)
    encoder = multihead_encoder_from_description(
        description["encoding"], tokenizer, strategy=strategy
    )
    entity_labels = encoder.entity_labels

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

    spans = multihead_predicted_spans(
        rows=rows,
        predictions=predictions,
        tokenizer=tokenizer,
        entity_labels=entity_labels,
        texts=dict(zip(corpus["doc_id"], corpus["text"], strict=True)),
        include_scores=True,
    )

    if min_score > 0:
        spans = spans[spans["score"] >= min_score].reset_index(drop=True)

    paths = {
        "predictions": write_predictions(spans, output_dir / PREDICTIONS_FILENAME),
        "predictions_parquet": write_predictions(spans, output_dir / PREDICTIONS_PARQUET_FILENAME),
    }
    metrics: dict[str, Any] | None = None

    gold = resolve_gold(reference, corpus, target_label=None)

    if gold is not None:
        gold = gold[gold["label"].isin(entity_labels)].reset_index(drop=True)
        paths["gold"] = write_gold(gold, output_dir / GOLD_FILENAME)
        metrics = score_heads(
            gold=gold,
            predicted=spans,
            entity_labels=entity_labels,
            min_overlap_percentage=min_overlap_percentage,
        )

        if has_gold(corpus):
            metrics.update(
                multihead_token_diagnostics(
                    rows=rows,
                    predictions=predictions,
                    entity_labels=entity_labels,
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
