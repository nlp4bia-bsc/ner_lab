"""Running a saved model over documents and writing the entities it predicts."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ner_lab.data.brat import resolve_documents
from ner_lab.data.corpus import DOCUMENT_COLUMNS, validate_corpus
from ner_lab.encoding.encoder import Encoder, WindowStrategy, encoder_from_description
from ner_lab.encoding.rows import IGNORE_INDEX
from ner_lab.evaluation.metrics import evaluate_predictions
from ner_lab.evaluation.multiclinner import (
    evaluate_annotations,
    format_summary,
    read_annotation_tsv,
    spans_from_corpus,
    write_annotation_tsv,
)
from ner_lab.evaluation.spans import SCORED_SPAN_COLUMNS, predicted_spans
from ner_lab.models.registry import Architecture, build_model
from ner_lab.provenance import write_manifest
from ner_lab.training.arguments import for_inference
from ner_lab.training.arguments import training_arguments as build_training_arguments
from ner_lab.training.dataset import to_dataset
from ner_lab.training.trainer import read_model_encoding, resolve_trainer_class

PREDICTIONS_FILENAME = "predictions.tsv"
GOLD_FILENAME = "gold.tsv"
METRICS_FILENAME = "prediction_metrics.json"
OFFICIAL_METRICS_FILENAME = "multiclinner_eval.json"
INFERENCE_MANIFEST_FILENAME = "inference_manifest.json"


@dataclass(frozen=True)
class InferenceResult:
    """What one inference run produced: the entities, any scores, and where they landed."""

    output_dir: Path
    spans: pd.DataFrame
    metrics: dict[str, Any] | None
    official: dict[str, Any] | None
    manifest: dict[str, Any]
    paths: dict[str, Path]

    def summary(self) -> str:
        """A one-line account of the run, plus the official scores when there are any."""
        line = f"{len(self.spans)} entities from {self.manifest['n_windows']} windows"

        if self.official is None:
            return line

        return f"{line}\n{format_summary(self.official)}"


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
    Documents carrying gold entities are also scored, both with this library's
    span and token metrics and with the official MultiClinNER scorer; a
    directory or mapping has no gold, so scoring those needs `reference`.

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
    official: dict[str, Any] | None = None

    if has_gold(corpus):
        metrics = evaluate_predictions(
            rows=rows,
            predictions=predictions,
            tokenizer=tokenizer,
            id2label=encoder.id2label,
            min_overlap_percentage=min_overlap_percentage,
            include_confusion=include_confusion,
        )
        paths["metrics"] = write_manifest(metrics, output_dir / METRICS_FILENAME)

    gold = read_reference(reference, encoder.target_label) if reference is not None else None

    if gold is None and has_gold(corpus):
        gold = spans_from_corpus(corpus, entity=encoder.target_label)

    if gold is not None:
        paths["gold"] = write_annotation_tsv(gold, output_dir / GOLD_FILENAME)
        official = evaluate_annotations(gold=gold, predicted=spans, entity=encoder.target_label)
        paths["official_metrics"] = write_manifest(
            official, output_dir / OFFICIAL_METRICS_FILENAME
        )

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
        "multiclinner_eval": official,
    }
    paths["inference_manifest"] = write_manifest(
        manifest, output_dir / INFERENCE_MANIFEST_FILENAME
    )

    return InferenceResult(
        output_dir=output_dir,
        spans=spans,
        metrics=metrics,
        official=official,
        manifest=manifest,
        paths=paths,
    )


def load_model(
    model_dir: str | Path,
    device: str = "auto",
    architecture: str | Architecture | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """
    Load a saved model, its tokenizer, and the `encoding.json` beside them.

    A linear model is a standard `from_pretrained` load. A CRF model is not a
    `PreTrainedModel`, so the Trainer saved it as a bare state dict: it is
    rebuilt over the backbone `encoding.json` names, then the fine-tuned weights
    are loaded on top. `architecture` overrides the recorded name, which a model
    trained with a custom architecture callable needs.
    """
    model_dir = Path(model_dir).resolve()
    description = read_model_encoding(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    label2id = description["encoding"]["label2id"]
    id2label = {int(index): label for label, index in label2id.items()}

    recorded = description["model"]
    architecture = recorded["architecture"] if architecture is None else architecture

    if architecture == "linear":
        model = build_model(
            checkpoint=str(model_dir),
            label2id=label2id,
            id2label=id2label,
            architecture=architecture,
            **recorded.get("architecture_kwargs", {}),
        )
    else:
        model = build_model(
            checkpoint=recorded["checkpoint"],
            label2id=label2id,
            id2label=id2label,
            architecture=architecture,
            **recorded.get("architecture_kwargs", {}),
        )
        model.load_state_dict(load_state_dict(model_dir))

    model.to(resolve_device(device))
    model.eval()

    return model, tokenizer, description


def load_state_dict(model_dir: str | Path) -> dict[str, Any]:
    """Read the weights the Trainer wrote for a model it could not `save_pretrained`."""
    model_dir = Path(model_dir)
    safetensors_path = model_dir / "model.safetensors"
    binary_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    if binary_path.exists():
        return torch.load(binary_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin in {model_dir}."
    )


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve `"auto"` against what this machine actually has."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested but CUDA is not available.")

    return torch.device(device)


def predict_logits(
    model,
    tokenizer,
    rows: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
    training_arguments: TrainingArguments | None = None,
    trainer_class: type[Trainer] | None = None,
) -> np.ndarray:
    """
    Run `model` over encoded rows and return one logit array, row by row.

    A CRF model goes through `CRFTrainer`, whose Viterbi paths come back
    re-encoded as one-hot logits, so both architectures reach decoding in the
    same shape. The arguments are forced to evaluate and save nothing —
    `Trainer` refuses to build otherwise when no eval dataset is passed.

    `device` reaches the built arguments rather than only the model: a `Trainer`
    places the model itself, so a model moved to the CPU by hand still runs on
    the GPU if the arguments allow it. Passing your own `training_arguments`
    makes them the authority, `device` included.

    Nothing is written: without `training_arguments`, the `output_dir` the
    Trainer insists on is a temporary directory that goes away with the call.
    """
    if training_arguments is not None:
        return _predict(
            model, tokenizer, rows, training_arguments, pad_to_multiple_of, trainer_class
        )

    on_cpu = resolve_device(device).type == "cpu"

    with tempfile.TemporaryDirectory() as temporary_dir:
        return _predict(
            model,
            tokenizer,
            rows,
            build_training_arguments(
                temporary_dir,
                per_device_eval_batch_size=batch_size,
                use_cpu=on_cpu,
                fp16=not on_cpu,
            ),
            pad_to_multiple_of,
            trainer_class,
        )


def decode_spans(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer,
    id2label: dict,
    texts: dict[str, str] | None = None,
    min_score: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> pd.DataFrame:
    """
    Decode predictions into scored entity spans, dropping those below `min_score`.

    The same span reconstruction evaluation scores with, asked for the surface
    form and the confidence a prediction file needs and a metric does not.
    """
    spans = predicted_spans(
        rows=rows,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=id2label,
        ignore_index=ignore_index,
        texts=texts or {},
        include_scores=True,
    )

    if min_score > 0:
        spans = spans[spans["score"] >= min_score].reset_index(drop=True)

    return spans


def write_predictions(spans: pd.DataFrame, path: str | Path) -> Path:
    """
    Write predicted entities as a TSV: the official schema plus a `score` column.

    The extra column is why this is not `write_annotation_tsv` — the official
    format has no place for confidence, and `multiclinner`'s reader takes the
    five columns it declares and ignores the rest, so the file scores as-is.
    """
    path = Path(path)
    spans[SCORED_SPAN_COLUMNS].to_csv(path, sep="\t", index=False)

    return path


def read_documents(
    documents: pd.DataFrame | dict[str, str] | str | Path,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Read what is to be predicted on into a canonical corpus frame.

    A corpus frame or parquet keeps its `entities_json`, which is what makes it
    scorable. Raw text has none, so every document gets an empty entity list —
    the encoder still labels every content token `O`, which is exactly the
    trustworthy position span reconstruction decodes.
    """
    if isinstance(documents, pd.DataFrame):
        return validate_corpus(documents)

    path = Path(documents) if not isinstance(documents, dict) else None

    if path is not None and path.is_file():
        return validate_corpus(pd.read_parquet(path))

    texts = resolve_documents(documents, pattern=pattern, encoding=encoding)

    if not texts:
        raise FileNotFoundError(f"No documents matching {pattern!r} in {documents}.")

    return validate_corpus(
        pd.DataFrame(
            [
                {
                    "doc_id": doc_id,
                    "text": text,
                    "entities_json": json.dumps([]),
                    "n_entities": 0,
                }
                for doc_id, text in sorted(texts.items())
            ],
            columns=DOCUMENT_COLUMNS,
        )
    )


def has_gold(corpus: pd.DataFrame) -> bool:
    """Whether a corpus carries any annotation to score predictions against."""
    return bool(corpus["n_entities"].sum())


def _predict(
    model,
    tokenizer,
    rows: pd.DataFrame,
    training_arguments: TrainingArguments,
    pad_to_multiple_of: int | None,
    trainer_class: type[Trainer] | None,
) -> np.ndarray:
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    arguments = {
        "model": model,
        "args": for_inference(training_arguments),
        "data_collator": collator,
    }
    resolved = trainer_class or resolve_trainer_class(model)

    try:
        trainer = resolved(processing_class=tokenizer, **arguments)
    except TypeError:
        trainer = resolved(tokenizer=tokenizer, **arguments)

    predictions = trainer.predict(to_dataset(rows)).predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    return np.asarray(predictions)


def read_reference(reference: str | Path, target_label: str | None = None) -> pd.DataFrame:
    """Read a gold set for the official scorer, from an annotation TSV or a corpus parquet."""
    path = Path(reference)

    if not path.exists():
        raise FileNotFoundError(f"Reference does not exist: {path}")

    if path.suffix == ".parquet":
        return spans_from_corpus(path, entity=target_label)

    annotations = read_annotation_tsv(path)

    if target_label:
        annotations = annotations[annotations["label"] == target_label].reset_index(drop=True)

    return annotations
