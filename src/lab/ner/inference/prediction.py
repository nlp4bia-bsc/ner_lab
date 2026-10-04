"""Running a model over encoded rows and decoding the entities it predicts."""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import torch
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from lab.ner.encoding.rows import IGNORE_INDEX
from lab.ner.evaluation.spans import predicted_spans
from lab.ner.inference.loading import resolve_device
from lab.ner.training.arguments import for_inference
from lab.ner.training.arguments import training_arguments as build_training_arguments
from lab.ner.training.dataset import to_dataset
from lab.ner.training.trainer import resolve_trainer_class


def predict_logits(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
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
    tokenizer: PreTrainedTokenizerBase,
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


def _predict(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
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
