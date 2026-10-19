"""A member's word table: its own tag and its tag probabilities at every word it labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from lab.ner.encoding.encoder import encoder_from_description
from lab.ner.encoding.rows import IGNORE_INDEX, special_token_template
from lab.ner.evaluation.spans import content_positions, ensure_int_list, ensure_word_ids, softmax
from lab.ner.inference.loading import load_model
from lab.ner.inference.prediction import predict_logits
from lab.ner.models.crf import decoded_paths_to_logits
from lab.ner.training.assessment import release_accelerator_memory
from lab.ner.training.trainer import CRFTrainer

TAGS = ("O", "B", "I")
PROBABILITY_COLUMNS = [f"p_{tag}" for tag in TAGS]
WORD_TABLE_COLUMNS = ["doc_id", "word_id", "tag", *PROBABILITY_COLUMNS]


class MarginalCRFTrainer(CRFTrainer):
    """
    A `CRFTrainer` whose predictions carry the CRF's marginals beside its Viterbi path.

    Each position's prediction is its log-marginals followed by the Viterbi path's
    one-hot logits, concatenated on the last axis, so one forward pass gives an
    ensemble both what the member believes and what it decides.
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss.detach() if outputs.loss is not None else None

            if prediction_loss_only:
                return loss, None, None

            emissions = outputs.logits
            attention_mask = inputs.get("attention_mask")
            decoder = model.module if hasattr(model, "module") else model

            log_marginals = decoder.marginals_from_emissions(
                emissions=emissions, attention_mask=attention_mask
            )
            viterbi = decoded_paths_to_logits(
                decoded_paths=decoder.decode_from_emissions(
                    emissions=emissions, attention_mask=attention_mask
                ),
                sequence_length=emissions.shape[1],
                num_labels=emissions.shape[2],
                device=emissions.device,
            )

        predictions = torch.cat([log_marginals, viterbi], dim=-1)

        return loss, predictions.detach(), labels.detach() if labels is not None else None


def predict_word_tables(
    model_dirs: Sequence[str | Path],
    corpus: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
) -> list[pd.DataFrame]:
    """
    Every member's word table over `corpus`, one member in memory at a time.

    Each member windows the corpus as its own `encoding.json` says, so members with
    different tokenizers, lengths and strategies still meet on the same words. The
    previous member is released before the next one loads.
    """
    tables = []

    for model_dir in model_dirs:
        tables.append(
            predict_word_table(
                model_dir=model_dir,
                corpus=corpus,
                batch_size=batch_size,
                device=device,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        )
        release_accelerator_memory()

    return tables


def predict_word_table(
    model_dir: str | Path,
    corpus: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
) -> pd.DataFrame:
    """
    One saved model's word table over `corpus`.

    A linear model's probabilities are the softmax of its logits and its tag their
    argmax. A model exposing `marginals_from_emissions` — the CRF — gives its
    forward-backward marginals as probabilities and its Viterbi path as its tag,
    which is not always the per-word argmax of those marginals.
    """
    model, tokenizer, description = load_model(model_dir, device=device)
    encoder = encoder_from_description(
        description["encoding"], tokenizer, require_target_label=False
    )
    rows = encoder.encode(corpus)

    if rows.empty:
        raise ValueError(
            f"No windows were produced from {len(corpus)} document(s) for {model_dir} — "
            "nothing to predict on."
        )

    has_marginals = hasattr(model, "marginals_from_emissions")
    predictions = predict_logits(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=batch_size,
        device=device,
        pad_to_multiple_of=pad_to_multiple_of,
        trainer_class=MarginalCRFTrainer if has_marginals else None,
    ).astype(np.float32)

    if has_marginals:
        log_marginals, viterbi = np.split(predictions, 2, axis=-1)
        probabilities, tag_ids = softmax(log_marginals), viterbi.argmax(axis=-1)
    else:
        probabilities, tag_ids = softmax(predictions), predictions.argmax(axis=-1)

    return word_table(
        rows=rows,
        probabilities=probabilities,
        tag_ids=tag_ids,
        tokenizer=tokenizer,
        id2label=encoder.id2label,
    )


def word_table(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    tag_ids: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
) -> pd.DataFrame:
    """
    One row per labelled word, keyed `(doc_id, word_id)`, with the tag and its probabilities.

    Only positions the encoder labelled are read — a word's first subword, inside a
    window's core — and `word_id` is the document-level UAX #29 word index, which
    every tokenizer shares. Probability columns are taken by label name, not by id.
    A word labelled twice, which a window starting mid-word can cause, keeps its
    first row.
    """
    tag_by_id = {int(index): _bio_tag(str(label)) for index, label in id2label.items()}
    id_by_tag = {tag: index for index, tag in tag_by_id.items()}
    probability_ids = [id_by_tag[tag] for tag in TAGS]
    prefix_ids, suffix_ids = special_token_template(tokenizer)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(rows.itertuples(index=False)):
        labels = ensure_int_list(row.labels)
        positions = content_positions(ensure_int_list(row.input_ids), prefix_ids, suffix_ids)
        word_by_position = dict(zip(positions, ensure_word_ids(row.word_ids)))

        for position, label_id in enumerate(labels):
            if label_id == ignore_index:
                continue

            row_probabilities = probabilities[index][position][probability_ids]
            records.append(
                {
                    "doc_id": str(row.doc_id),
                    "word_id": int(word_by_position[position]),
                    "tag": tag_by_id[int(tag_ids[index][position])],
                    **dict(zip(PROBABILITY_COLUMNS, row_probabilities.tolist())),
                }
            )

    table_df = pd.DataFrame(records, columns=WORD_TABLE_COLUMNS)

    return table_df.drop_duplicates(subset=["doc_id", "word_id"]).reset_index(drop=True)


def _bio_tag(label: str) -> str:
    if label == "O":
        return "O"

    prefix, _, _ = label.partition("-")

    if prefix not in ("B", "I"):
        raise ValueError(f"Label {label!r} is not a BIO label; expected O, B-* or I-*.")

    return prefix
