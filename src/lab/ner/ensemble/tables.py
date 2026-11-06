"""A member's word table: its own tag and its tag probabilities at every word it labels."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from lab.core.spans import span_dataframe
from lab.ner.encoding.encoder import encoder_from_description
from lab.ner.encoding.rows import IGNORE_INDEX, special_token_template
from lab.ner.evaluation.spans import (
    _aligned_offsets,
    bio_to_spans,
    content_positions,
    ensure_int_list,
    ensure_word_ids,
    softmax,
)
from lab.ner.inference.loading import load_model
from lab.ner.inference.prediction import predict_logits
from lab.ner.models.crf import decoded_paths_to_logits
from lab.ner.training.assessment import release_accelerator_memory
from lab.ner.training.trainer import CRFTrainer, WordMarginalTrainer, read_model_encoding

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


@dataclass(frozen=True)
class MemberPredictions:
    """Sparse labelled word distributions and original per-window decoded spans."""

    words: pd.DataFrame
    spans: pd.DataFrame
    entity_labels: tuple[str, ...]
    timings: dict[str, float] = field(default_factory=dict)


def predict_members(
    model_dirs: Sequence[str | Path],
    corpus: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
    use_annotations: bool = False,
) -> list[MemberPredictions]:
    """Run each checkpoint once, releasing it before loading the next member."""
    inputs = corpus if use_annotations else corpus.assign(entities_json="[]", n_entities=0)
    predictions = []
    for model_dir in model_dirs:
        predictions.append(
            predict_member(model_dir, inputs, batch_size, device, pad_to_multiple_of)
        )
        release_accelerator_memory()

    return predictions


def predict_member(
    model_dir: str | Path,
    corpus: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
) -> MemberPredictions:
    """Recover all heads, decisions and probabilities from a single batched inference."""
    started = time.perf_counter()
    model, tokenizer, description = load_model(model_dir, device=device)
    loaded = time.perf_counter()
    encoder = encoder_from_description(
        description["encoding"], tokenizer, require_target_label=False
    )
    inputs = (
        corpus.assign(entities_json="[]", n_entities=0)
        if getattr(model, "label_schema", None) == "multilabel"
        else corpus
    )
    rows = encoder.encode(inputs)
    if rows.empty:
        raise ValueError(f"No windows from {len(corpus)} documents for {model_dir}.")
    encoded = time.perf_counter()
    binary = getattr(model, "label_schema", None) == "multilabel"
    structured = hasattr(model, "structured_from_emissions")
    has_marginals = structured or hasattr(model, "marginals_from_emissions")
    predictions = predict_logits(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=batch_size,
        device=device,
        pad_to_multiple_of=pad_to_multiple_of,
        trainer_class=(WordMarginalTrainer if structured else MarginalCRFTrainer)
        if has_marginals
        else None,
    ).astype(np.float32)
    inferred = time.perf_counter()
    if binary:
        from dataclasses import asdict

        from lab.ner.evaluation.multilabel import binary_predictions, categorical_rows

        rows.attrs["multilabel_decoding"] = asdict(model.decoding)
        decisions, probabilities = binary_predictions(rows, predictions, model.entity_labels)
        tag_ids = decisions
        rows = categorical_rows(rows)
    elif has_marginals:
        log_marginals, viterbi = np.split(predictions, 2, axis=-1)
        probabilities = np.exp(log_marginals) if structured else softmax(log_marginals)
        tag_ids = viterbi.argmax(axis=-1)
    else:
        probabilities, tag_ids = softmax(predictions), predictions.argmax(axis=-1)
    labels = tuple(encoder.entity_labels or [encoder.target_label])
    adapted = adapt_predictions(
        rows,
        probabilities,
        tag_ids,
        tokenizer,
        encoder.id2label,
        labels,
        dict(zip(corpus["doc_id"], corpus["text"])),
    )
    return MemberPredictions(
        adapted.words,
        adapted.spans,
        labels,
        {
            "load_sec": loaded - started,
            "encode_sec": encoded - loaded,
            "inference_sec": inferred - encoded,
            "adapt_decode_sec": time.perf_counter() - inferred,
        },
    )


def predict_word_table(
    model_dir: str | Path,
    corpus: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
    pad_to_multiple_of: int | None = 8,
) -> pd.DataFrame:
    """Return the historical table for a classic member, or a labelled table for Multi-Head."""
    prediction = predict_member(model_dir, corpus, batch_size, device, pad_to_multiple_of)
    if read_model_encoding(model_dir)["encoding"]["target_label"] is not None:
        return prediction.words[WORD_TABLE_COLUMNS]

    return prediction.words


def adapt_predictions(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    tag_ids: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    id2label: dict,
    entity_labels: tuple[str, ...],
    texts: dict[str, str],
) -> MemberPredictions:
    """Align every head to UAX words and decode original offsets without rerunning a model."""
    multihead = probabilities.ndim == 4
    if multihead:
        if probabilities.shape[2:] != (len(entity_labels), 3):
            raise ValueError("Multi-Head probabilities must match entity_labels and BIO classes.")
    else:
        tag_by_id = {int(index): _bio_tag(label) for index, label in id2label.items()}
        class_ids = [
            next(index for index, tag in tag_by_id.items() if tag == value) for value in TAGS
        ]
        probabilities = probabilities[:, :, class_ids][:, :, None, :]
        tag_ids = np.asarray([[TAGS.index(tag_by_id[int(tag)]) for tag in row] for row in tag_ids])
        tag_ids = tag_ids[:, :, None]
    prefix, suffix = special_token_template(tokenizer)
    frames, spans = [], []
    for index, row in enumerate(rows.itertuples(index=False)):
        labels = np.asarray(ensure_int_list(row.labels))
        labels = labels if multihead else labels[:, None]
        positions = content_positions(ensure_int_list(row.input_ids), prefix, suffix)
        words = dict(zip(positions, ensure_word_ids(row.word_ids)))
        aligned = _aligned_offsets(row, prefix, suffix)
        for head, label in enumerate(entity_labels):
            supervised = [
                p for p in positions if labels[p, head] != IGNORE_INDEX and words[p] is not None
            ]
            values = probabilities[index, supervised, head, :]
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.allclose(values.sum(axis=-1), 1, atol=1e-5)
            ):
                raise ValueError(f"Invalid BIO probabilities for entity {label!r}.")
            tags = tag_ids[index, supervised, head]
            frames.append(
                pd.DataFrame(
                    {
                        "doc_id": str(row.doc_id),
                        "word_id": [words[p] for p in supervised],
                        "entity_label": label,
                        "tag": [TAGS[int(tag)] for tag in tags],
                        **dict(zip(PROBABILITY_COLUMNS, values.T)),
                    }
                )
            )
            spans.extend(
                bio_to_spans(
                    str(row.doc_id),
                    [aligned[p] for p in supervised],
                    ["O" if tag == 0 else f"{TAGS[int(tag)]}-{label}" for tag in tags],
                    scores=values[np.arange(len(tags)), tags].tolist(),
                    text=texts[str(row.doc_id)],
                )
            )
    columns = [*WORD_TABLE_COLUMNS, "entity_label"]
    words_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    words_df = words_df.drop_duplicates(["doc_id", "word_id", "entity_label"]).reset_index(
        drop=True
    )

    return MemberPredictions(words_df, span_dataframe(spans, scored=True), entity_labels)


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
