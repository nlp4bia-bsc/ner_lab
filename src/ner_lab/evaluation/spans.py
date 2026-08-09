"""Reconstructing character-offset spans from encoded rows and model predictions."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from ner_lab.encoding.rows import IGNORE_INDEX

SPAN_COLUMNS = ["filename", "label", "start_span", "end_span", "text"]
SCORED_SPAN_COLUMNS = [*SPAN_COLUMNS, "score"]
ROW_COLUMNS = ("doc_id", "input_ids", "labels", "token_offsets", "word_ids")


def gold_spans(
    rows: pd.DataFrame,
    tokenizer,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
) -> pd.DataFrame:
    """Rebuild the annotated spans that a set of encoded rows encodes."""
    _require_columns(rows)

    id2label = _normalize_id2label(id2label)
    spans: list[dict[str, Any]] = []

    for row in rows.itertuples(index=False):
        labels = ensure_int_list(row.labels)
        aligned = _aligned_offsets(row, tokenizer)

        tags, offsets = [], []

        for position, label_id in enumerate(labels):
            if label_id == ignore_index or position not in aligned:
                continue

            tags.append(id2label[label_id])
            offsets.append(aligned[position])

        spans.extend(bio_to_spans(str(row.doc_id), offsets, tags))

    return span_dataframe(spans)


def predicted_spans(
    rows: pd.DataFrame,
    predictions: np.ndarray,
    tokenizer,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
    texts: dict[str, str] | None = None,
    include_scores: bool = False,
) -> pd.DataFrame:
    """
    Rebuild the spans a model predicted.

    Only positions the encoder left unmasked are decoded, so special tokens,
    continuation subwords and the context strategy's unlabelled flanks are
    skipped — a prediction there has no gold counterpart to be scored against.

    Scoring reads offsets alone, so `texts` and `include_scores` are off by
    default and exist for inference, which reports spans rather than scores them:
    `texts` fills each span's surface form from `{doc_id: text}`, and
    `include_scores` adds the mean softmax probability of the tokens behind it.
    """
    _require_columns(rows)

    predicted_ids = np.argmax(predictions, axis=-1)

    if len(rows) != len(predicted_ids):
        raise ValueError(
            f"rows and predictions must be the same length. "
            f"Received {len(rows)} rows and {len(predicted_ids)} predictions."
        )

    id2label = _normalize_id2label(id2label)
    probabilities = softmax(predictions) if include_scores else None
    spans: list[dict[str, Any]] = []

    for index, row in enumerate(rows.itertuples(index=False)):
        labels = ensure_int_list(row.labels)
        aligned = _aligned_offsets(row, tokenizer)
        row_predictions = predicted_ids[index][: len(labels)].tolist()

        tags, offsets, scores = [], [], []

        for position in range(min(len(labels), len(row_predictions))):
            if labels[position] == ignore_index or position not in aligned:
                continue

            predicted_id = int(row_predictions[position])

            tags.append(id2label[predicted_id])
            offsets.append(aligned[position])

            if probabilities is not None:
                scores.append(float(probabilities[index][position][predicted_id]))

        spans.extend(
            bio_to_spans(
                str(row.doc_id),
                offsets,
                tags,
                scores=scores if probabilities is not None else None,
                text=None if texts is None else texts.get(str(row.doc_id), ""),
            )
        )

    return span_dataframe(spans, scored=include_scores)


def bio_to_spans(
    filename: str,
    token_offsets: list[tuple[int, int]],
    bio_tags: list[str],
    scores: list[float] | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Merge a BIO tag sequence into span rows.

    `text` is left empty unless the document's own text is passed, since only
    offsets and labels are available here and the scorers match on offsets.
    With `scores`, each span carries the mean of its tokens' scores.
    """
    if len(token_offsets) != len(bio_tags):
        raise ValueError("token_offsets and bio_tags must be the same length.")

    if scores is not None and len(scores) != len(bio_tags):
        raise ValueError("scores and bio_tags must be the same length.")

    spans: list[dict[str, Any]] = []
    collected: list[list[float]] = []
    current: dict[str, Any] | None = None

    for position, ((start, end), tag) in enumerate(zip(token_offsets, bio_tags)):
        prefix, _, label = tag.partition("-")

        if not label or prefix not in ("B", "I"):
            current = None
            continue

        if prefix == "I" and current is not None and current["label"] == label:
            current["end_span"] = int(end)

            if scores is not None:
                collected[-1].append(scores[position])

            continue

        current = {
            "filename": filename,
            "label": label,
            "start_span": int(start),
            "end_span": int(end),
            "text": "",
        }
        spans.append(current)

        if scores is not None:
            collected.append([scores[position]])

    for index, span in enumerate(spans):
        if text is not None:
            span["text"] = text[span["start_span"]:span["end_span"]]

        if scores is not None:
            span["score"] = float(np.mean(collected[index]))

    return spans


def span_dataframe(
    spans: list[dict[str, Any]],
    scored: bool | None = None,
) -> pd.DataFrame:
    """
    Build the canonical span frame, dropping duplicate spans.

    Scored spans keep the highest-scoring copy of a span two windows both
    predicted, and come back in document order; unscored ones keep the first
    copy and stay in the order the rows produced them.

    `scored` is read off the spans themselves unless it is given. A caller that
    decoded scores passes it, so predicting nothing still returns a frame with
    the `score` column its next step reads.
    """
    scored = (bool(spans) and "score" in spans[0]) if scored is None else scored
    columns = SCORED_SPAN_COLUMNS if scored else SPAN_COLUMNS

    if not spans:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(spans, columns=columns)

    if scored:
        frame = frame.sort_values("score", ascending=False)

    frame = frame.drop_duplicates(subset=["filename", "label", "start_span", "end_span"])

    if scored:
        frame = frame.sort_values(["filename", "start_span", "end_span"])

    return frame.reset_index(drop=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, shifted by the row maximum so it cannot overflow."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)

    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def content_positions(input_ids: list[int], tokenizer) -> list[int]:
    """Positions in `input_ids` that are not special tokens."""
    mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

    return [position for position, is_special in enumerate(mask) if not is_special]


def align_offsets(
    positions: list[int],
    token_offsets: list[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    """Map each `input_ids` position onto the character offset of its content token."""
    if len(positions) != len(token_offsets):
        raise ValueError(
            "Content positions and token offsets differ in length: "
            f"positions={len(positions)}, offsets={len(token_offsets)}"
        )

    return {
        int(position): (int(start), int(end))
        for position, (start, end) in zip(positions, token_offsets)
    }


def expand_to_word_extent(
    token_offsets: list[tuple[int, int]],
    word_ids: list[int],
) -> list[tuple[int, int]]:
    """
    Grow each token's offset to cover its whole word.

    Only a word's first subword carries a label, so its own offset ends mid-word
    — which would truncate any entity whose final word splits into several
    subwords. `word_ids` ties the first subword back to its continuations.
    """
    word_end: dict[int, int] = {}

    for word_id, (_, end) in zip(word_ids, token_offsets):
        word_end[word_id] = max(word_end.get(word_id, end), end)

    return [(start, word_end[word_id]) for (start, _), word_id in zip(token_offsets, word_ids)]


def strip_bio_prefix(label: str) -> str:
    """`B-DISEASE` and `I-DISEASE` both become `DISEASE`."""
    if label == "O" or "-" not in label:
        return label

    return label.split("-", maxsplit=1)[1]


def entity_tags(id2label: dict) -> list[str]:
    """The entity types a label vocabulary covers, without BIO prefixes."""
    return sorted(
        {strip_bio_prefix(str(label)) for label in id2label.values() if str(label) != "O"}
    )


def ensure_int_list(value: Any) -> list[int]:
    """Coerce one cell to a list of ints, parsing the string form parquet can yield."""
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [int(item) for item in value]


def ensure_offsets(value: Any) -> list[tuple[int, int]]:
    """Coerce one cell to a list of `(start, end)` pairs."""
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if hasattr(value, "tolist"):
        value = value.tolist()

    return [(int(start), int(end)) for start, end in value]


def _aligned_offsets(row, tokenizer) -> dict[int, tuple[int, int]]:
    offsets = expand_to_word_extent(
        ensure_offsets(row.token_offsets), ensure_int_list(row.word_ids)
    )

    return align_offsets(content_positions(ensure_int_list(row.input_ids), tokenizer), offsets)


def _require_columns(rows: pd.DataFrame) -> None:
    missing = sorted(set(ROW_COLUMNS) - set(rows.columns))

    if missing:
        raise ValueError(f"Missing required columns for span reconstruction: {missing}")


def _normalize_id2label(id2label: dict) -> dict[int, str]:
    return {int(index): str(label) for index, label in id2label.items()}
