"""Sentence segmentation and whole-document tokenization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pysbd

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def split_into_sentences(text: str, language: str) -> list[dict]:
    """
    Split a document into sentence spans, in document order.

    Spans are contiguous and exhaustive: trailing whitespace belongs to the
    sentence before it, so concatenating every span reconstructs `text` exactly.
    """
    segmenter = pysbd.Segmenter(language=language, clean=False, char_span=True)

    return [
        {"start": span.start, "end": span.end, "text": span.sent}
        for span in segmenter.segment(text)
    ]


def tokenize_document(text: str, tokenizer: PreTrainedTokenizerBase) -> list[dict]:
    """
    Tokenize a whole document with a fast tokenizer, keeping each token's char span.

    Tokenizing the document once gives every downstream step the same token
    boundaries, so window sizes and entity alignment can never disagree. No
    special tokens are added; they are inserted per window by `build_row`.

    Each token's start is trimmed past any leading whitespace, so no offset
    begins on whitespace unless the token is whitespace throughout. Byte-level
    tokenizers attach the preceding space to the token and report a trimmed
    offset only when their post-processor is configured to, so normalizing here
    leaves windowing, sentence assignment and span decoding independent of that
    configuration.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        truncation=False,
    )

    input_ids = encoded["input_ids"]
    token_texts = tokenizer.convert_ids_to_tokens(input_ids)

    return [
        {
            "token_id": int(token_id),
            "token": str(token_text),
            "start": _trimmed_start(text, int(start), int(end)),
            "end": int(end),
            "word_id": word_id,
        }
        for token_id, token_text, (start, end), word_id in zip(
            input_ids,
            token_texts,
            encoded["offset_mapping"],
            encoded.word_ids(batch_index=0),
            strict=True,
        )
        if end > start
    ]


def sentence_token_ranges(sentences: list[dict], tokens: list[dict]) -> list[dict]:
    """
    Attach each sentence's `[token_start, token_end)` range to its span.

    A token belongs to the sentence containing its start character. Sentences tile
    the document and tokens are in order, so one forward sweep assigns each token
    to exactly one sentence.
    """
    ranges = []
    token_index = 0

    for sentence in sentences:
        token_start = token_index

        while token_index < len(tokens) and tokens[token_index]["start"] < sentence["end"]:
            token_index += 1

        ranges.append({**sentence, "token_start": token_start, "token_end": token_index})

    return ranges


def merge_sentences_crossing_entities(
    sentences: list[dict],
    entities: list[dict],
    text: str,
) -> list[dict]:
    """
    Fuse consecutive sentences whenever an entity span crosses their boundary.

    Sentence boundaries are a heuristic; entity spans are ground truth. Merging
    here means no windowing strategy built on these spans can cut an entity.
    """
    merged: list[dict] = []

    for sentence in sentences:
        crosses = merged and any(
            entity["start"] < sentence["start"] < entity["end"] for entity in entities
        )

        if crosses:
            start, end = merged[-1]["start"], sentence["end"]
            merged[-1] = {"start": start, "end": end, "text": text[start:end]}
        else:
            merged.append(sentence)

    return merged


def merge_short_sentences(
    sentence_ranges: list[dict],
    min_sentence_tokens: int,
    text: str,
) -> list[dict]:
    """
    Merge sentences shorter than `min_sentence_tokens` into a neighbour.

    One left-to-right pass accumulates consecutive short sentences until the
    buffer reaches the threshold, cascading through any run of them. A short
    buffer left at the end merges backward into the previous sentence instead, or
    is kept as-is when it is the whole document.
    """
    merged: list[dict] = []
    buffer: dict | None = None

    for sentence in sentence_ranges:
        buffer = sentence if buffer is None else merge_ranges(buffer, sentence, text)

        if buffer["token_end"] - buffer["token_start"] >= min_sentence_tokens:
            merged.append(buffer)
            buffer = None

    if buffer is not None:
        if merged:
            merged[-1] = merge_ranges(merged[-1], buffer, text)
        else:
            merged.append(buffer)

    return merged


def merge_ranges(first: dict, second: dict, text: str) -> dict:
    """Combine two adjacent sentence ranges into one spanning both."""
    start, end = first["start"], second["end"]

    return {
        "start": start,
        "end": end,
        "text": text[start:end],
        "token_start": first["token_start"],
        "token_end": second["token_end"],
    }


def _trimmed_start(text: str, start: int, end: int) -> int:
    trimmed = start

    while trimmed < end and text[trimmed].isspace():
        trimmed += 1

    return start if trimmed == end else trimmed
