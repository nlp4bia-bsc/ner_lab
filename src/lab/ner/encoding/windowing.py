"""Windowing strategies: cutting a document's tokens into model-sized pieces.

A strategy takes `(sentence_ranges, tokens, entities, text, max_content_length)` and
returns windows carrying `token_start`, `token_end` and their char span. The context
strategy additionally tags each window's labelled `core_token_start`/`core_token_end`.
Strategy-specific settings are bound by the caller, so any callable with that signature
can be used in place of these.
"""

from __future__ import annotations

from lab.core.segmentation import merge_short_sentences


def select_greedy_windows(
    sentence_ranges: list[dict],
    tokens: list[dict],
    entities: list[dict],
    text: str,
    max_content_length: int,
) -> list[dict]:
    """
    Tile a document into non-overlapping windows of whole sentences.

    Walks left to right packing as many complete sentences as fit, then starts the
    next window from the first that did not. A sentence too long on its own is cut
    by `split_oversized_sentence`.
    """
    windows = []
    cursor = 0

    while cursor < len(sentence_ranges):
        sentence = sentence_ranges[cursor]

        if sentence["token_end"] - sentence["token_start"] > max_content_length:
            windows.extend(
                split_oversized_sentence(sentence, tokens, entities, text, max_content_length)
            )
            cursor += 1
            continue

        window_last = cursor
        token_start = sentence["token_start"]
        token_end = sentence["token_end"]

        while window_last + 1 < len(sentence_ranges):
            candidate = sentence_ranges[window_last + 1]

            if candidate["token_end"] - token_start > max_content_length:
                break

            window_last += 1
            token_end = candidate["token_end"]

        windows.append(build_window(token_start, token_end, tokens, text))
        cursor = window_last + 1

    return windows


def select_context_windows(
    sentence_ranges: list[dict],
    tokens: list[dict],
    entities: list[dict],
    text: str,
    max_content_length: int,
    context_tokens: int,
    min_sentence_tokens: int = 4,
) -> list[dict]:
    """
    Build one labelled core sentence plus unlabelled context per window, FLERT-style.

    Every sentence becomes the core of its own window, flanked by up to
    `context_tokens` raw tokens each side. Short sentences are merged first. A core
    too long for the budget is split entity-safely and each piece becomes its own
    core. Context fill is trimmed in order: to the document edges, reallocating
    unused budget to the other side; by any overflow past `max_content_length`,
    taken off the trailing side first; then one token at a time if a boundary would
    land inside an entity. Unlike the greedy strategy, windows overlap by design.
    """
    sentence_ranges = merge_short_sentences(sentence_ranges, min_sentence_tokens, text)
    num_tokens = len(tokens)

    cores: list[dict] = []

    for sentence in sentence_ranges:
        if sentence["token_end"] - sentence["token_start"] > max_content_length:
            cores.extend(
                split_oversized_sentence(sentence, tokens, entities, text, max_content_length)
            )
        else:
            cores.append(sentence)

    windows = []

    for core in cores:
        core_start, core_end = core["token_start"], core["token_end"]
        available_before = core_start
        available_after = num_tokens - core_end

        target_before = min(
            context_tokens + max(0, context_tokens - available_after), available_before
        )
        target_after = min(
            context_tokens + max(0, context_tokens - available_before), available_after
        )

        remaining_budget = max_content_length - (core_end - core_start)
        overflow = max(0, target_before + target_after - remaining_budget)
        take_after = min(target_after, overflow)
        target_after -= take_after
        target_before -= overflow - take_after

        before_start = _safe_preceding_boundary(
            max(core_start - target_before, 0), core_start, tokens, entities
        )
        after_end = _safe_succeeding_boundary(
            min(core_end + target_after, num_tokens), core_end, num_tokens, tokens, entities
        )

        window = build_window(before_start, after_end, tokens, text)
        window["core_token_start"] = core_start
        window["core_token_end"] = core_end
        windows.append(window)

    return windows


def build_window(token_start: int, token_end: int, tokens: list[dict], text: str) -> dict:
    """Build one window's char span from a token index range."""
    start = tokens[token_start]["start"]
    end = tokens[token_end - 1]["end"]

    return {
        "token_start": token_start,
        "token_end": token_end,
        "start": start,
        "end": end,
        "text": text[start:end],
    }


def split_oversized_sentence(
    sentence: dict,
    tokens: list[dict],
    entities: list[dict],
    text: str,
    max_content_length: int,
) -> list[dict]:
    """
    Cut a sentence too long for one window into several, entity-safely.

    Each cut lands at the token boundary closest to but not exceeding the budget
    that does not fall inside an entity, backing off one token at a time.
    """
    windows = []
    chunk_start = sentence["token_start"]
    sentence_end = sentence["token_end"]

    while chunk_start < sentence_end:
        chunk_end = min(chunk_start + max_content_length, sentence_end)

        while chunk_end < sentence_end and any(
            entity["start"] < tokens[chunk_end]["start"] < entity["end"] for entity in entities
        ):
            chunk_end -= 1

        if chunk_end <= chunk_start:
            raise ValueError(
                "An entity is longer than max_content_length tokens; no safe cut "
                f"exists within budget starting at token {chunk_start}."
            )

        windows.append(build_window(chunk_start, chunk_end, tokens, text))
        chunk_start = chunk_end

    return windows


def _safe_preceding_boundary(
    candidate: int, limit: int, tokens: list[dict], entities: list[dict]
) -> int:
    while candidate < limit and any(
        entity["start"] < tokens[candidate]["start"] < entity["end"] for entity in entities
    ):
        candidate += 1

    return candidate


def _safe_succeeding_boundary(
    candidate: int, limit: int, num_tokens: int, tokens: list[dict], entities: list[dict]
) -> int:
    while limit < candidate < num_tokens and any(
        entity["start"] < tokens[candidate]["start"] < entity["end"] for entity in entities
    ):
        candidate -= 1

    return candidate
