"""Turning labelled token windows into model-ready rows."""

from __future__ import annotations

IGNORE_INDEX = -100


def special_token_template(tokenizer) -> tuple[list[int], list[int]]:
    """
    The special-token ids a tokenizer adds before and after a single sequence.

    Derived by encoding a probe with and without special tokens and locating the
    probe in the result, rather than asked for directly: transformers v5 removed
    `build_inputs_with_special_tokens` and the `already_has_special_tokens=False`
    path of `get_special_tokens_mask`. Probing covers CLS/SEP, BOS/EOS, EOS-only
    and no-specials layouts without per-family knowledge.
    """
    probe = tokenizer.encode("a", add_special_tokens=False)
    formatted = tokenizer.encode("a", add_special_tokens=True)

    for start in range(len(formatted) - len(probe) + 1):
        if formatted[start:start + len(probe)] == probe:
            return formatted[:start], formatted[start + len(probe):]

    raise ValueError(
        f"Could not locate probe {probe} inside {formatted}; "
        "cannot infer this tokenizer's special-token template."
    )


def compute_max_content_length(
    prefix_ids: list[int],
    suffix_ids: list[int],
    max_length: int,
) -> int:
    """
    Token budget left for content once the special-token template is accounted for.

    Takes the template rather than the tokenizer so the budget and the tokens
    `build_row` actually inserts come from one source and cannot disagree.
    """
    return max_length - len(prefix_ids) - len(suffix_ids)


def build_row(
    labelled_tokens: list[dict],
    label2id: dict,
    prefix_ids: list[int],
    suffix_ids: list[int],
) -> dict:
    """
    Turn one labelled token window into a model-ready row.

    Only a word's first subword carries its label id; continuation subwords, the
    special tokens, and any token with no `label` key (the context strategy's
    unlabelled flanks) get `IGNORE_INDEX`, so loss is never computed on them.

    `word_ids` rides alongside `token_offsets` so span reconstruction at evaluation
    time can grow a labelled position back out to its whole word — without it a
    span ending on a multi-subword word would stop at that word's first subword.
    """
    label_ids = []
    previous_word_id = None

    for token in labelled_tokens:
        is_continuation = token["word_id"] is not None and token["word_id"] == previous_word_id
        previous_word_id = token["word_id"]

        if "label" not in token or is_continuation:
            label_ids.append(IGNORE_INDEX)
        else:
            label_ids.append(label2id[token["label"]])

    input_ids = [*prefix_ids, *(token["token_id"] for token in labelled_tokens), *suffix_ids]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prefix_ids) + label_ids + [IGNORE_INDEX] * len(suffix_ids),
        "token_offsets": [(token["start"], token["end"]) for token in labelled_tokens],
        "word_ids": [token["word_id"] for token in labelled_tokens],
    }
