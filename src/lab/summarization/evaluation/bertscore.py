"""BERTScore F1 of generated summaries against their references, with a multilingual encoder."""

from __future__ import annotations

from collections.abc import Sequence

import bert_score.utils
from bert_score import BERTScorer

from lab.summarization.devices import resolve_device

DEFAULT_BERTSCORE_MODEL = "microsoft/mdeberta-v3-base"
DEFAULT_BERTSCORE_LAYERS = 12
MAX_SENTENCE_TOKENS = 512


def score_bertscore(
    generated: Sequence[str],
    references: Sequence[str],
    language: str,
    model: str = DEFAULT_BERTSCORE_MODEL,
    num_layers: int = DEFAULT_BERTSCORE_LAYERS,
    rescale_with_baseline: bool = False,
    batch_size: int = 32,
    device: str = "auto",
) -> list[float]:
    """
    BERTScore F1 for each generated summary against its reference, in one language.

    `model` is a Hub identifier or a local snapshot. `bert_score` only knows the
    layer count of the Hub names it ships a table for, so `num_layers` is always
    passed; 12 is the last layer of mdeberta-v3-base. `rescale_with_baseline`
    rescales against `bert_score`'s baseline for `language`, which exists only for
    Hub names it ships baselines for; MultiClinSum2 ran without it.

    Summaries are scored `batch_size` pairs at a time, so the encoder's memory
    stays bounded when a judge model shares the GPU, and each text is cut at 512
    tokens. Scores are clamped to [0, 1] and rounded to six decimals.
    """
    if len(generated) != len(references):
        raise ValueError(
            f"generated has {len(generated)} summaries and references {len(references)}; "
            "they are scored pairwise."
        )

    scorer = BERTScorer(
        model_type=model,
        num_layers=num_layers,
        lang=language,
        rescale_with_baseline=rescale_with_baseline,
        device=resolve_device(device),
    )
    original_sent_encode = bert_score.utils.sent_encode
    bert_score.utils.sent_encode = _sent_encode
    scores: list[float] = []

    try:
        for start in range(0, len(generated), batch_size):
            _, _, f1 = scorer.score(
                cands=list(generated[start : start + batch_size]),
                refs=list(references[start : start + batch_size]),
            )
            scores.extend(max(0.0, min(1.0, round(float(value), 6))) for value in f1)
    finally:
        bert_score.utils.sent_encode = original_sent_encode

    return scores


def _sent_encode(tokenizer, sentence: str) -> list[int]:
    # bert_score passes tokenizer.model_max_length as max_length, int(1e30) for mdeberta,
    # which overflows the Rust tokenizer.
    return tokenizer.encode(
        sentence, add_special_tokens=True, max_length=MAX_SENTENCE_TOKENS, truncation=True
    )
