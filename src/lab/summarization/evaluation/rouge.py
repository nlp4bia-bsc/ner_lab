"""ROUGE-1, ROUGE-2 and ROUGE-Lsum F1 of generated summaries against their references."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import pandas as pd
from rouge_score import rouge_scorer

ROUGE_TYPES = ("rouge1", "rouge2", "rougeLsum")
ROUGE_TOKENIZER = "unicode-nfkc-v1"
WORD_PATTERN = re.compile(r"\w+")


class UnicodeTokenizer:
    """Lower-cased Unicode word tokens after NFKC, the tokenizer `rouge_score` is given."""

    def tokenize(self, text: str) -> list[str]:
        """The `\\w+` runs of `text` after NFKC normalization and lower-casing."""
        return WORD_PATTERN.findall(unicodedata.normalize("NFKC", text).lower())


def score_rouge(generated: Sequence[str], references: Sequence[str]) -> pd.DataFrame:
    """
    ROUGE-1, ROUGE-2 and ROUGE-Lsum F1 for each generated summary, one row per pair.

    Tokens are `UnicodeTokenizer`'s, recorded as `ROUGE_TOKENIZER`: NFKC, lower-case,
    then `\\w+`, so letters in any alphabet count and a composed and a decomposed
    accent tokenize alike. `rouge_score`'s default tokenizer deletes everything
    outside `[a-z0-9]` instead, which scored Greek and Russian on their digits
    and split accented words; MultiClinSum2's official ROUGE was computed with it
    and matches this one for English only. No stemming. ROUGE-Lsum splits on
    newlines before the longest common subsequence, which suits multi-sentence
    summaries better than ROUGE-L. Scores are rounded to six decimals.
    """
    if len(generated) != len(references):
        raise ValueError(
            f"generated has {len(generated)} summaries and references {len(references)}; "
            "they are scored pairwise."
        )

    scorer = rouge_scorer.RougeScorer(
        list(ROUGE_TYPES), use_stemmer=False, tokenizer=UnicodeTokenizer()
    )
    rows = []

    for prediction, target in zip(generated, references):
        scores = scorer.score(target=target, prediction=prediction)
        rows.append({name: round(scores[name].fmeasure, 6) for name in ROUGE_TYPES})

    return pd.DataFrame(rows, columns=list(ROUGE_TYPES))
