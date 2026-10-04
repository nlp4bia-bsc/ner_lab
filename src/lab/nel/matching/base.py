"""The matcher every lexical method shares: score each entry, keep the best per code, rank."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from lab.nel.preprocessing import normalize_text
from lab.nel.schemas import Candidate, GazetteerEntry, Mention, labels_compatible


class LexicalMatcher:
    """
    Score a mention against every compatible gazetteer entry and rank the results.

    A subclass names its `method` and implements `_score_candidates`, yielding
    `(entry, score)` pairs; everything else — the threshold, keeping the best
    entry per code, `top_k` and the ranks — happens here. With `label_aware`, an
    entry labelled differently from the mention is never proposed.
    """

    method = "base"

    def __init__(
        self,
        gazetteer: list[GazetteerEntry],
        threshold: float = 0.0,
        top_k: int = 10,
        label_aware: bool = True,
        keep_duplicate_codes: bool = False,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self.gazetteer = gazetteer
        self.threshold = threshold
        self.top_k = top_k
        self.label_aware = label_aware
        self.keep_duplicate_codes = keep_duplicate_codes
        self.normalizer = normalizer or normalize_text

    def fit(self) -> LexicalMatcher:
        """Return the matcher; an index, where a method needs one, is built on first use."""
        return self

    def predict(self, mentions: list[Mention]) -> list[list[Candidate]]:
        """The ranked candidates of each mention."""
        return [self.score_mention(mention) for mention in mentions]

    def score_mention(self, mention: Mention) -> list[Candidate]:
        """One mention's candidates at or above the threshold, best first, ranked from 1."""
        candidates = [
            Candidate(
                mention_id=None,
                filename=mention.filename,
                text=mention.text,
                label=mention.label,
                code=entry.code,
                term=entry.term,
                score=float(score),
                method=self.method,
            )
            for entry, score in self._score_candidates(mention)
            if not score < self.threshold
        ]

        if not self.keep_duplicate_codes:
            candidates = _best_per_code(candidates)

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        candidates = candidates[: self.top_k]

        for rank, candidate in enumerate(candidates, 1):
            candidate.rank = rank

        return candidates

    def _compatible_entries(self, mention: Mention) -> Iterable[GazetteerEntry]:
        return (
            entry
            for entry in self.gazetteer
            if not self.label_aware or labels_compatible(mention.label, entry.label)
        )

    def _score_candidates(self, mention: Mention) -> Iterable[tuple[GazetteerEntry, float]]:
        raise NotImplementedError


def _best_per_code(candidates: list[Candidate]) -> list[Candidate]:
    best_by_code: dict[str | None, Candidate] = {}

    for candidate in candidates:
        current = best_by_code.get(candidate.code)

        if current is None or candidate.score > current.score:
            best_by_code[candidate.code] = candidate

    return list(best_by_code.values())
