"""Reciprocal Rank Fusion: merging several candidate rankings into one."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from lab.nel.schemas import Candidate


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Candidate]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[Candidate]:
    """
    Fuse rankings keyed by source into one, scoring each code `sum(1 / (k + rank))`.

    A code counts once per source, at its first position; a candidate without a
    `rank` takes its position in the list. Ties are broken by the code's best
    rank in any source, then by the code itself. Each fused candidate is the
    first one seen for its code, re-scored with `method="rrf"`, and its metadata
    records the sources it came from with their scores and ranks.
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    if not rankings:
        return []

    fused_scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, Candidate] = {}
    source_scores: dict[str, dict[str, float]] = defaultdict(dict)
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for source, candidates in rankings.items():
        seen: set[str] = set()

        for fallback_rank, candidate in enumerate(candidates, 1):
            if candidate.code is None or str(candidate.code) in seen:
                continue

            code = str(candidate.code)
            seen.add(code)
            rank = int(candidate.rank or fallback_rank)
            fused_scores[code] += 1.0 / (k + rank)
            source_scores[code][source] = float(candidate.score)
            source_ranks[code][source] = rank
            representatives.setdefault(code, candidate)

    ordered_codes = sorted(
        fused_scores,
        key=lambda code: (-fused_scores[code], min(source_ranks[code].values()), code),
    )

    return [
        _fused_candidate(
            representatives[code],
            score=fused_scores[code],
            rank=rank,
            sources={
                "retrieval_sources": sorted(source_ranks[code]),
                "source_scores": source_scores[code],
                "source_ranks": source_ranks[code],
            },
        )
        for rank, code in enumerate(ordered_codes[:top_k], 1)
    ]


def _fused_candidate(
    representative: Candidate, score: float, rank: int, sources: dict[str, object]
) -> Candidate:
    return Candidate(
        mention_id=representative.mention_id,
        filename=representative.filename,
        text=representative.text,
        label=representative.label,
        code=representative.code,
        term=representative.term,
        score=score,
        method="rrf",
        rank=rank,
        metadata={**representative.metadata, **sources},
    )
