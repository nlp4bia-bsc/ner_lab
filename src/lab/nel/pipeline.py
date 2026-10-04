"""Linking mentions: candidate generation, fusion across generators, optional reranking."""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from lab.nel.io import read_brat_dir, read_mentions_tsv
from lab.nel.rrf import reciprocal_rank_fusion
from lab.nel.schemas import Candidate, LinkedEntity, Mention


class EntityLinkingPipeline:
    """
    Run one or more candidate generators, then an optional reranker.

    A generator exposes `predict(mentions)` or `search(mentions)`. Several
    generators are fused with reciprocal rank fusion; a single one keeps its own
    ranking and scores.

    A reranker either exposes `rerank(mention, candidates)` returning candidates,
    or the cross-encoder form `rerank(texts, packed, k)` taking and returning
    aligned term, code and score lists.
    """

    def __init__(
        self,
        candidate_generator: Any | None = None,
        *,
        candidate_generators: Sequence[Any] | None = None,
        reranker: Any | None = None,
        top_k_candidates: int = 25,
        top_k_final: int = 1,
        rrf_k: int = 60,
    ) -> None:
        generators = list(candidate_generators or [])

        if candidate_generator is not None:
            generators.insert(0, candidate_generator)

        if not generators:
            raise ValueError("At least one candidate generator is required")

        self.candidate_generators = generators
        self.candidate_generator = generators[0]
        self.reranker = reranker
        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k

    def fit(self, gazetteer: Any | None = None) -> EntityLinkingPipeline:
        """Index each generator on `gazetteer` if it builds an index, or fit it if it fits."""
        for generator in self.candidate_generators:
            if hasattr(generator, "build_index") and gazetteer is not None:
                generator.build_index(gazetteer)
            elif hasattr(generator, "fit"):
                generator.fit()

        return self

    def link_mentions(self, mentions: list[Mention]) -> list[LinkedEntity]:
        """Link each mention to its ranked candidates and their top prediction."""
        generated = [_generate(generator, mentions) for generator in self.candidate_generators]
        linked: list[LinkedEntity] = []

        for index, mention in enumerate(mentions):
            candidates = self._candidates(generated, index)
            candidates = self._rerank(mention, candidates)[: self.top_k_candidates]
            top = candidates[0] if candidates else None

            linked.append(
                LinkedEntity(
                    mention=mention,
                    candidates=candidates,
                    predicted_code=top.code if top else None,
                    predicted_term=top.term if top else None,
                    score=top.score if top else None,
                    method=top.method if top else "none",
                )
            )

        return linked

    def link_from_tsv(self, path: str | Path) -> list[LinkedEntity]:
        """Link the mentions of a mention TSV."""
        return self.link_mentions(read_mentions_tsv(str(path)))

    def link_from_brat_dir(self, path: str | Path) -> list[LinkedEntity]:
        """Link the mentions of every BRAT `.ann` file in a directory."""
        return self.link_mentions(read_brat_dir(str(path)))

    def write_outputs(self, linked: Sequence[LinkedEntity], path: str | Path) -> Path:
        """Write one TSV row per mention: its gold code, the prediction, and every candidate."""
        rows = []

        for entity in linked:
            top = entity.candidates[0] if entity.candidates else None
            rows.append(
                {
                    "filename": entity.mention.filename,
                    "start_span": entity.mention.start_span,
                    "end_span": entity.mention.end_span,
                    "text": entity.mention.text,
                    "gold_code": entity.mention.code,
                    "predicted_code": top.code if top else None,
                    "predicted_term": top.term if top else None,
                    "score": top.score if top else None,
                    "method": top.method if top else "none",
                    "candidates_json": json.dumps(
                        [dataclasses.asdict(candidate) for candidate in entity.candidates],
                        ensure_ascii=False,
                    ),
                }
            )

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)

        return output_path

    def _candidates(self, generated: list[list[list[Candidate]]], index: int) -> list[Candidate]:
        if len(generated) == 1:
            return list(generated[0][index])[: self.top_k_candidates]

        rankings = {
            getattr(generator, "method", type(generator).__name__): generated[position][index]
            for position, generator in enumerate(self.candidate_generators)
        }

        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_k=self.top_k_candidates)

    def _rerank(self, mention: Mention, candidates: list[Candidate]) -> list[Candidate]:
        if self.reranker is None or not candidates:
            return candidates

        if "k" in inspect.signature(self.reranker.rerank).parameters:
            return self._rerank_packed(mention, candidates)

        reranked = self.reranker.rerank(mention, candidates)

        if not isinstance(reranked, list) or (reranked and not isinstance(reranked[0], Candidate)):
            raise TypeError("The reranker must return a list of Candidate objects")

        return reranked

    def _rerank_packed(self, mention: Mention, candidates: list[Candidate]) -> list[Candidate]:
        packed = [
            {
                "terms": [candidate.term or "" for candidate in candidates],
                "codes": [candidate.code or "" for candidate in candidates],
            }
        ]
        result = self.reranker.rerank([mention.text], packed, k=self.top_k_candidates)[0]
        codes = result.get("codes", [])
        score_by_code = {
            str(code): float(score) for code, score in zip(codes, result.get("similarity", []))
        }
        by_code = {
            str(candidate.code): candidate for candidate in candidates if candidate.code is not None
        }
        reranked = []

        for rank, code in enumerate(codes, 1):
            candidate = by_code[str(code)]
            candidate.score = score_by_code[str(code)]
            candidate.method = "cross_encoder"
            candidate.rank = rank
            reranked.append(candidate)

        return reranked


def _generate(generator: Any, mentions: list[Mention]) -> list[list[Candidate]]:
    if hasattr(generator, "predict"):
        return generator.predict(mentions)

    if hasattr(generator, "search"):
        return generator.search(mentions)

    raise TypeError(
        f"Candidate generator {type(generator).__name__} has neither predict nor search"
    )
