"""Linking a span table to an ontology: the `nel.link_entities` task."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from lab.core.provenance import file_sha256, write_manifest
from lab.core.spans import SPAN_COLUMNS
from lab.nel.evaluation import (
    load_snomed_graph_pickle,
    ontology_summary_from_codes,
    retrieval_metrics_from_codes,
)
from lab.nel.io import read_table
from lab.nel.matching import MATCHER_REGISTRY, build_matcher
from lab.nel.pipeline import EntityLinkingPipeline
from lab.nel.retrieval import DenseRetriever, FaissBiEncoder, HerbertFaissBiEncoder, MatrixBiEncoder
from lab.nel.schemas import GazetteerEntry, LinkedEntity, MatchCandidate, MentionAnnotation

GOLD_COLUMN = "gold_code"
LINKED_COLUMNS = ["code", "code_term", "code_score", "candidates_json"]
SOURCE_COLUMN = "code_source"
GAZETTEER_COLUMNS = ["term", "code"]

PREDICTIONS_FILENAME = "predictions.tsv"
METRICS_FILENAME = "linking_metrics.json"
LINKING_MANIFEST_FILENAME = "linking_manifest.json"

RETRIEVERS = {"matrix": MatrixBiEncoder, "faiss": FaissBiEncoder}
ENCODER_METHODS = ("transformer_faiss", "dense")
METHODS = (*sorted(MATCHER_REGISTRY), *sorted(RETRIEVERS), *ENCODER_METHODS)


@dataclass(frozen=True)
class LinkingResult:
    output_dir: Path
    spans: pd.DataFrame
    metrics: dict[str, Any] | None
    manifest: dict[str, Any]
    paths: dict[str, Path]

    def __repr__(self) -> str:
        line = f"{self.manifest['n_linked']} of {self.manifest['n_mentions']} mentions linked"

        if self.metrics is not None:
            line += f", recall@1 {self.metrics.get('recall@1', 0.0):.4f}"

        return f"LinkingResult({line})"


def link_entities(
    spans: pd.DataFrame | str | Path,
    gazetteer: pd.DataFrame | str | Path,
    output_dir: str | Path,
    method: str = "matrix",
    base_model: str | None = None,
    method_kwargs: dict[str, Any] | None = None,
    reranker: str | Path | None = None,
    reranker_kwargs: dict[str, Any] | None = None,
    top_k: int = 25,
    k_values: Sequence[int] = (1, 5, 25),
    hierarchy: str | Path | None = None,
    keep_gold: bool = False,
) -> LinkingResult:
    """
    Link every span to a gazetteer code and write the result as a span table.

    `spans` is a span table (frame, TSV or parquet) with the canonical columns;
    a `code` column on it is taken as gold, kept as `gold_code`, and scored.
    `gazetteer` has `term` and `code` columns, plus `label` when candidates
    should be restricted to the mention's entity type.

    `method` names a lexical matcher, a sparse retriever (`matrix`, `faiss`) or
    an encoder retriever (`transformer_faiss`, `dense`); the encoder ones need
    `base_model`. `method_kwargs` reach the method's constructor. A `reranker`
    is a cross-encoder model path applied to the `top_k` candidates.

    Appends `code`, `code_term`, `code_score` and `candidates_json` to the input
    columns, so the result is still a span table.

    `keep_gold=True` completes instead of evaluating: spans with a gold code
    keep it as their `code`, only the others are linked, `code_source` says
    `gold` or `predicted`, and nothing is scored.
    """
    _validate(method, base_model, top_k, k_values)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    span_frame = read_spans(spans)
    gazetteer_frame = read_gazetteer(gazetteer)
    entries = gazetteer_entries(gazetteer_frame)
    mentions = mentions_from_spans(span_frame)
    uncoded = [mention.code is None for mention in mentions]
    to_link = [mention for mention, keep in zip(mentions, uncoded) if keep] if keep_gold else mentions

    generator = build_candidate_generator(
        method, entries, gazetteer_frame, base_model, top_k, method_kwargs or {}
    )
    reranker_model = build_reranker(reranker, reranker_kwargs or {})
    pipeline = EntityLinkingPipeline(generator, reranker=reranker_model, top_k_candidates=top_k)
    linked = pipeline.fit().link_mentions(to_link) if to_link else []

    if keep_gold:
        result = completed_frame(span_frame, uncoded, linked)
        metrics = None
    else:
        result = linked_frame(span_frame, linked)
        metrics = score_linking(result, linked, k_values, hierarchy)

    paths = {"predictions": write_linked(result, output_dir / PREDICTIONS_FILENAME)}

    if metrics is not None:
        metrics = {"method": run_method(method, reranker), **metrics}
        paths["metrics"] = write_manifest(metrics, output_dir / METRICS_FILENAME)

    manifest = {
        "spans": _source_path(spans),
        "spans_sha256": _source_sha256(spans),
        "gazetteer": _source_path(gazetteer),
        "gazetteer_sha256": _source_sha256(gazetteer),
        "n_gazetteer_entries": int(len(entries)),
        "n_mentions": int(len(mentions)),
        "n_linked": sum(entity.predicted_code is not None for entity in linked),
        "keep_gold": keep_gold,
        "n_kept_gold": int(len(mentions) - len(to_link)),
        "method": method,
        "base_model": base_model,
        "method_kwargs": method_kwargs or {},
        "reranker": None if reranker is None else str(Path(reranker).resolve()),
        "reranker_kwargs": reranker_kwargs or {},
        "top_k": int(top_k),
        "k_values": [int(k) for k in k_values],
        "hierarchy": None if hierarchy is None else str(Path(hierarchy).resolve()),
        "output_dir": str(output_dir),
        "scored_against_gold": metrics is not None,
    }
    paths["linking_manifest"] = write_manifest(manifest, output_dir / LINKING_MANIFEST_FILENAME)

    return LinkingResult(
        output_dir=output_dir, spans=result, metrics=metrics, manifest=manifest, paths=paths
    )


def read_spans(spans: pd.DataFrame | str | Path) -> pd.DataFrame:
    """
    Read a span table, taking any `code` column on it as gold.

    A table that was linked before carries `gold_code` already; its linking
    columns are dropped so they can be written afresh.
    """
    frame = spans.copy() if isinstance(spans, pd.DataFrame) else read_table(spans)
    missing = [column for column in SPAN_COLUMNS if column not in frame.columns]

    if missing:
        raise ValueError(f"Span table is missing columns {missing}. Required: {SPAN_COLUMNS}")

    if GOLD_COLUMN in frame.columns:
        linked_columns = [*LINKED_COLUMNS, SOURCE_COLUMN]
        frame = frame.drop(columns=[column for column in linked_columns if column in frame.columns])
    elif "code" in frame.columns:
        frame = frame.rename(columns={"code": GOLD_COLUMN})

    return frame.reset_index(drop=True)


def read_gazetteer(gazetteer: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Read a gazetteer table with `term` and `code` columns."""
    frame = gazetteer.copy() if isinstance(gazetteer, pd.DataFrame) else read_table(gazetteer)
    missing = [column for column in GAZETTEER_COLUMNS if column not in frame.columns]

    if missing:
        raise ValueError(f"Gazetteer is missing columns {missing}. Required: {GAZETTEER_COLUMNS}")

    frame = frame.dropna(subset=GAZETTEER_COLUMNS).reset_index(drop=True)
    frame["term"] = frame["term"].astype(str)
    frame["code"] = frame["code"].astype(str)

    if frame.empty:
        raise ValueError("Gazetteer contains no term-code pairs.")

    return frame


def gazetteer_entries(gazetteer: pd.DataFrame) -> list[GazetteerEntry]:
    """Turn a gazetteer table into the records the matchers and retrievers index."""
    known = {"term", "code", "label", "semantic_tag", "sem_tag"}
    entries: list[GazetteerEntry] = []

    for row in gazetteer.to_dict("records"):
        entries.append(
            GazetteerEntry(
                term=str(row["term"]),
                code=str(row["code"]),
                label=_optional_str(row.get("label")),
                semantic_tag=_optional_str(row.get("semantic_tag", row.get("sem_tag"))),
                metadata={key: value for key, value in row.items() if key not in known},
            )
        )

    return entries


def mentions_from_spans(spans: pd.DataFrame) -> list[MentionAnnotation]:
    """Turn span rows into the mention records the pipeline links."""
    gold = [_optional_str(value) for value in spans[GOLD_COLUMN]] if GOLD_COLUMN in spans.columns else None

    return [
        MentionAnnotation(
            filename=str(row.filename),
            label=_optional_str(row.label),
            start_span=int(row.start_span),
            end_span=int(row.end_span),
            text=str(row.text),
            code=None if gold is None else gold[index],
        )
        for index, row in enumerate(spans.itertuples(index=False))
    ]


def build_candidate_generator(
    method: str,
    entries: list[GazetteerEntry],
    gazetteer: pd.DataFrame,
    base_model: str | None,
    top_k: int,
    kwargs: dict[str, Any],
) -> Any:
    """Construct and index the candidate generator a method name refers to."""
    if method in MATCHER_REGISTRY:
        return build_matcher(method, gazetteer=entries, top_k=top_k, **kwargs)

    if method in RETRIEVERS:
        retriever = RETRIEVERS[method](top_k=top_k, **kwargs)
        retriever.build_index(entries)

        return retriever

    vocabulary = gazetteer[GAZETTEER_COLUMNS]

    if method == "transformer_faiss":
        retriever = HerbertFaissBiEncoder(base_model, **kwargs)
        retriever.fit_faiss(vocab=vocabulary)

        return EncoderRetriever(retriever, retriever.method, top_k)

    retriever = DenseRetriever(vocabulary, base_model, **kwargs)

    return EncoderRetriever(retriever, "dense", top_k)


class EncoderRetriever:
    """Gives the text-in, lists-out encoder retrievers the `search` the pipeline expects."""

    def __init__(self, retriever: Any, method: str, top_k: int) -> None:
        self.retriever = retriever
        self.method = method
        self.top_k = top_k

    def search(self, mentions: list[MentionAnnotation]) -> list[list[MatchCandidate]]:
        texts = [mention.text for mention in mentions]

        if hasattr(self.retriever, "get_candidates"):
            terms, codes, scores = self.retriever.get_candidates(texts, k=self.top_k)
        else:
            hits = self.retriever.retrieve_top_k(texts, k=self.top_k)
            terms = [hit["terms"] for hit in hits]
            codes = [hit["codes"] for hit in hits]
            scores = [hit["similarity"] for hit in hits]

        return [
            [
                MatchCandidate(
                    mention_id=None,
                    filename=mention.filename,
                    text=mention.text,
                    label=mention.label,
                    code=str(code),
                    candidate_term=str(term),
                    score=float(score),
                    method=self.method,
                    rank=rank,
                )
                for rank, (code, term, score) in enumerate(
                    zip(mention_codes, mention_terms, mention_scores), 1
                )
            ]
            for mention, mention_codes, mention_terms, mention_scores in zip(
                mentions, codes, terms, scores
            )
        ]


def build_reranker(reranker: str | Path | None, kwargs: dict[str, Any]) -> Any | None:
    """Load the cross-encoder reranker a model path refers to."""
    if reranker is None:
        return None

    from lab.nel.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(str(reranker), **kwargs)


def linked_frame(spans: pd.DataFrame, linked: list[LinkedEntity]) -> pd.DataFrame:
    """Append the linking columns to the span rows they were linked from."""
    result = spans.copy()
    result["code"] = [entity.predicted_code for entity in linked]
    result["code_term"] = [entity.predicted_term for entity in linked]
    result["code_score"] = [entity.score for entity in linked]
    result["candidates_json"] = [
        json.dumps([candidate_record(candidate) for candidate in entity.candidates], ensure_ascii=False)
        for entity in linked
    ]

    return result


def completed_frame(
    spans: pd.DataFrame,
    uncoded: list[bool],
    linked: list[LinkedEntity],
) -> pd.DataFrame:
    """Gold codes where the spans carry them, the linking columns everywhere else."""
    result = spans.copy()
    result[LINKED_COLUMNS] = None

    if linked:
        result.loc[uncoded, LINKED_COLUMNS] = linked_frame(spans.loc[uncoded], linked)[LINKED_COLUMNS]

    if GOLD_COLUMN in spans.columns:
        coded = [not keep for keep in uncoded]
        result.loc[coded, "code"] = spans.loc[coded, GOLD_COLUMN].map(_optional_str)

    result[SOURCE_COLUMN] = ["predicted" if keep else "gold" for keep in uncoded]

    return result


def candidate_record(candidate: MatchCandidate) -> dict[str, Any]:
    """One candidate as plain data, without the mention fields the row already carries."""
    return {
        "code": candidate.code,
        "term": candidate.candidate_term,
        "score": candidate.score,
        "method": candidate.method,
        "rank": candidate.rank,
        "metadata": candidate.metadata,
    }


def score_linking(
    result: pd.DataFrame,
    linked: list[LinkedEntity],
    k_values: Sequence[int],
    hierarchy: str | Path | None = None,
) -> dict[str, Any] | None:
    """Recall@k, MRR and coverage over the rows that carry a gold code, or None without any."""
    if GOLD_COLUMN not in result.columns:
        return None

    gold = result[GOLD_COLUMN].map(_optional_str)
    valid = gold.notna()

    if not valid.any():
        return None

    gold_codes = gold[valid].tolist()
    candidate_codes = [
        [candidate.code for candidate in entity.candidates]
        for entity, keep in zip(linked, valid)
        if keep
    ]
    metrics: dict[str, Any] = retrieval_metrics_from_codes(gold_codes, candidate_codes, list(k_values))

    if hierarchy is not None:
        graph, undirected = load_snomed_graph_pickle(hierarchy)
        metrics.update(ontology_summary_from_codes(gold_codes, candidate_codes, graph, undirected))

    metrics["n_input_rows"] = int(len(result))
    metrics["n_evaluated_rows"] = int(valid.sum())
    metrics["n_rows_without_gold"] = int((~valid).sum())

    return metrics


def write_linked(spans: pd.DataFrame, path: str | Path) -> Path:
    """Write a linked span table as TSV."""
    path = Path(path)
    spans.to_csv(path, sep="\t", index=False)

    return path


def run_method(method: str, reranker: str | Path | None) -> str:
    return f"{method}+cross_encoder" if reranker is not None else method


def _validate(method: str, base_model: str | None, top_k: int, k_values: Sequence[int]) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Available methods: {', '.join(METHODS)}.")

    if method in ENCODER_METHODS and base_model is None:
        raise ValueError(f"method={method!r} requires base_model.")

    if method not in ENCODER_METHODS and base_model is not None:
        raise ValueError(f"base_model only applies to {', '.join(ENCODER_METHODS)}.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not k_values or any(int(k) < 1 or int(k) > top_k for k in k_values):
        raise ValueError(f"k_values must lie between 1 and top_k={top_k}, got {list(k_values)}.")


def _optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()

    return text or None


def _source_path(source: pd.DataFrame | str | Path) -> str | None:
    return None if isinstance(source, pd.DataFrame) else str(Path(source).resolve())


def _source_sha256(source: pd.DataFrame | str | Path) -> str | None:
    return None if isinstance(source, pd.DataFrame) else file_sha256(source)
