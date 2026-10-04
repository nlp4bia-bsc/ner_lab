"""The `nel.link_entities` task: one call from a span table to a linked one."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.provenance import file_sha256, write_manifest
from lab.nel.linking.methods import (
    ENCODER_METHODS,
    METHODS,
    build_candidate_generator,
    build_reranker,
    resolve_index_dir,
)
from lab.nel.linking.scoring import score_linking
from lab.nel.linking.tables import (
    completed_frame,
    gazetteer_entries,
    linked_frame,
    mentions_from_spans,
    read_gazetteer,
    read_spans,
)
from lab.nel.pipeline import EntityLinkingPipeline

PREDICTIONS_FILENAME = "predictions.tsv"
METRICS_FILENAME = "linking_metrics.json"
LINKING_MANIFEST_FILENAME = "linking_manifest.json"


@dataclass(frozen=True)
class LinkingResult:
    """What one linking run produced: the linked span table, its scores, and where they landed."""

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
    index_dir: str | Path | None = None,
    save_index: bool = True,
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

    The encoder methods keep the gazetteer's embeddings in `index_dir`,
    `<output_dir>/gazetteer_index` by default: an index already there is loaded
    and checked against this run, otherwise the gazetteer is encoded and, unless
    `save_index=False`, saved there.
    """
    _validate(method, base_model, top_k, k_values, index_dir)
    method_kwargs = method_kwargs or {}
    reranker_kwargs = reranker_kwargs or {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = resolve_index_dir(method, output_dir, index_dir)

    spans_df = read_spans(spans)
    gazetteer_df = read_gazetteer(gazetteer)
    entries = gazetteer_entries(gazetteer_df)
    mentions = mentions_from_spans(spans_df)
    uncoded = [mention.code is None for mention in mentions]
    to_link = (
        [mention for mention, is_uncoded in zip(mentions, uncoded) if is_uncoded]
        if keep_gold
        else mentions
    )

    generator, index_source = build_candidate_generator(
        method=method,
        entries=entries,
        gazetteer_df=gazetteer_df,
        base_model=base_model,
        top_k=top_k,
        method_kwargs=method_kwargs,
        index_dir=index_dir,
        save_index=save_index,
    )
    pipeline = EntityLinkingPipeline(
        generator,
        reranker=build_reranker(reranker, reranker_kwargs),
        top_k_candidates=top_k,
    )
    linked = pipeline.fit().link_mentions(to_link) if to_link else []

    if keep_gold:
        result_df = completed_frame(spans_df, uncoded, linked)
        metrics = None
    else:
        result_df = linked_frame(spans_df, linked)
        metrics = score_linking(result_df, linked, k_values, hierarchy)

    predictions_path = output_dir / PREDICTIONS_FILENAME
    result_df.to_csv(predictions_path, sep="\t", index=False)
    paths = {"predictions": predictions_path}

    if metrics is not None:
        run_method = method if reranker is None else f"{method}+cross_encoder"
        metrics = {"method": run_method, **metrics}
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
        "method_kwargs": method_kwargs,
        "reranker": None if reranker is None else str(Path(reranker).resolve()),
        "reranker_kwargs": reranker_kwargs,
        "index_dir": str(index_dir) if index_source in ("loaded", "built") else None,
        "index": index_source,
        "top_k": int(top_k),
        "k_values": [int(k) for k in k_values],
        "hierarchy": None if hierarchy is None else str(Path(hierarchy).resolve()),
        "output_dir": str(output_dir),
        "scored_against_gold": metrics is not None,
    }
    paths["linking_manifest"] = write_manifest(manifest, output_dir / LINKING_MANIFEST_FILENAME)

    return LinkingResult(
        output_dir=output_dir,
        spans=result_df,
        metrics=metrics,
        manifest=manifest,
        paths=paths,
    )


def _validate(
    method: str,
    base_model: str | None,
    top_k: int,
    k_values: Sequence[int],
    index_dir: str | Path | None = None,
) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Available methods: {', '.join(METHODS)}.")

    if method in ENCODER_METHODS and base_model is None:
        raise ValueError(f"method={method!r} requires base_model.")

    if method not in ENCODER_METHODS and base_model is not None:
        raise ValueError(f"base_model only applies to {', '.join(ENCODER_METHODS)}.")

    if method not in ENCODER_METHODS and index_dir is not None:
        raise ValueError(f"index_dir only applies to {', '.join(ENCODER_METHODS)}.")

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    if not k_values or any(int(k) < 1 or int(k) > top_k for k in k_values):
        raise ValueError(f"k_values must lie between 1 and top_k={top_k}, got {list(k_values)}.")


def _source_path(source: pd.DataFrame | str | Path) -> str | None:
    return None if isinstance(source, pd.DataFrame) else str(Path(source).resolve())


def _source_sha256(source: pd.DataFrame | str | Path) -> str | None:
    return None if isinstance(source, pd.DataFrame) else file_sha256(source)
