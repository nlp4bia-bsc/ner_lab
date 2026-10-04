"""The annotation table: mentions, surface forms, entity length and span relations."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from lab.core.segmentation import tokenize_document
from lab.core.stats.primitives import (
    WHITESPACE,
    entities_of,
    load_tokenizer,
    quantiles,
    share,
    write_stats,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def mention_token_length(entity: dict, starts: Sequence[int], ends: Sequence[int]) -> int:
    """
    How many tokens the entity's character span touches.

    Tokens are ordered and non-overlapping, so both boundaries are a binary
    search: the first token ending after the span starts, and the first token
    starting at or after the span ends.
    """
    first = bisect_right(ends, entity["start"])
    last = bisect_left(starts, entity["end"])

    return max(last - first, 0)


def span_relations(entities: Sequence[dict]) -> dict[str, int]:
    """
    Mentions involved in an identical, nested, or crossing span relation.

    A mention is counted once per relation type it takes part in. Spans are swept
    in start order and the inner loop stops at the first span beginning after the
    current one ends, since no later span can overlap it either.
    """
    spans = sorted((entity["start"], entity["end"], index) for index, entity in enumerate(entities))
    involved: dict[str, set[int]] = {"identical": set(), "nested": set(), "crossing": set()}

    for position, (start, end, index) in enumerate(spans):
        for offset in range(position + 1, len(spans)):
            other_start, other_end, other_index = spans[offset]

            if other_start >= end:
                break

            if (start, end) == (other_start, other_end):
                relation = "identical"
            elif other_end <= end or other_start <= start:
                relation = "nested"
            else:
                relation = "crossing"

            involved[relation].update((index, other_index))

    return {relation: len(mentions) for relation, mentions in involved.items()}


@dataclass
class AnnotationMeasurements:
    """
    Entity-layer measurements, accumulated document by document.

    `discontinuous_mentions` is always 0: a `{start, end}` span cannot express a
    gap, so the format itself rules them out.
    """

    documents: int = 0
    tokens: int = 0
    annotated_documents: int = 0
    mention_counts: list[int] = field(default_factory=list)
    mention_lengths: list[int] = field(default_factory=list)
    labels: Counter[str] = field(default_factory=Counter)
    surface_forms: Counter[str] = field(default_factory=Counter)
    relations: Counter[str] = field(default_factory=Counter)

    def add(self, text: str, entities: Sequence[dict], tokenizer: PreTrainedTokenizerBase) -> None:
        tokens = tokenize_document(text, tokenizer)

        self.documents += 1
        self.tokens += len(tokens)
        self.mention_counts.append(len(entities))

        if not entities:
            return

        self.annotated_documents += 1
        starts = [token["start"] for token in tokens]
        ends = [token["end"] for token in tokens]

        for entity in entities:
            self.labels[entity["label"]] += 1
            self.surface_forms[WHITESPACE.sub(" ", entity["text"].strip().lower())] += 1
            self.mention_lengths.append(mention_token_length(entity, starts, ends))

        self.relations.update(span_relations(entities))

    def row(self) -> dict:
        mentions = sum(self.mention_counts)
        unique_forms = len(self.surface_forms)
        singletons = sum(1 for count in self.surface_forms.values() if count == 1)
        multi_token = sum(1 for length in self.mention_lengths if length > 1)

        return {
            "mentions": mentions,
            "unique_surface_forms": unique_forms,
            **quantiles("mentions_per_document", self.mention_counts, (25, 50, 75)),
            "mentions_per_1k_tokens": (
                mentions / self.tokens * 1000 if self.tokens else float("nan")
            ),
            "entity_types": len(self.labels),
            **quantiles("entity_length_tokens", self.mention_lengths, (25, 50, 75)),
            "multi_token_pct": share(multi_token, mentions),
            "unique_surface_form_ratio": unique_forms / mentions if mentions else float("nan"),
            "singleton_surface_form_ratio": (
                singletons / unique_forms if unique_forms else float("nan")
            ),
            "documents_with_entities_pct": share(self.annotated_documents, self.documents),
            "nested_mentions": self.relations["nested"],
            "nested_pct": share(self.relations["nested"], mentions),
            "crossing_mentions": self.relations["crossing"],
            "crossing_pct": share(self.relations["crossing"], mentions),
            "identical_span_mentions": self.relations["identical"],
            "identical_span_pct": share(self.relations["identical"], mentions),
            "discontinuous_mentions": 0,
        }


def compute_annotation_stats(
    frame: pd.DataFrame,
    base_model: str,
    *,
    normalize_labels: bool = False,
    output_dir: str | Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Annotated-corpus metrics: mention counts, surface-form diversity, entity
    length, span-relation complexity and the per-entity-type breakdown, as a
    single-row DataFrame.

    The breakdown is spread into flat `mentions_<LABEL>` / `pct_<LABEL>` columns
    ordered by corpus-wide frequency, so the table round-trips through parquet and
    JSON without nesting. Pass `output_dir` to also write `annotation_stats.json` /
    `annotation_stats.parquet` there.
    """
    tokenizer = load_tokenizer(base_model)
    measurements = AnnotationMeasurements()

    if progress:
        print(f"    annotation stats: {len(frame):,} documents", flush=True)

    for text, raw in zip(frame["text"], frame["entities_json"]):
        measurements.add(text or "", entities_of(raw, normalize_labels), tokenizer)

    row = measurements.row()
    mentions = row["mentions"]

    for label, count in measurements.labels.most_common():
        row[f"mentions_{label}"] = count
        row[f"pct_{label}"] = share(count, mentions)

    result = pd.DataFrame([row])

    if output_dir is not None:
        write_stats(result, output_dir, "annotation_stats")

    return result
