"""Turning mention-candidate pairs and anchor-positive-negative triplets into training examples."""

from __future__ import annotations

from typing import Any

from sentence_transformers import InputExample


def pairs_to_labeled_texts(
    pairs: Any,
    mention_col: str = "mention",
    candidate_col: str = "candidate",
    label_col: str = "label",
) -> list[tuple[str, str, float]]:
    """
    `(mention, candidate, label)` tuples from pair records, the binary cross-encoder objective.

    `pairs` is a DataFrame or an iterable of dicts or objects; a missing label is 0.0,
    and a pair with an empty mention or candidate is dropped.
    """
    labeled: list[tuple[str, str, float]] = []

    for row in _records(pairs):
        mention = _text(row, mention_col)
        candidate = _text(row, candidate_col)
        label = float(_field(row, label_col, 0.0))

        if mention and candidate:
            labeled.append((mention, candidate, label))

    return labeled


def triplets_to_labeled_texts(
    triplets: Any,
    anchor_col: str = "anchor",
    positive_col: str = "positive",
    negative_col: str = "negative",
) -> list[tuple[str, str, float]]:
    """
    Each triplet as a `1.0` anchor-positive pair and a `0.0` anchor-negative pair.

    A side left empty gives no pair.
    """
    labeled: list[tuple[str, str, float]] = []

    for row in _records(triplets):
        anchor = _text(row, anchor_col)
        positive = _text(row, positive_col)
        negative = _text(row, negative_col)

        if anchor and positive:
            labeled.append((anchor, positive, 1.0))

        if anchor and negative:
            labeled.append((anchor, negative, 0.0))

    return labeled


def triplets_to_texts(
    triplets: Any,
    anchor_col: str = "anchor",
    positive_col: str = "positive",
    negative_col: str = "negative",
) -> list[tuple[str, str, str]]:
    """`(anchor, positive, negative)` tuples from triplet records with all three sides present."""
    texts = (
        (_text(row, anchor_col), _text(row, positive_col), _text(row, negative_col))
        for row in _records(triplets)
    )

    return [triplet for triplet in texts if all(triplet)]


def triplets_to_knowledge_graph_labeled_texts(
    triplets: Any,
    anchor_col: str = "anchor",
    positive_col: str = "positive",
    negative_col: str = "negative",
) -> list[tuple[str, str, float]]:
    """
    Binary examples the way the KnowledgeGraph recipe builds them from triplets.

    Each distinct `(anchor, positive)` pair once with label 1.0, then each distinct
    `(anchor, negative)` pair once with label 0.0, both sorted — so a positive is
    not repeated once per negative. A negative equal to its positive is dropped.
    """
    positive_pairs: set[tuple[str, str]] = set()
    negative_pairs: set[tuple[str, str]] = set()

    for row in _records(triplets):
        anchor = _text(row, anchor_col)
        positive = _text(row, positive_col)
        negative = _text(row, negative_col)

        if anchor and positive:
            positive_pairs.add((anchor, positive))

        if anchor and negative and negative != positive:
            negative_pairs.add((anchor, negative))

    return [
        *((anchor, positive, 1.0) for anchor, positive in sorted(positive_pairs)),
        *((anchor, negative, 0.0) for anchor, negative in sorted(negative_pairs)),
    ]


def prepare_triplets(triplets: Any) -> list[InputExample]:
    """
    Triplets as binary `InputExample`s: distinct positives labelled 1.0, distinct negatives 0.0.

    A DataFrame keeps its first-occurrence order and its pairs as they are; other
    records go through `triplets_to_knowledge_graph_labeled_texts`.
    """
    if not hasattr(triplets, "loc"):
        return [
            InputExample(texts=[anchor, candidate], label=float(label))
            for anchor, candidate, label in triplets_to_knowledge_graph_labeled_texts(triplets)
        ]

    positives = triplets[["anchor", "positive"]].drop_duplicates().reset_index(drop=True)
    negatives = triplets[["anchor", "negative"]].drop_duplicates().reset_index(drop=True)

    return [
        *(
            InputExample(texts=[str(row["anchor"]), str(row["positive"])], label=1.0)
            for _, row in positives.iterrows()
        ),
        *(
            InputExample(texts=[str(row["anchor"]), str(row["negative"])], label=0.0)
            for _, row in negatives.iterrows()
        ),
    ]


def transform_triplets_rankingeval(examples: list[Any]) -> list[dict[str, Any]]:
    """Binary examples grouped by query, as the samples a reranking evaluator takes."""
    grouped: dict[str, dict[str, Any]] = {}

    for example in examples:
        query = example.texts[0]
        sample = grouped.setdefault(query, {"query": query, "positive": set(), "negative": set()})
        sample["positive" if int(example.label) == 1 else "negative"].add(example.texts[1])

    return [
        {"query": query, "positive": list(sample["positive"]), "negative": list(sample["negative"])}
        for query, sample in grouped.items()
    ]


def _records(table: Any) -> Any:
    return table.to_dict("records") if hasattr(table, "to_dict") else table


def _field(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)

    return getattr(record, key, default)


def _text(record: Any, key: str) -> str:
    return str(_field(record, key, "")).strip()
