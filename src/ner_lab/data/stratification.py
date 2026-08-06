"""Greedy entity-stratified assignment of documents to partitions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_RANDOM_STATE = 20260701


def parse_document_label_counts(documents_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return the sorted label vocabulary and a per-document label count matrix."""
    counters: list[Counter[str]] = []
    labels_seen: set[str] = set()

    for row in documents_df[["doc_id", "entities_json", "n_entities"]].itertuples(index=False):
        try:
            entities = json.loads(str(row.entities_json))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid entities_json for document {row.doc_id!r}.") from error

        if not isinstance(entities, list):
            raise ValueError(f"entities_json must be a JSON list for {row.doc_id!r}.")

        counter: Counter[str] = Counter()

        for entity in entities:
            if not isinstance(entity, dict) or entity.get("label") is None:
                raise ValueError(f"Invalid entity/label in document {row.doc_id!r}.")

            counter[str(entity["label"])] += 1

        parsed = sum(counter.values())

        if parsed != int(row.n_entities):
            raise ValueError(
                f"n_entities mismatch for {row.doc_id!r}: stored={row.n_entities}, parsed={parsed}."
            )

        counters.append(counter)
        labels_seen.update(counter)

    labels = sorted(labels_seen)
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(documents_df), len(labels)), dtype=np.int32)

    for document_index, counter in enumerate(counters):
        for label, count in counter.items():
            matrix[document_index, label_to_index[label]] = count

    return labels, matrix


def normalize_ratios(ratios: Sequence[float]) -> list[float]:
    """Validate partition ratios and rescale them to sum to 1."""
    ratios = list(ratios)

    if len(ratios) < 2:
        raise ValueError("ratios must contain at least 2 partitions.")

    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All ratios must be positive.")

    total = float(sum(ratios))

    return [float(ratio) / total for ratio in ratios]


def stratify_documents(
    label_counts: np.ndarray,
    ratios: Sequence[float],
    random_state: int = DEFAULT_RANDOM_STATE,
    n_restarts: int = 64,
) -> tuple[np.ndarray, float]:
    """
    Assign each document a partition index, keeping the best of `n_restarts` attempts.

    Returns the assignment vector and its objective value, lower being better.
    """
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1.")

    normalized_ratios = normalize_ratios(ratios)

    if len(label_counts) < len(normalized_ratios):
        raise ValueError("Number of partitions cannot exceed the number of documents.")

    best_assignments: np.ndarray | None = None
    best_objective = np.inf

    for restart_index in range(n_restarts):
        assignments, objective = _assign_once(
            label_counts=label_counts,
            ratios=normalized_ratios,
            rng=np.random.default_rng(random_state + restart_index),
        )

        if objective < best_objective:
            best_assignments = assignments.copy()
            best_objective = objective

    if best_assignments is None:
        raise RuntimeError("Could not create partition assignments.")

    return best_assignments, float(best_objective)


def _assign_once(
    label_counts: np.ndarray,
    ratios: list[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    n_documents, n_labels = label_counts.shape
    n_partitions = len(ratios)
    document_entity_counts = label_counts.sum(axis=1).astype(float)
    total_label_counts = label_counts.sum(axis=0).astype(float)
    total_entities = float(document_entity_counts.sum())

    capacities = _partition_capacities(n_documents, ratios, rng)
    target_docs = capacities.astype(float)
    target_entities = total_entities * capacities / n_documents
    target_labels = np.outer(capacities / n_documents, total_label_counts)

    # Deviations are scaled by the geometric mean of a shared global scale and each
    # partition's own target. A shared scale alone biases assignment toward the larger
    # partition under unequal ratios; each partition's own target alone overcorrects into
    # the smaller one. See docs/PLAN.md for the measurements behind this.
    shared_label_scale = np.maximum(total_label_counts / n_partitions, 1.0)
    shared_entity_scale = max(total_entities / n_partitions, 1.0)
    shared_document_scale = max(n_documents / n_partitions, 1.0)

    label_scale = np.sqrt(np.maximum(target_labels, 1.0) * shared_label_scale)
    entity_scale = np.sqrt(np.maximum(target_entities, 1.0) * shared_entity_scale)
    document_scale = np.sqrt(np.maximum(target_docs, 1.0) * shared_document_scale)

    fold_docs = np.zeros(n_partitions, dtype=np.int32)
    fold_entities = np.zeros(n_partitions, dtype=float)
    fold_labels = np.zeros((n_partitions, n_labels), dtype=float)
    assignments = np.full(n_documents, -1, dtype=np.int16)

    rarity_weights = 1.0 / np.maximum(total_label_counts, 1.0)
    rarity_score = label_counts.astype(float) @ rarity_weights
    tie_breaker = rng.random(n_documents)
    document_order = np.lexsort((tie_breaker, -document_entity_counts, -rarity_score))

    for document_index in document_order:
        label_vector = label_counts[document_index].astype(float)
        entity_count = document_entity_counts[document_index]
        candidate_folds = np.flatnonzero(fold_docs < capacities)

        if not len(candidate_folds):
            raise RuntimeError("No remaining partition capacity.")

        scores: list[float] = []

        for fold_id in candidate_folds:
            label_before = (fold_labels[fold_id] - target_labels[fold_id]) / label_scale[fold_id]
            label_after = (
                fold_labels[fold_id] + label_vector - target_labels[fold_id]
            ) / label_scale[fold_id]
            label_delta = np.square(label_after).sum() - np.square(label_before).sum()

            entity_before = (
                fold_entities[fold_id] - target_entities[fold_id]
            ) / entity_scale[fold_id]
            entity_after = (
                fold_entities[fold_id] + entity_count - target_entities[fold_id]
            ) / entity_scale[fold_id]
            entity_delta = entity_after**2 - entity_before**2

            document_before = (fold_docs[fold_id] - target_docs[fold_id]) / document_scale[fold_id]
            document_after = (
                fold_docs[fold_id] + 1 - target_docs[fold_id]
            ) / document_scale[fold_id]
            document_delta = document_after**2 - document_before**2

            scores.append(label_delta + 0.25 * entity_delta + 0.10 * document_delta)

        scores_array = np.asarray(scores)
        best_folds = candidate_folds[
            np.isclose(scores_array, scores_array.min(), rtol=0.0, atol=1e-12)
        ]
        selected_fold = int(rng.choice(best_folds))

        assignments[document_index] = selected_fold
        fold_docs[selected_fold] += 1
        fold_entities[selected_fold] += entity_count
        fold_labels[selected_fold] += label_vector

    if (assignments < 0).any():
        raise RuntimeError("Some documents were not assigned to a partition.")

    objective = float(
        np.square((fold_labels - target_labels) / label_scale).sum()
        + 0.25 * np.square((fold_entities - target_entities) / entity_scale).sum()
        + 0.10 * np.square((fold_docs - target_docs) / document_scale).sum()
    )

    return assignments, objective


def _partition_capacities(
    n_documents: int,
    ratios: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    raw_sizes = [ratio * n_documents for ratio in ratios]
    capacities = np.array([int(size) for size in raw_sizes], dtype=np.int32)
    remainder = n_documents - int(capacities.sum())

    if remainder:
        fractional_parts = np.array([raw - cap for raw, cap in zip(raw_sizes, capacities)])
        order = np.lexsort((rng.random(len(ratios)), -fractional_parts))
        capacities[order[:remainder]] += 1

    return capacities
