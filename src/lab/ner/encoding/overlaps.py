"""Resolving overlapping gold annotations, at encoding time."""

from __future__ import annotations

from typing import Literal

OverlapPolicy = Literal[
    "none",
    "error",
    "keep_longest",
    "keep_shortest",
    "merge_same_label_then_keep_longest",
]


def resolve_entities(
    doc_id: str,
    entities: list[dict],
    text: str,
    policy: OverlapPolicy,
) -> list[dict]:
    """
    Validate one document's entities and reduce overlapping spans to one each.

    `none` validates only, leaving overlaps exactly as annotated. `error` raises on
    any overlap. The rest collapse each cluster of transitively overlapping spans
    to a single representative.
    """
    entities = _validate(doc_id, text, entities)

    if policy == "none":
        return entities

    if policy == "merge_same_label_then_keep_longest":
        entities = _merge_same_label(entities, text)

    clusters = _cluster_by_overlap(entities)

    if policy == "error":
        conflict = next((cluster for cluster in clusters if len(cluster) > 1), None)

        if conflict is not None:
            raise ValueError(f"Overlapping annotations in document {doc_id!r}: {conflict}")

        return entities

    if policy in ("keep_longest", "merge_same_label_then_keep_longest"):
        keep = "longest"
    elif policy == "keep_shortest":
        keep = "shortest"
    else:
        raise ValueError(f"Unsupported annotation overlap policy: {policy!r}")

    return [_pick_representative(cluster, keep) for cluster in clusters]


def _validate(doc_id: str, text: str, entities: list[dict]) -> list[dict]:
    validated = []

    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])

        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"Invalid span in document {doc_id!r}: "
                f"start={start}, end={end}, text length={len(text)}"
            )

        mention = text[start:end]

        if str(entity["text"]) != mention:
            raise ValueError(
                f"Entity text does not match document in {doc_id!r}: "
                f"expected {entity['text']!r}, found {mention!r} at [{start}:{end}]"
            )

        validated.append(
            {**entity, "start": start, "end": end, "label": str(entity["label"]), "text": mention}
        )

    return validated


def _cluster_by_overlap(entities: list[dict]) -> list[list[dict]]:
    if not entities:
        return []

    sorted_entities = sorted(entities, key=lambda entity: (entity["start"], entity["end"]))

    clusters = [[sorted_entities[0]]]
    cluster_end = sorted_entities[0]["end"]

    for entity in sorted_entities[1:]:
        if entity["start"] < cluster_end:
            clusters[-1].append(entity)
            cluster_end = max(cluster_end, entity["end"])
        else:
            clusters.append([entity])
            cluster_end = entity["end"]

    return clusters


def _merge_same_label(entities: list[dict], text: str) -> list[dict]:
    merged = []

    for label in sorted({entity["label"] for entity in entities}):
        same_label = [entity for entity in entities if entity["label"] == label]

        for group in _cluster_by_overlap(same_label):
            if len(group) == 1:
                merged.append(group[0])
                continue

            start = min(entity["start"] for entity in group)
            end = max(entity["end"] for entity in group)
            merged.append({"start": start, "end": end, "label": label, "text": text[start:end]})

    return merged


def _pick_representative(cluster: list[dict], keep: Literal["longest", "shortest"]) -> dict:
    if len(cluster) == 1:
        return cluster[0]

    if keep == "longest":
        return max(cluster, key=lambda entity: (entity["end"] - entity["start"], -entity["start"]))

    return min(cluster, key=lambda entity: (entity["end"] - entity["start"], entity["start"]))
