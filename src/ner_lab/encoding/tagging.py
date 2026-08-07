"""IOB2 tagging of tokens against one entity type."""

from __future__ import annotations


def build_iob2_labels(tokens: list[dict], entities: list[dict], target_label: str) -> list[dict]:
    """
    Tag each token `O` / `B-<target>` / `I-<target>` for one entity type.

    Entities of any other type are treated as background, so a model trained on
    `target_label` never sees another type's boundaries. A token counts as part of
    an entity when their spans overlap at all, which stays correct when an entity
    boundary does not land on a token boundary.
    """
    target_entities = sorted(
        (entity for entity in entities if entity["label"] == target_label),
        key=lambda entity: entity["start"],
    )

    labelled = []
    entity_index = 0
    inside = False

    for token in tokens:
        while (
            entity_index < len(target_entities)
            and token["start"] >= target_entities[entity_index]["end"]
        ):
            entity_index += 1
            inside = False

        entity = target_entities[entity_index] if entity_index < len(target_entities) else None
        overlaps = entity is not None and token["start"] < entity["end"] and token["end"] > entity["start"]

        if overlaps:
            label = f"I-{target_label}" if inside else f"B-{target_label}"
            inside = True
        else:
            label = "O"
            inside = False

        labelled.append({**token, "label": label})

    return labelled


def build_label_vocabulary(target_label: str) -> dict:
    """
    Build the fixed three-class `label2id`/`id2label` mapping for one entity type.

    Always `O=0`, `B-<target>=1`, `I-<target>=2`. Shared by row building and model
    instantiation so training data and model config agree on what an id means.
    """
    if not isinstance(target_label, str) or not target_label.strip():
        raise ValueError(f"target_label must be a non-empty string, got {target_label!r}.")

    labels = ["O", f"B-{target_label}", f"I-{target_label}"]

    return {
        "label2id": {label: index for index, label in enumerate(labels)},
        "id2label": dict(enumerate(labels)),
    }
