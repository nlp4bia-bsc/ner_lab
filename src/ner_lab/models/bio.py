"""BIO label-scheme reasoning, kept free of torch so it can be reasoned about on its own."""

from __future__ import annotations


def normalize_label2id(label2id: dict) -> dict[str, int]:
    """Coerce a label-to-id mapping to `str -> int`."""
    return {str(label): int(index) for label, index in label2id.items()}


def normalize_id2label(id2label: dict) -> dict[int, str]:
    """Coerce an id-to-label mapping to `int -> str`."""
    return {int(index): str(label) for index, label in id2label.items()}


def is_inside_label(label: str) -> bool:
    """True for `I-*` continuation labels."""
    return label.startswith("I-")


def bio_entity(label: str) -> str | None:
    """The entity type a BIO label refers to, or None for `O`."""
    if label == "O":
        return None

    if "-" not in label:
        return label

    return label.split("-", maxsplit=1)[1]


def bio_constraint_masks(id2label: dict) -> tuple[list[bool], list[list[bool]]]:
    """
    Mark the BIO transitions a well-formed sequence can never take.

    Invalid: starting on `I-X`, `O -> I-X`, and `B-X`/`I-X -> I-Y` for `X != Y`.
    Returned as plain nested booleans; the CRF turns them into buffers.
    """
    id2label = normalize_id2label(id2label)
    num_labels = len(id2label)

    invalid_start = [False] * num_labels
    invalid_transitions = [[False] * num_labels for _ in range(num_labels)]

    for to_id, to_label in id2label.items():
        if is_inside_label(to_label):
            invalid_start[to_id] = True

    for from_id, from_label in id2label.items():
        for to_id, to_label in id2label.items():
            if not is_inside_label(to_label):
                continue

            if from_label == "O" or bio_entity(from_label) != bio_entity(to_label):
                invalid_transitions[from_id][to_id] = True

    return invalid_start, invalid_transitions
