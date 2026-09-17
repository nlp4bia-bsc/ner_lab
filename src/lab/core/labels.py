"""Entity label normalization."""

from __future__ import annotations

import unicodedata
from typing import Mapping

import pandas as pd

CANONICAL_LABELS: tuple[str, ...] = ("DISEASE", "PROCEDURE", "SYMPTOM", "MEDICATION")

LABEL_ALIASES: dict[str, str] = {
    "ENFERMEDAD": "DISEASE",
    "ENFERMEDADES": "DISEASE",
    "DISEASE": "DISEASE",
    "DISEASES": "DISEASE",

    "PROCEDIMIENTO": "PROCEDURE",
    "PROCEDIMIENTOS": "PROCEDURE",
    "PROCEDURE": "PROCEDURE",
    "PROCEDURES": "PROCEDURE",

    "SINTOMA": "SYMPTOM",
    "SINTOMAS": "SYMPTOM",
    "SYMPTOM": "SYMPTOM",
    "SYMPTOMS": "SYMPTOM",

    "FARMACO": "MEDICATION",
    "FARMACOS": "MEDICATION",
    "MEDICAMENTO": "MEDICATION",
    "MEDICAMENTOS": "MEDICATION",
    "MEDICATION": "MEDICATION",
    "MEDICATIONS": "MEDICATION",
}


def normalize_label(label: object, aliases: Mapping[str, str] = LABEL_ALIASES) -> str:
    """
    Map a label alias onto its canonical form, ignoring case and accents.

    Labels absent from `aliases` are canonicalized but never renamed. Pass a
    different `aliases` mapping for a corpus with its own vocabulary.
    """
    raw_label = str(label).strip()

    if not raw_label:
        return raw_label

    key = _strip_accents(raw_label).upper().replace("-", "_").replace(" ", "_")

    return aliases.get(key, key)


def normalize_entity_labels(
    entities: list[dict],
    aliases: Mapping[str, str] = LABEL_ALIASES,
) -> list[dict]:
    """Normalize the label of every entity in one document's entity list."""
    return [{**entity, "label": normalize_label(entity["label"], aliases)} for entity in entities]


def normalize_annotation_labels(
    annotations: pd.DataFrame,
    label_column: str = "label",
    aliases: Mapping[str, str] = LABEL_ALIASES,
) -> pd.DataFrame:
    """Normalize an annotation DataFrame's label column, returning a copy."""
    if label_column not in annotations.columns:
        raise ValueError(
            f"Missing label column {label_column!r}. Available columns: {list(annotations.columns)}"
        )

    output = annotations.copy()
    output[label_column] = output[label_column].map(lambda label: normalize_label(label, aliases))

    return output


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)

    return "".join(character for character in normalized if not unicodedata.combining(character))
