"""Validate complete BIO2 gold at structural word positions before model training."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_word_supervision(rows: pd.DataFrame, entity_labels: list[str]) -> None:
    """Reject unknown words and illegal gold with document, window, head and position."""
    for row in rows.itertuples(index=False):
        values = np.asarray(row.labels)
        values = values[:, None] if values.ndim == 1 else values
        mask = np.asarray(row.crf_word_mask, dtype=bool)
        if len(mask) != len(row.input_ids) or values.shape != (len(mask), len(entity_labels)):
            raise ValueError(f"Invalid CRF dimensions in document {row.doc_id!r}.")
        active = values[mask]
        unknown = np.argwhere((active < 0) | (active > 2))
        illegal = np.zeros_like(active, dtype=bool)
        if len(active):
            illegal[0] = active[0] == 2
            illegal[1:] = (active[1:] == 2) & (active[:-1] == 0)
        bad = unknown if len(unknown) else np.argwhere(illegal)
        if len(bad):
            word, head = bad[0]
            position = np.flatnonzero(mask)[word]
            previous = "START" if word == 0 else str(active[word - 1, head])
            raise ValueError(
                f"Invalid BIO2 supervision: document={row.doc_id!r}, "
                f"window=[{row.window_start},{row.window_end}), "
                f"label={entity_labels[head]!r}, position={position}, "
                f"transition={previous}->{active[word, head]}. "
                "Unknown whole words require an explicit partial-supervision policy."
            )
