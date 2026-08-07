"""Token-level scoring, for diagnosing what span metrics only summarize."""

from __future__ import annotations

import numpy as np

from ner_lab.encoding.rows import IGNORE_INDEX
from ner_lab.evaluation.scoring import safe_f1
from ner_lab.evaluation.spans import strip_bio_prefix

TOKEN_METRIC_NAMES = (
    "token_accuracy",
    "token_micro_precision",
    "token_micro_recall",
    "token_micro_f1",
    "token_macro_precision",
    "token_macro_recall",
    "token_macro_f1",
)


def token_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
    outside_label: str = "O",
) -> dict[str, float]:
    """
    Micro and macro token scores, plus per-tag counts.

    The outside label is excluded from precision, recall and F1; accuracy is over
    every unmasked token.
    """
    predicted, gold = decode_pairs(predictions, labels, id2label, ignore_index)

    if not gold:
        return dict.fromkeys(TOKEN_METRIC_NAMES, 0.0)

    accuracy = sum(a == b for a, b in zip(predicted, gold)) / len(gold)
    tags = sorted({label for label in set(gold) | set(predicted) if label != outside_label})

    micro = {"tp": 0, "fp": 0, "fn": 0}
    per_tag: dict[str, float | int] = {}
    scores = []

    for tag in tags:
        score = _counts(predicted, gold, lambda label, tag=tag: label == tag)
        scores.append(score)

        for key in micro:
            micro[key] += score[key]

        for key, value in score.items():
            per_tag[f"token_tag_{tag}_{key}"] = value

    overall = safe_f1(**micro)

    return {
        "token_accuracy": round(float(accuracy), 4),
        "token_micro_precision": overall["precision"],
        "token_micro_recall": overall["recall"],
        "token_micro_f1": overall["f1"],
        "token_macro_precision": _mean(scores, "precision"),
        "token_macro_recall": _mean(scores, "recall"),
        "token_macro_f1": _mean(scores, "f1"),
        **per_tag,
    }


def token_metrics_by_entity(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, float]:
    """Token scores with `B-` and `I-` collapsed, so `DISEASE` is scored as one class."""
    predicted, gold = decode_pairs(predictions, labels, id2label, ignore_index)

    entities = sorted(
        {strip_bio_prefix(label) for label in set(gold) | set(predicted) if label != "O"}
    )

    results: dict[str, float] = {}

    for entity in entities:
        score = _counts(
            predicted, gold, lambda label, entity=entity: strip_bio_prefix(label) == entity
        )

        for key, value in score.items():
            results[f"token_entity_{entity}_{key}"] = value

    return results


def token_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, int]:
    """A flattened gold-by-predicted count matrix over the label vocabulary."""
    predicted, gold = decode_pairs(predictions, labels, id2label, ignore_index)

    tags = sorted(str(label) for label in id2label.values())
    counts: dict[tuple[str, str], int] = {}

    for predicted_label, gold_label in zip(predicted, gold):
        key = (gold_label, predicted_label)
        counts[key] = counts.get(key, 0) + 1

    return {
        f"confusion_gold_{gold_label}_pred_{predicted_label}": counts.get(
            (gold_label, predicted_label), 0
        )
        for gold_label in tags
        for predicted_label in tags
    }


def decode_pairs(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: dict,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[str], list[str]]:
    """Flatten predictions and labels into aligned label-name lists, dropping masked positions."""
    id2label = {int(index): str(label) for index, label in id2label.items()}
    predicted_ids = np.argmax(predictions, axis=-1)

    predicted, gold = [], []

    for predicted_row, gold_row in zip(predicted_ids, labels):
        for predicted_id, gold_id in zip(predicted_row, gold_row):
            if int(gold_id) == ignore_index:
                continue

            gold.append(id2label[int(gold_id)])
            predicted.append(id2label[int(predicted_id)])

    return predicted, gold


def _counts(predicted: list[str], gold: list[str], matches) -> dict[str, float | int]:
    tp = fp = fn = 0

    for predicted_label, gold_label in zip(predicted, gold):
        predicted_hit = matches(predicted_label)
        gold_hit = matches(gold_label)

        if predicted_hit and gold_hit:
            tp += 1
        elif predicted_hit:
            fp += 1
        elif gold_hit:
            fn += 1

    return safe_f1(tp=tp, fp=fp, fn=fn)


def _mean(scores: list[dict], key: str) -> float:
    return round(float(np.mean([score[key] for score in scores])), 4) if scores else 0.0
