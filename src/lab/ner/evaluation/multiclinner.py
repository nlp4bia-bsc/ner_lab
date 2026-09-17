"""
Official MultiClinNER shared-task scoring, kept as an option beside the canonical one.

The scoring math is carried over from the shared task's own script unchanged,
quirks included, so numbers produced here stay comparable to published results.
`lab.ner.evaluation.scoring` remains this library's canonical evaluation.

Both sides use half-open offsets (`text[start:end]`), which is what the official
overlap arithmetic already assumes — unlike nervaluate, no conversion is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lab.core.scoring import group_by_document, normalize_spans, safe_f1
from lab.core.spans import SPAN_COLUMNS

MATCH_KEY = ["filename", "label", "off0", "off1"]


def evaluate_annotations(
    gold: pd.DataFrame,
    predicted: pd.DataFrame,
    entity: str | None = None,
) -> dict[str, Any]:
    """
    Score predictions with the official metrics, returning `{entity, strict, char_f1}`.

    Both frames use the canonical span columns.
    """
    gold_spans = normalize_spans(gold)
    predicted_spans = normalize_spans(predicted).drop_duplicates(subset=MATCH_KEY)

    if entity:
        gold_spans = gold_spans[gold_spans.label == entity]
        predicted_spans = predicted_spans[predicted_spans.label == entity]

    gold_documents = group_by_document(gold_spans)
    predicted_documents = group_by_document(predicted_spans)

    return {
        "entity": entity,
        "strict": strict_metric(gold_documents, predicted_documents),
        "char_f1": char_metric(gold_documents, predicted_documents),
    }


def evaluate_tsv(
    reference_path: str | Path,
    predicted_path: str | Path,
    entity: str | None = None,
) -> dict[str, Any]:
    """Score one prediction TSV against one gold TSV — the original script's entry point."""
    return evaluate_annotations(
        gold=read_annotation_tsv(reference_path),
        predicted=read_annotation_tsv(predicted_path),
        entity=entity,
    )


def strict_metric(
    gold_documents: dict[str, list[dict[str, Any]]],
    predicted_documents: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    """
    Exact-match precision, recall and F1 on document, label and both offsets.

    Matching is exclusive on both sides, so a repeated prediction of one gold span
    scores one true positive and the rest false positives.

    Verbatim from the official script, including the quirk that only documents
    present in the *gold* keys are visited: predictions in a document with no gold
    annotations at all are never counted as false positives. `char_metric` uses the
    union of both key sets instead.
    """
    tp = fp = fn = 0

    for doc_id in set(gold_documents):
        gold = gold_documents.get(doc_id, [])
        predicted = predicted_documents.get(doc_id, [])

        matched_gold: set[int] = set()
        matched_predicted: set[int] = set()

        for gold_index, gold_annotation in enumerate(gold):
            for predicted_index, predicted_annotation in enumerate(predicted):
                if predicted_index in matched_predicted:
                    continue

                if strict_match(gold_annotation, predicted_annotation):
                    matched_gold.add(gold_index)
                    matched_predicted.add(predicted_index)
                    tp += 1
                    break

        fp += len(predicted) - len(matched_predicted)
        fn += len(gold) - len(matched_gold)

    return safe_f1(tp, fp, fn)


def char_metric(
    gold_documents: dict[str, list[dict[str, Any]]],
    predicted_documents: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    """
    Character-overlap precision, recall and F1, averaged over annotations.

    Recall is the mean best-overlap score of every gold annotation, precision the
    mean over every predicted one. Matching is per-annotation and not exclusive —
    one prediction can be the best match for several gold spans at once. Official
    behaviour, and the reason the nervaluate partial/ent_type scoring is kept.
    """
    gold_scores: list[float] = []
    predicted_scores: list[float] = []

    for doc_id in set(gold_documents) | set(predicted_documents):
        gold = gold_documents.get(doc_id, [])
        predicted = predicted_documents.get(doc_id, [])

        gold_scores.extend(
            max((char_overlap_f1(annotation, other) for other in predicted), default=0.0)
            for annotation in gold
        )
        predicted_scores.extend(
            max((char_overlap_f1(other, annotation) for other in gold), default=0.0)
            for annotation in predicted
        )

    recall = sum(gold_scores) / len(gold_scores) if gold_scores else 0.0
    precision = sum(predicted_scores) / len(predicted_scores) if predicted_scores else 0.0

    return {
        **safe_prf(precision, recall),
        "n_gold": len(gold_scores),
        "n_pred": len(predicted_scores),
    }


def strict_match(gold: dict[str, Any], predicted: dict[str, Any]) -> bool:
    """True when document, label and both offsets are identical."""
    return (
        gold["filename"] == predicted["filename"]
        and gold["label"] == predicted["label"]
        and gold["off0"] == predicted["off0"]
        and gold["off1"] == predicted["off1"]
    )


def char_overlap_f1(gold: dict[str, Any], predicted: dict[str, Any]) -> float:
    """F1 of the character overlap between one gold and one predicted annotation."""
    if gold["filename"] != predicted["filename"] or gold["label"] != predicted["label"]:
        return 0.0

    intersection = max(
        0, min(gold["off1"], predicted["off1"]) - max(gold["off0"], predicted["off0"])
    )

    if intersection == 0:
        return 0.0

    recall = intersection / (gold["off1"] - gold["off0"])
    precision = intersection / (predicted["off1"] - predicted["off0"])

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def safe_prf(precision: float, recall: float) -> dict[str, float]:
    """F1 from precision and recall directly, no counts involved."""
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def spans_from_corpus(
    corpus: pd.DataFrame | str | Path,
    entity: str | None = None,
) -> pd.DataFrame:
    """
    Turn a canonical corpus into the official annotation schema.

    Labels are taken verbatim, and overlaps are left as annotated: the official
    gold is scored as distributed, not as an overlap policy would rewrite it. Span
    text is re-derived from the document rather than trusted from the entity.
    """
    if not isinstance(corpus, pd.DataFrame):
        corpus = pd.read_parquet(corpus, columns=["doc_id", "text", "entities_json"])

    rows: list[dict[str, Any]] = []

    for document in corpus.itertuples(index=False):
        text = str(document.text)
        entities = json.loads(document.entities_json) if document.entities_json else []

        for annotation in entities:
            label = str(annotation["label"])

            if entity and label != entity:
                continue

            start, end = int(annotation["start"]), int(annotation["end"])
            rows.append(
                {
                    "filename": str(document.doc_id),
                    "label": label,
                    "start_span": start,
                    "end_span": end,
                    "text": text[start:end],
                }
            )

    if not rows:
        return pd.DataFrame(columns=SPAN_COLUMNS)

    return (
        pd.DataFrame(rows, columns=SPAN_COLUMNS)
        .sort_values(["filename", "start_span", "end_span"])
        .reset_index(drop=True)
    )


def read_annotation_tsv(path: str | Path) -> pd.DataFrame:
    """
    Read an official-format annotation TSV.

    Everything is read as text with pandas' NA sentinels disabled, so a mention
    literally spelled `NA` or `null` survives as itself.
    """
    annotations = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    missing = [column for column in SPAN_COLUMNS if column not in annotations.columns]

    if missing:
        raise ValueError(f"{path} is missing columns {missing}. Required: {SPAN_COLUMNS}")

    return annotations[SPAN_COLUMNS].copy()


def write_annotation_tsv(annotations: pd.DataFrame, path: str | Path) -> Path:
    """Write an annotation frame in the official TSV format."""
    path = Path(path)
    annotations[SPAN_COLUMNS].to_csv(path, sep="\t", index=False)

    return path


def format_summary(results: dict[str, Any]) -> str:
    """One-line summary of an `evaluate_annotations` result, for run logs."""
    strict = results["strict"]
    char_f1 = results["char_f1"]

    return (
        f"[MultiClinNER official] entity={results['entity']} | "
        f"strict P/R/F1 = {strict['precision']}/{strict['recall']}/{strict['f1']} "
        f"(tp={strict['tp']} fp={strict['fp']} fn={strict['fn']}) | "
        f"char P/R/F1 = {char_f1['precision']}/{char_f1['recall']}/{char_f1['f1']} "
        f"(n_gold={char_f1['n_gold']} n_pred={char_f1['n_pred']})"
    )
