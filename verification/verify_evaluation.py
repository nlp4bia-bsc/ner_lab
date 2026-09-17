"""Verify span reconstruction and scoring, and diff both against NER-API's metrics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run
from fixtures import samples_root, synthetic_corpus

from lab.core import safe_f1
from lab.ner.evaluation import (
    bio_to_spans,
    entity_tags,
    evaluate_predictions,
    expand_to_word_extent,
    gold_spans,
    predicted_spans,
    span_metrics,
    strip_bio_prefix,
    token_confusion_matrix,
    token_metrics,
    token_metrics_by_entity,
)
from lab.ner.evaluation import multiclinner

NER_API_ENV = "NER_API_ROOT"
DEFAULT_NER_API = Path.home() / "bsc" / "NER-API"

ID2LABEL = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE"}


def one_hot(label_ids: list[list[int]], num_labels: int = 3) -> np.ndarray:
    width = max(len(row) for row in label_ids)
    logits = np.full((len(label_ids), width, num_labels), -10.0, dtype=np.float32)

    for row, ids in enumerate(label_ids):
        for position, label_id in enumerate(ids):
            logits[row, position, max(label_id, 0)] = 10.0

    return logits


def verify_helpers(checks: Checks) -> None:
    checks.equal("B- prefix stripped", strip_bio_prefix("B-DISEASE"), "DISEASE")
    checks.equal("O is left alone", strip_bio_prefix("O"), "O")
    checks.equal("tags come from the vocabulary", entity_tags(ID2LABEL), ["DISEASE"])

    score = safe_f1(tp=2, fp=1, fn=1)
    checks.equal("precision", score["precision"], 0.6667)
    checks.equal("recall", score["recall"], 0.6667)
    checks.equal("f1", score["f1"], 0.6667)
    checks.equal("empty counts give zero", safe_f1(0, 0, 0)["f1"], 0.0)

    spans = bio_to_spans("d", [(0, 5), (6, 11), (12, 14)], ["B-DISEASE", "I-DISEASE", "O"])
    checks.equal("B then I is one span", len(spans), 1)
    checks.equal("the span covers both tokens", (spans[0]["start_span"], spans[0]["end_span"]), (0, 11))

    two = bio_to_spans("d", [(0, 5), (6, 11)], ["B-DISEASE", "B-DISEASE"])
    checks.equal("B then B is two spans", len(two), 2)

    orphan = bio_to_spans("d", [(0, 5)], ["I-DISEASE"])
    checks.equal("a leading I- still opens a span", len(orphan), 1)

    switch = bio_to_spans("d", [(0, 5), (6, 11)], ["B-DISEASE", "I-GENE"])
    checks.equal("I- of another type starts a new span", len(switch), 2)

    checks.equal("all O gives nothing", bio_to_spans("d", [(0, 5)], ["O"]), [])
    checks.raises(
        "mismatched lengths raise", ValueError, bio_to_spans, "d", [(0, 5)], ["O", "O"]
    )

    expanded = expand_to_word_extent([(0, 3), (3, 7), (8, 12)], [0, 0, 1])
    checks.equal(
        "a first subword grows to its word's end", expanded, [(0, 7), (3, 7), (8, 12)]
    )


def verify_reconstruction(checks: Checks, tokenizer) -> None:
    from lab.ner.encoding import Encoder

    corpus = synthetic_corpus(10)
    encoder = Encoder(tokenizer, "DISEASE", "es", max_length=64)
    rows = encoder.encode(corpus)

    gold = gold_spans(rows, tokenizer, encoder.id2label)

    checks.check("gold spans are recovered", len(gold) > 0)
    checks.equal(
        "recovered spans match the corpus entity count",
        len(gold),
        int(corpus["n_entities"].sum()),
    )

    corpus_spans = multiclinner.spans_from_corpus(corpus)
    recovered = set(zip(gold["filename"], gold["start_span"], gold["end_span"]))
    annotated = set(
        zip(corpus_spans["filename"], corpus_spans["start_span"], corpus_spans["end_span"])
    )

    checks.equal("recovered offsets equal the annotated ones", recovered, annotated)

    perfect = one_hot([[max(int(label), 0) for label in row] for row in rows["labels"]])
    predicted = predicted_spans(rows, perfect, tokenizer, encoder.id2label)

    checks.frames_equal("perfect predictions reproduce the gold spans", predicted, gold)

    metrics = span_metrics(gold, predicted, id2label=encoder.id2label)
    checks.equal("perfect predictions score 1.0 strict", metrics["span_strict_f1"], 1.0)
    checks.equal("nothing is missed", metrics["span_strict_missed"], 0)
    checks.equal("nothing is spurious", metrics["span_strict_spurious"], 0)

    empty = np.zeros((len(rows), max(len(row) for row in rows["labels"]), 3), dtype=np.float32)
    empty[:, :, 0] = 10.0

    none_predicted = predicted_spans(rows, empty, tokenizer, encoder.id2label)
    checks.equal("predicting all O gives no spans", len(none_predicted), 0)

    empty_metrics = span_metrics(gold, none_predicted, id2label=encoder.id2label)
    checks.equal("scoring nothing gives zero F1", empty_metrics["span_strict_f1"], 0.0)
    checks.equal("every gold span is missed", empty_metrics["span_strict_missed"], len(gold))

    checks.raises(
        "a prediction/row length mismatch raises",
        ValueError,
        predicted_spans,
        rows,
        perfect[:-1],
        tokenizer,
        encoder.id2label,
    )
    checks.raises(
        "missing columns raise",
        ValueError,
        gold_spans,
        rows.drop(columns=["word_ids"]),
        tokenizer,
        encoder.id2label,
    )

    all_metrics = evaluate_predictions(
        rows, perfect, tokenizer, encoder.id2label, include_confusion=True
    )

    checks.equal("span and token metrics agree at perfection", all_metrics["token_micro_f1"], 1.0)
    checks.equal("token accuracy is perfect too", all_metrics["token_accuracy"], 1.0)
    checks.check("per-entity token metrics appear", "token_entity_DISEASE_f1" in all_metrics)
    checks.check("the confusion matrix appears", "confusion_gold_O_pred_O" in all_metrics)
    checks.check("per-tag token metrics appear", "token_tag_B-DISEASE_f1" in all_metrics)


def verify_token_metrics(checks: Checks) -> None:
    labels = np.array([[0, 1, 2, -100]])
    perfect = one_hot([[0, 1, 2, 0]])

    metrics = token_metrics(perfect, labels, ID2LABEL)

    checks.equal("perfect token f1", metrics["token_micro_f1"], 1.0)
    checks.equal("the masked position is skipped", metrics["token_tag_B-DISEASE_tp"], 1)

    wrong = one_hot([[0, 0, 0, 0]])
    missed = token_metrics(wrong, labels, ID2LABEL)

    checks.equal("all-O predictions score zero", missed["token_micro_f1"], 0.0)
    checks.equal("both entity tokens are false negatives", missed["token_tag_B-DISEASE_fn"], 1)
    checks.equal("accuracy still counts the O token", missed["token_accuracy"], 0.3333)

    empty = token_metrics(one_hot([[0]]), np.array([[-100]]), ID2LABEL)
    checks.equal("all-masked input gives zeros", empty["token_micro_f1"], 0.0)


def verify_multiclinner(checks: Checks) -> None:
    gold = pd.DataFrame(
        [
            {"filename": "d1", "label": "DISEASE", "start_span": 0, "end_span": 10, "text": "x"},
            {"filename": "d1", "label": "DISEASE", "start_span": 20, "end_span": 30, "text": "y"},
        ]
    )

    identical = multiclinner.evaluate_annotations(gold, gold)

    checks.equal("identical input scores 1.0 strict", identical["strict"]["f1"], 1.0)
    checks.equal("identical input scores 1.0 char", identical["char_f1"]["f1"], 1.0)
    checks.equal("both gold spans are true positives", identical["strict"]["tp"], 2)

    shifted = gold.assign(start_span=gold["start_span"] + 2)
    partial = multiclinner.evaluate_annotations(gold, shifted)

    checks.equal("a shifted span is not a strict match", partial["strict"]["f1"], 0.0)
    checks.check("but it still scores on character overlap", partial["char_f1"]["f1"] > 0.5)

    empty = pd.DataFrame(columns=list(gold.columns))
    nothing = multiclinner.evaluate_annotations(gold, empty)

    checks.equal("predicting nothing scores zero", nothing["strict"]["f1"], 0.0)
    checks.equal("every gold span is a false negative", nothing["strict"]["fn"], 2)

    checks.equal(
        "char overlap is symmetric on identical spans",
        multiclinner.char_overlap_f1(
            {"filename": "d", "label": "L", "off0": 0, "off1": 10},
            {"filename": "d", "label": "L", "off0": 0, "off1": 10},
        ),
        1.0,
    )
    checks.equal(
        "a different label scores zero overlap",
        multiclinner.char_overlap_f1(
            {"filename": "d", "label": "L", "off0": 0, "off1": 10},
            {"filename": "d", "label": "M", "off0": 0, "off1": 10},
        ),
        0.0,
    )

    checks.check("the summary formats", "MultiClinNER" in multiclinner.format_summary(identical))

    corpus = synthetic_corpus(4)
    spans = multiclinner.spans_from_corpus(corpus)

    checks.equal("corpus conversion keeps every entity", len(spans), int(corpus["n_entities"].sum()))
    checks.equal("labels are verbatim", set(spans["label"]), {"DISEASE"})
    checks.check(
        "span text is re-derived from the document",
        all(
            corpus.loc[corpus["doc_id"] == row.filename, "text"].iloc[0][
                row.start_span:row.end_span
            ]
            == row.text
            for row in spans.itertuples(index=False)
        ),
    )

    verbatim = multiclinner.spans_from_corpus(synthetic_corpus(4, label="FARMACO"))
    checks.equal("no normalization happens here either", set(verbatim["label"]), {"FARMACO"})


def verify_against_ner_api(checks: Checks, tokenizer) -> None:
    ner_api = Path(os.environ.get(NER_API_ENV, DEFAULT_NER_API))

    if not (ner_api / "src").is_dir():
        checks.skip("NER-API metrics equivalence", f"{ner_api} not found")
        return

    sys.path.insert(0, str(ner_api / "src"))

    try:
        from metrics import metrics as old
        from metrics import multiclinner_eval as old_official
    except Exception as error:
        checks.skip("NER-API metrics equivalence", f"cannot import metrics: {error}")
        return

    from lab.ner.encoding import Encoder

    samples = samples_root()
    source = samples / "MultiClinNER-es-train-disease" if samples else None

    if source is not None and (source / "txt").is_dir():
        from lab.core import build_corpus

        corpus = build_corpus(source / "txt", source / "ann", normalize_labels=True).head(40)
    else:
        corpus = synthetic_corpus(20)

    encoder = Encoder(tokenizer, "DISEASE", "es", max_length=128)
    rows = encoder.encode(corpus.reset_index(drop=True))

    rng = np.random.default_rng(0)
    predictions = rng.normal(size=(len(rows), max(len(row) for row in rows["input_ids"]), 3))
    predictions = predictions.astype(np.float32)

    old_gold = old.dataframe_gold_spans_from_token_labels(
        df=rows, tokenizer=tokenizer, id2label=encoder.id2label
    )
    new_gold = gold_spans(rows, tokenizer, encoder.id2label)

    checks.frames_equal("gold spans identical to NER-API", new_gold, old_gold)

    old_predicted = old.dataframe_pred_spans_from_token_logits(
        df=rows, predictions=predictions, tokenizer=tokenizer, id2label=encoder.id2label
    )
    new_predicted = predicted_spans(rows, predictions, tokenizer, encoder.id2label)

    checks.frames_equal("predicted spans identical to NER-API", new_predicted, old_predicted)

    old_span = old.compute_span_metrics_from_logits(
        df=rows,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=encoder.id2label,
        max_length=128,
    )
    new_span = span_metrics(new_gold, new_predicted, id2label=encoder.id2label)

    checks.equal("span metrics identical to NER-API", new_span, old_span)

    labels = np.array(
        [
            [int(label) for label in row] + [-100] * (predictions.shape[1] - len(row))
            for row in rows["labels"]
        ]
    )

    checks.equal(
        "token metrics identical to NER-API",
        token_metrics(predictions, labels, encoder.id2label),
        old.compute_token_classification_metrics(predictions, labels, encoder.id2label),
    )
    checks.equal(
        "per-entity token metrics identical to NER-API",
        token_metrics_by_entity(predictions, labels, encoder.id2label),
        old.compute_token_classification_metrics_by_entity(
            predictions, labels, encoder.id2label
        ),
    )
    checks.equal(
        "the confusion matrix is identical to NER-API",
        token_confusion_matrix(predictions, labels, encoder.id2label),
        old.compute_token_confusion_matrix(predictions, labels, encoder.id2label),
    )

    checks.equal(
        "official scoring identical to NER-API",
        multiclinner.evaluate_annotations(new_gold, new_predicted),
        old_official.evaluate_annotations(new_gold, new_predicted),
    )


def main() -> int:
    from transformers import AutoTokenizer

    checks = Checks("evaluation")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    verify_helpers(checks)
    verify_reconstruction(checks, tokenizer)
    verify_token_metrics(checks)
    verify_multiclinner(checks)
    verify_against_ner_api(checks, tokenizer)

    return checks.report()


if __name__ == "__main__":
    run(main)
