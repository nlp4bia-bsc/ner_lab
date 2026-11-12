"""Verify multihead: encoding, per-head scoring, the model, training, inference, the sweep."""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run

LABELS = ["DISEASE", "SYMPTOM"]
ARGUMENTS = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "fp16": False,
    "load_best_model_at_end": False,
    "save_strategy": "no",
}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


def one_hot(rows: pd.DataFrame, heads: int) -> np.ndarray:
    """Logits that predict exactly the gold the rows encode."""
    width = max(len(cell) for cell in rows["labels"])
    logits = np.zeros((len(rows), width, heads, 3), dtype=np.float32)

    for index, cell in enumerate(rows["labels"]):
        for position, ids in enumerate(cell):
            for head, label_id in enumerate(ids):
                logits[index, position, head, max(label_id, 0)] = 1.0

    return logits


def verify_encoding(checks: Checks, tokenizer) -> None:
    from fixtures import multitype_corpus

    from lab.ner.encoding import Encoder
    from lab.ner.multihead import (
        MultiHeadEncoder,
        describe_multihead_encoder,
        head_rows,
        multihead_encoder_from_description,
        resolve_entities_by_type,
        validate_entity_labels,
    )

    corpus = multitype_corpus(8)
    encoder = MultiHeadEncoder(tokenizer, LABELS, "es", max_length=64)
    rows = encoder.encode(corpus)

    checks.check(
        "labels hold one id per head", all(len(ids) == 2 for ids in rows.iloc[0]["labels"])
    )
    checks.equal(
        "labels run the length of the inputs",
        len(rows.iloc[0]["labels"]),
        len(rows.iloc[0]["input_ids"]),
    )

    for head, label in enumerate(LABELS):
        single = Encoder(tokenizer, label, "es", max_length=64, overlap_policy="none").encode(
            corpus
        )

        checks.equal(
            f"the {label} head is the single-label encoding",
            head_rows(rows, head)["labels"].tolist(),
            single["labels"].tolist(),
        )
        checks.equal(
            f"the inputs are the single-label ones ({label})",
            rows["input_ids"].tolist(),
            single["input_ids"].tolist(),
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MultiHeadEncoder(tokenizer, ["DISEASE"], "es", max_length=64).encode(corpus)

    checks.check(
        "a type with no head warns, with counts",
        any("type(s) with no head" in str(w.message) for w in caught),
    )

    text = "abcdefghij"
    entities = [
        {"start": 0, "end": 5, "label": "A", "text": "abcde"},
        {"start": 2, "end": 7, "label": "B", "text": "cdefg"},
    ]
    checks.equal(
        "types never resolve against each other",
        len(resolve_entities_by_type("d", entities, text, "error")),
        2,
    )
    checks.raises(
        "a same-type overlap is still resolved by the policy",
        ValueError,
        resolve_entities_by_type,
        "d",
        [*entities, {"start": 1, "end": 3, "label": "A", "text": "bc"}],
        text,
        "error",
        match="Overlapping",
    )

    checks.raises(
        "an empty catalogue raises", ValueError, validate_entity_labels, [], match="empty"
    )
    checks.raises(
        "a repeated head raises", ValueError, validate_entity_labels, ["A", "A"], match="repeats"
    )

    rebuilt = multihead_encoder_from_description(describe_multihead_encoder(encoder), tokenizer)

    checks.equal(
        "the description rebuilds the same encoder",
        rebuilt.encode(corpus)["labels"].tolist(),
        rows["labels"].tolist(),
    )


def verify_scoring(checks: Checks, tokenizer) -> None:
    from fixtures import multitype_corpus

    from lab.core import score_spans, sum_span_results
    from lab.ner.multihead import (
        MultiHeadEncoder,
        evaluate_multihead_predictions,
        multihead_gold_spans,
        multihead_predicted_spans,
        score_heads,
    )

    corpus = multitype_corpus(8)
    rows = MultiHeadEncoder(tokenizer, LABELS, "es", max_length=64).encode(corpus)
    logits = one_hot(rows, 2)

    gold = multihead_gold_spans(rows, tokenizer, LABELS)
    predicted = multihead_predicted_spans(rows, logits, tokenizer, LABELS)

    checks.check(
        "cross-type overlaps survive into the gold",
        gold.groupby(["filename", "start_span"])["label"].nunique().max() == 2,
    )
    checks.frames_equal(
        "perfect logits decode to the gold",
        predicted.sort_values(["filename", "label", "start_span"]).reset_index(drop=True)[
            ["filename", "label", "start_span", "end_span"]
        ],
        gold.sort_values(["filename", "label", "start_span"]).reset_index(drop=True)[
            ["filename", "label", "start_span", "end_span"]
        ],
    )

    metrics = evaluate_multihead_predictions(rows, logits, tokenizer, LABELS)

    checks.equal("micro strict F1 is perfect", metrics["span_strict_f1"], 1.0)
    checks.equal("macro strict F1 is perfect", metrics["span_strict_macro_f1"], 1.0)
    checks.equal("each head is scored", metrics["head_SYMPTOM_span_strict_f1"], 1.0)
    checks.check(
        "token diagnostics come per head",
        any(key.startswith("head_DISEASE_token_") for key in metrics),
    )

    no_symptoms = score_heads(gold, predicted[predicted["label"] == "DISEASE"], LABELS)
    n_disease = int((gold["label"] == "DISEASE").sum())

    checks.equal("a silent head scores zero in the macro", no_symptoms["span_strict_macro_f1"], 0.5)
    checks.equal(
        "the micro recall counts its misses",
        no_symptoms["span_strict_recall"],
        round(n_disease / len(gold), 4),
    )

    halved = predicted[predicted["label"] == "DISEASE"].iloc[::2]
    per_type = [
        score_spans(gold[gold["label"] == label], halved[halved["label"] == label], [label])
        for label in LABELS
    ]
    summed = sum_span_results(per_type)["strict"]

    checks.equal(
        "summing adds the counts",
        summed.correct,
        sum(result["strict"].correct for result in per_type),
    )
    checks.equal(
        "and recomputes precision from them", summed.precision, summed.correct / summed.actual
    )
    checks.equal("and recall", summed.recall, summed.correct / summed.possible)


def verify_collation(checks: Checks, tokenizer) -> None:
    from fixtures import multitype_corpus

    from lab.ner.multihead import MultiHeadEncoder
    from lab.ner.training.collation import TokenClassificationCollator
    from lab.ner.training.dataset import to_dataset

    rows = MultiHeadEncoder(tokenizer, LABELS, "es", max_length=64).encode(multitype_corpus(4))
    dataset = to_dataset(rows)
    batch = TokenClassificationCollator(tokenizer)([dataset[0], dataset[1]])

    checks.equal("inputs pad to a multiple of eight", batch["input_ids"].shape[1] % 8, 0)
    checks.equal(
        "labels pad to [batch, length, heads]",
        tuple(batch["labels"].shape),
        (2, batch["input_ids"].shape[1], 2),
    )
    checks.check(
        "padding is ignored", (batch["labels"][0, len(dataset[0]["labels"]) :] == -100).all().item()
    )

    tokenizer.padding_side = "left"
    left = TokenClassificationCollator(tokenizer)([dataset[0], dataset[1]])
    tokenizer.padding_side = "right"

    checks.check(
        "a left-padding tokenizer still pads right", left["attention_mask"][:, 0].all().item()
    )

    parquet_like = rows.copy()
    parquet_like["labels"] = [
        np.array([np.array(ids) for ids in cell], dtype=object) for cell in rows["labels"]
    ]

    checks.equal(
        "labels read back from parquet convert",
        to_dataset(parquet_like)[0]["labels"],
        dataset[0]["labels"],
    )


def verify_model(checks: Checks, base_model: Path) -> None:
    import torch

    from lab.ner.multihead import MultiHeadForTokenClassification, head_balanced_loss

    model = MultiHeadForTokenClassification(str(base_model), LABELS)
    input_ids = torch.randint(100, 1000, (2, 12))
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    checks.equal("logits are [batch, length, heads, 3]", tuple(output.logits.shape), (2, 12, 2, 3))

    logits = torch.randn(2, 5, 2, 3, requires_grad=True)
    labels = torch.full((2, 5, 2), -100)
    labels[:, :, 0] = 1

    only_first = head_balanced_loss(logits, labels)
    first_alone = torch.nn.functional.cross_entropy(
        logits[:, :, 0].reshape(-1, 3), labels[:, :, 0].reshape(-1)
    )

    checks.check(
        "a head with no labels does not dilute the loss",
        torch.isclose(only_first, first_alone).item(),
    )

    masked = head_balanced_loss(logits, torch.full((2, 5, 2), -100))
    masked.backward()

    checks.equal("an all-masked batch gives zero", masked.item(), 0.0)


def verify_training_and_inference(checks: Checks, root: Path, base_model: Path) -> None:
    from fixtures import multitype_corpus

    from lab.core import prepare_dataset, write_corpus
    from lab.ner.multihead import load_multihead_model, predict_multihead, train_multihead

    source = write_corpus(multitype_corpus(24), root / "source" / "documents.parquet")
    kfold = prepare_dataset(
        output_dir=root / "datasets", source_parquet=source, dataset_name="multi", kfolds=3
    )

    checks.raises(
        "a listed type annotated nowhere raises",
        ValueError,
        train_multihead,
        split_dir=kfold.split_dir,
        output_dir=root / "runs" / "typo",
        base_model=str(base_model),
        language="es",
        entity_labels=["DISEASE", "SYMPTON"],
        training_arguments=ARGUMENTS,
        max_length=64,
        match="SYMPTON",
    )

    result = train_multihead(
        split_dir=kfold.split_dir,
        output_dir=root / "runs" / "kfold",
        base_model=str(base_model),
        language="es",
        training_arguments=ARGUMENTS,
        max_length=64,
        early_stopping_patience=None,
        save_model=True,
        track_resources=False,
    )

    checks.equal("one run per rotatable fold", len(result.fold_metrics), 2)
    checks.equal(
        "the catalogue defaults to every annotated type",
        result.manifest["encoding"]["entity_labels"],
        LABELS,
    )
    checks.equal(
        "selection defaults to macro F1",
        result.manifest["training_arguments"]["metric_for_best_model"],
        "span_strict_macro_f1",
    )
    checks.check(
        "fold metrics carry the macro",
        any("macro" in column for column in result.fold_metrics.columns),
    )

    model_dir = result.run_dir / "fold_01" / "best_model"
    model, _, description = load_multihead_model(model_dir, device="cpu")

    checks.equal("the saved model keeps its head order", model.entity_labels, LABELS)
    checks.check("the backbone config is recorded", "model_type" in description["backbone_config"])
    checks.check("the model comes back in eval mode", not model.training)

    reordered = root / "reordered"
    reordered.mkdir()

    for path in model_dir.iterdir():
        (reordered / path.name).write_bytes(path.read_bytes())

    encoding = json.loads((reordered / "encoding.json").read_text(encoding="utf-8"))
    encoding["encoding"]["entity_labels"] = list(reversed(LABELS))
    (reordered / "encoding.json").write_text(json.dumps(encoding), encoding="utf-8")

    checks.raises(
        "weights under a reordered catalogue refuse to load",
        ValueError,
        load_multihead_model,
        reordered,
        match="different head catalogue",
    )

    corpus = multitype_corpus(6)
    predicted = predict_multihead(
        model_dir=model_dir, documents=corpus, output_dir=root / "inference", device="cpu"
    )

    checks.check("predictions are written", predicted.paths["predictions"].exists())
    checks.check(
        "gold in the documents is scored",
        predicted.metrics is not None and "span_strict_macro_f1" in predicted.metrics,
    )
    checks.check(
        "token diagnostics come per head",
        any(key.startswith("head_") and "_token_" in key for key in predicted.metrics),
    )
    checks.check("predicted labels are head types", set(predicted.spans["label"]) <= set(LABELS))

    unscored = predict_multihead(
        model_dir=model_dir,
        documents=dict(zip(corpus["doc_id"], corpus["text"], strict=True)),
        output_dir=root / "unscored",
        device="cpu",
    )

    checks.check("plain documents predict without scoring", unscored.metrics is None)


def verify_search(checks: Checks, root: Path, base_model: Path) -> None:
    from fixtures import multitype_corpus

    from lab.cli import read_config
    from lab.core import prepare_dataset, write_corpus
    from lab.ner.multihead import search_multihead, train_multihead

    source = write_corpus(multitype_corpus(24), root / "source" / "documents.parquet")
    split_dir = prepare_dataset(
        output_dir=root / "datasets", source_parquet=source, dataset_name="multi"
    ).split_dir

    result = search_multihead(
        split_dir=split_dir,
        output_dir=root / "sweep",
        base_models=str(base_model),
        language="es",
        search_space={
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-4},
            "weight_decay": None,
            "warmup_steps": None,
            "lr_scheduler_type": "linear",
            "effective_train_batch_size": {"type": "choice", "categories": [4, 8]},
        },
        training_arguments={"num_train_epochs": 1, "fp16": False},
        n_trials=2,
        seeds_per_trial=1,
        top_k_epochs=1,
        max_lengths=[64],
        max_micro_batch_size=4,
        early_stopping_patience=None,
        track_resources=False,
    )

    checks.check("the sweep succeeded", result.summary["success"])
    checks.equal("the objective defaults to macro F1", result.metric, "span_strict_macro_f1")
    checks.check(
        "scores reach the trials table", result.trials["span_strict_macro_f1"].notna().all()
    )
    checks.equal(
        "the winner retrains with train_multihead", result.best["task"], "ner.train_multihead"
    )
    checks.equal("the winner lists the catalogue", result.best["entity_labels"], LABELS)
    checks.check("the summary names the retraining task", "train_multihead" in result.summary)
    checks.equal(
        "the winner config is the winner block", read_config(result.paths["winner"]), result.best
    )

    block = {key: value for key, value in result.best.items() if key != "task"}
    block["training_arguments"] = {
        **block["training_arguments"],
        "num_train_epochs": 1,
        "fp16": False,
    }
    rerun = train_multihead(**block, track_resources=False)

    checks.equal(
        "the winner block runs through train_multihead",
        rerun.manifest["encoding"]["entity_labels"],
        LABELS,
    )


def verify_tasks(checks: Checks) -> None:
    from lab.core.tasks import resolve_task
    from lab.ner.multihead import predict_multihead, search_multihead, train_multihead

    checks.check(
        "ner.train_multihead resolves", resolve_task("ner.train_multihead") is train_multihead
    )
    checks.check(
        "ner.predict_multihead resolves", resolve_task("ner.predict_multihead") is predict_multihead
    )
    checks.check(
        "ner.search_multihead resolves", resolve_task("ner.search_multihead") is search_multihead
    )


def main() -> int:
    checks = Checks("multihead")

    if not torch_available():
        checks.skip("multihead", "torch is not installed in this environment")

        return checks.report()

    from fixtures import tiny_base_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    verify_encoding(checks, tokenizer)
    verify_scoring(checks, tokenizer)
    verify_collation(checks, tokenizer)
    verify_tasks(checks)

    shared = tempfile.TemporaryDirectory()
    root = Path(shared.name)
    base_model = tiny_base_model(root, tokenizer)

    verify_model(checks, base_model)
    verify_training_and_inference(checks, root / "training", base_model)
    verify_search(checks, root / "search", base_model)

    shared.cleanup()

    return checks.report()


if __name__ == "__main__":
    run(main)
