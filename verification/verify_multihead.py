"""Verify shared multihead BIO encoding, training, scoring and checkpoint round trips offline."""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from _harness import Checks, run
from fixtures import tiny_base_model
from transformers import AutoModel, BertTokenizer, DistilBertConfig

from lab.core import build_corpus, prepare_dataset
from lab.core.spans import span_dataframe
from lab.ner import Encoder, build_model, evaluate_predictions, training_arguments
from lab.ner.encoding.encoder import describe_encoder, encoder_from_description
from lab.ner.evaluation.multihead import strict_multihead_metrics
from lab.ner.evaluation.spans import gold_spans, predicted_spans
from lab.ner.hpo.variants import build_variants, encode_variants
from lab.ner.inference import load_model, predict_entities, predict_logits
from lab.ner.models.capacity import check_max_length
from lab.ner.training import model_encoding, train
from lab.ner.training.collation import MultiHeadCollator
from lab.ner.training.dataset import to_dataset


def local_tokenizer():
    """Small real WordPiece tokenizer, including a split clinical word."""
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "paciente",
        "con",
        "dolor",
        "abdominal",
        "intens",
        "##o",
        "sin",
        "fiebre",
        ".",
        "a",
        "otro",
    ]
    return BertTokenizer(vocab={word: index for index, word in enumerate(words)})


def overlap_corpus():
    text = "Paciente con dolor abdominal intenso."
    entities = [
        ("SYMPTOM", "dolor abdominal"),
        ("FINDING", "abdominal intenso"),
        ("NESTED", "abdominal"),
        ("TOKEN", "abdominal"),
    ]
    annotations = []
    documents = {}
    for index in range(8):
        name = f"doc_{index}"
        documents[name] = text
        for label, mention in entities:
            start = text.index(mention)
            annotations.append(
                dict(
                    filename=name,
                    label=label,
                    start_span=start,
                    end_span=start + len(mention),
                    text=mention,
                )
            )
    documents["empty"] = "Sin fiebre."

    return build_corpus(documents, pd.DataFrame(annotations))


def oracle(rows):
    labels = np.asarray(rows["labels"].tolist())
    return np.eye(3)[np.maximum(labels, 0)] * 20


def verify_encoding(checks, tokenizer, corpus):
    encoder = Encoder(tokenizer, None, "es", max_length=32)
    from unittest.mock import patch

    from lab.ner.encoding.encoder import tokenize_document

    with patch("lab.ner.encoding.encoder.tokenize_document", wraps=tokenize_document) as tokenized:
        rows = encoder.encode(corpus)
        checks.equal("each document tokenized once", tokenized.call_count, len(corpus))
    checks.equal(
        "deterministic automatic heads",
        encoder.entity_labels,
        ["FINDING", "NESTED", "SYMPTOM", "TOKEN"],
    )
    checks.equal("one shared row per window", len(rows), len(corpus))
    checks.equal(
        "overlaps and nesting survive BIO roundtrip",
        len(gold_spans(rows, tokenizer, encoder.id2label)),
        32,
    )
    for row in rows.itertuples():
        labels = np.asarray(row.labels)
        checks.equal("shape L K", labels.shape, (len(row.input_ids), 4))
        checks.check("specials masked in all heads", np.all(labels[[0, -1]] == -100))
    checks.check(
        "subwords masked in all heads",
        any(np.all(np.asarray(row)[1:-1] == -100, axis=1).any() for row in rows.labels),
    )
    empty = np.asarray(rows[rows.doc_id == "empty"].iloc[0].labels)
    checks.check("unannotated document O", np.all(empty[empty != -100] == 0))
    from lab.core.segmentation import split_into_words

    checks.equal(
        "UAX words preserved",
        [word["text"] for word in split_into_words("a:b 12.5")],
        ["a:b", "12.5"],
    )
    text = "dolor abdominal. Otro dolor."
    annotations = [
        dict(start=0, end=5, label="SYMPTOM", text="dolor"),
        dict(start=6, end=15, label="FINDING", text="abdominal"),
        dict(start=22, end=27, label="SYMPTOM", text="dolor"),
    ]
    disjoint = pd.DataFrame(encoder.encode_document("disjoint", text, annotations))
    checks.equal(
        "multiple disjoint same and different labels",
        len(gold_spans(disjoint, tokenizer, encoder.id2label, entity_labels=encoder.entity_labels)),
        3,
    )
    described = describe_encoder(encoder)
    checks.equal(
        "encoder description roundtrip",
        describe_encoder(encoder_from_description(described, tokenizer)),
        described,
    )
    dataset = to_dataset(rows)
    collator = MultiHeadCollator(tokenizer, pad_to_multiple_of=8)
    batch = collator([dataset[0], dataset[-1]])
    checks.equal("variable padded labels", tuple(batch["labels"].shape), (2, 16, 4))
    checks.check("padding is vector -100", (batch["labels"][1, -1] == -100).all().item())
    tokenizer.padding_side = "left"
    left = collator([dataset[0], dataset[-1]])
    checks.check("left padding is vector -100", (left["labels"][1, 0] == -100).all().item())
    tokenizer.padding_side = "right"
    first = corpus.iloc[0]
    conflicting = json.loads(first.entities_json)
    conflicting.append({**conflicting[0], "start": 13, "end": 35, "text": first.text[13:35]})
    checks.raises(
        "intraclass overlap strict",
        ValueError,
        encoder.encode_document,
        first.doc_id,
        first.text,
        conflicting,
        match="Overlapping annotations",
    )
    checks.raises(
        "none policy cannot hide overlaps",
        ValueError,
        Encoder(
            tokenizer, None, "es", entity_labels=encoder.entity_labels, overlap_policy="none"
        ).encode_document,
        first.doc_id,
        first.text,
        conflicting,
        match="intraclass",
    )
    for strategy, context in [("greedy", None), ("context", 2)]:
        window_encoder = Encoder(
            tokenizer,
            None,
            "es",
            max_length=12,
            strategy=strategy,
            context_tokens=context,
            entity_labels=encoder.entity_labels,
        )
        windows = window_encoder.encode(corpus)
        checks.check(
            f"{strategy} bounded windows", all(map(lambda ids: len(ids) <= 12, windows.input_ids))
        )
        checks.check(
            f"{strategy} labels L K", all(np.asarray(row).shape[1] == 4 for row in windows.labels)
        )
    single = Encoder(tokenizer, None, "es")
    one = corpus.copy()
    one["entities_json"] = one.entities_json.map(
        lambda raw: json.dumps([e for e in json.loads(raw) if e["label"] == "TOKEN"])
    )
    one["n_entities"] = one.entities_json.map(lambda raw: len(json.loads(raw)))
    checks.equal(
        "K=1 automatic discovery", np.asarray(single.encode(one).iloc[0].labels).shape[1], 1
    )

    return encoder, rows, batch


def verify_model(checks, base, encoder, batch, root):
    model = build_model(
        str(base),
        encoder.label2id,
        encoder.id2label,
        architecture="multihead",
        entity_labels=encoder.entity_labels,
    )
    calls = []
    handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    output = model(**batch)
    checks.equal("B L K 3 logits", tuple(output.logits.shape), (2, 16, 4, 3))
    checks.equal("single backbone forward", len(calls), 1)
    output.loss.backward()
    checks.check(
        "finite gradients",
        all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
    )
    model.eval()
    with torch.no_grad():
        output = model(**batch)
        losses = []
        for head in range(4):
            losses.append(
                torch.nn.functional.cross_entropy(
                    output.logits[:, :, head, :].reshape(-1, 3),
                    batch["labels"][:, :, head].reshape(-1),
                    ignore_index=-100,
                )
            )
        checks.check(
            "balanced loss equals per-head CE mean",
            torch.allclose(output.loss, torch.stack(losses).mean()),
        )
        masked = dict(batch, labels=torch.full_like(batch["labels"], -100))
        checks.equal("all-masked batch finite zero", model(**masked).loss.item(), 0.0)
        unequal = batch["labels"].clone()
        unequal[:, :, 0] = -100
        sparse = model(**dict(batch, labels=unequal))
        checks.check(
            "empty head excluded from head mean",
            torch.allclose(sparse.loss, torch.stack(losses[1:]).mean()),
        )
    model.zero_grad(set_to_none=True)
    model.train()
    model(**dict(batch, labels=torch.full_like(batch["labels"], -100))).loss.backward()
    checks.check(
        "all masked gradients zero",
        all(
            parameter.grad is None or parameter.grad.count_nonzero().item() == 0
            for parameter in model.parameters()
        ),
    )
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    checks.check(
        "activation checkpointing backward",
        torch.isfinite(model.classifier.weight.grad).all().item(),
    )
    model.backbone.gradient_checkpointing_disable()
    handle.remove()
    check_max_length(model, encoder.tokenizer, 32, str(base), report=False)
    checks.check("capacity forward", True)
    distil = root / "distil"
    config = DistilBertConfig(
        vocab_size=encoder.tokenizer.vocab_size, dim=32, hidden_dim=64, n_layers=1, n_heads=2
    )
    AutoModel.from_config(config).save_pretrained(distil)
    no_types = build_model(
        str(distil),
        encoder.label2id,
        encoder.id2label,
        architecture="multihead",
        entity_labels=encoder.entity_labels,
    )
    checks.check(
        "backbone without token types",
        torch.isfinite(
            no_types(**dict(batch, token_type_ids=torch.zeros_like(batch["input_ids"]))).loss
        ).item(),
    )
    for labels in [["A"], ["A", "B"]]:
        small = build_model(
            str(base),
            encoder.label2id,
            encoder.id2label,
            architecture="multihead",
            entity_labels=labels,
        )
        checks.equal(
            f"K={len(labels)}",
            small(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits.shape[2],
            len(labels),
        )
    if torch.cuda.is_available():
        cuda_batch = {k: v.cuda() for k, v in batch.items()}
        checks.check("CUDA forward", torch.isfinite(model.cuda()(**cuda_batch).loss).item())
        model.cpu()
    else:
        checks.skip("CUDA", "no CUDA device available")

    return model


def verify_metrics(checks, encoder, rows):
    short = rows[rows.doc_id != "empty"].reset_index(drop=True)
    logits = oracle(short)
    metrics = evaluate_predictions(short, logits, encoder.tokenizer, encoder.id2label)
    checks.equal("overlap exact macro", metrics["span_strict_macro_f1"], 1.0)
    checks.equal("overlap exact micro", metrics["span_strict_micro_f1"], 1.0)
    checks.equal("per head strict", metrics["span_strict_f1__FINDING"], 1.0)
    decoded = predicted_spans(
        short,
        logits,
        encoder.tokenizer,
        encoder.id2label,
        include_scores=True,
        texts={name: "Paciente con dolor abdominal intenso." for name in short.doc_id},
    )
    checks.equal("all overlapping spans", len(decoded), 32)
    checks.check("scores softmax", decoded.score.between(0, 1).all())
    checks.check(
        "surface from original offsets",
        (decoded.text.str.len() == decoded.end_span - decoded.start_span).all(),
    )
    gold = span_dataframe(
        [
            dict(filename="d", label="A", start_span=0, end_span=3, text="abc"),
            dict(filename="d", label="B", start_span=1, end_span=4, text="bcd"),
        ]
    )
    pred = pd.concat([gold.iloc[:1]] * 3, ignore_index=True)
    scored = strict_multihead_metrics(gold, pred, ["A", "B", "C"])
    checks.equal("macro includes zero-support heads", scored["span_strict_macro_f1"], 1 / 3)
    checks.equal("micro counts not mean F1", scored["span_strict_micro_f1"], 2 / 3)
    checks.equal("duplicate predictions count once", scored["span_strict_micro_tp"], 1)
    checks.equal("missing overlap counts FN", scored["span_strict_fn__B"], 1)
    with warnings.catch_warnings(record=True) as caught:
        scored = strict_multihead_metrics(gold, pred, ["A"])
        checks.equal("unknown labels explicit FN", scored["unknown_gold_entities"], 1)
        checks.check("unknown label warning", bool(caught))
    duplicate_rows = pd.concat([short, short], ignore_index=True)
    repeated = np.concatenate([logits * 0.1, logits])
    deduplicated = predicted_spans(
        duplicate_rows, repeated, encoder.tokenizer, encoder.id2label, include_scores=True
    )
    checks.equal("duplicate windows deduplicated", len(deduplicated), 32)
    checks.check("duplicate windows keep highest confidence", (deduplicated.score > 0.99).all())


def verify_learning(checks, encoder, rows, model):
    dataset = to_dataset(rows)
    batch = MultiHeadCollator(encoder.tokenizer)([dataset[i] for i in range(len(dataset))])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    model.eval()
    initial = model(**batch).loss.item()
    model.train()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    checks.check("joint optimization lowers loss", output.loss.item() < initial * 0.1)
    metrics = evaluate_predictions(rows, output.logits.numpy(), encoder.tokenizer, encoder.id2label)
    checks.equal("learns overlapping types simultaneously", metrics["span_strict_macro_f1"], 1.0)


def verify_training(checks, base, encoder, rows, model, root, corpus):
    from lab.ner import build_compute_metrics

    arguments = training_arguments(
        root / "direct",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        metric_for_best_model="span_strict_macro_f1",
        use_cpu=True,
        disable_tqdm=True,
    )
    result = train(
        model,
        encoder.tokenizer,
        rows,
        rows,
        arguments,
        compute_metrics=build_compute_metrics(rows, encoder.tokenizer, encoder.id2label),
        save_model=True,
        model_encoding=model_encoding(encoder, str(base), "multihead"),
        early_stopping_patience=1,
    )
    checks.equal("default Trainer executed training", result.trainer.state.global_step > 0, True)
    checks.check("per-epoch macro metric", "span_strict_macro_f1" in result.epoch_metrics)
    checks.check(
        "early stopping callback installed",
        any(
            type(callback).__name__ == "EarlyStoppingCallback"
            for callback in result.trainer.callback_handler.callbacks
        ),
    )
    checks.check(
        "selected best epoch persisted", result.trainer.state.best_model_checkpoint is not None
    )
    checks.check("early stopping stops before epoch cap", result.trainer.state.epoch < 5)
    saved = result.paths["best_model"]
    original = predict_logits(result.model, encoder.tokenizer, rows, device="cpu", batch_size=4)
    loaded, tokenizer, description = load_model(saved, device="cpu")
    restored = predict_logits(loaded, tokenizer, rows, device="cpu", batch_size=4)
    checks.check("saved logits roundtrip", np.allclose(original, restored, atol=1e-6))
    checks.equal("persisted ordered heads", description["entity_labels"], encoder.entity_labels)
    checks.equal(
        "low-level multihead arguments default",
        training_arguments(root / "arguments", architecture="multihead").metric_for_best_model,
        "span_strict_macro_f1",
    )
    # Equal-width left padding checks the Trainer's alignment without position shifts.
    equal_rows = rows[rows.doc_id != "empty"].reset_index(drop=True)
    tokenizer.padding_side = "left"
    left_logits = predict_logits(
        loaded, tokenizer, equal_rows, device="cpu", batch_size=4, pad_to_multiple_of=None
    )
    tokenizer.padding_side = "right"
    right_logits = predict_logits(
        loaded, tokenizer, equal_rows, device="cpu", batch_size=4, pad_to_multiple_of=None
    )
    checks.check("left padding prediction alignment", np.allclose(left_logits, right_logits))
    variable = rows.iloc[[0, -1]].reset_index(drop=True)
    tokenizer.padding_side = "left"
    padded = MultiHeadCollator(tokenizer)([to_dataset(variable)[i] for i in range(2)])
    with torch.no_grad():
        expected = loaded(**padded).logits.numpy()
    actual = predict_logits(loaded, tokenizer, variable, batch_size=2, device="cpu")
    for index, row in enumerate(variable.itertuples(index=False)):
        width = len(row.input_ids)
        checks.check(
            "left variable batch logits align with original offsets",
            np.allclose(actual[index, :width], expected[index, -width:], atol=1e-6),
        )
    tokenizer.padding_side = "right"
    # The saved config must suffice even if the original backbone location disappears.
    base.rename(root / "hidden-base")
    standalone, _, _ = load_model(saved, device="cpu")
    checks.check("standalone checkpoint", standalone.entity_labels == encoder.entity_labels)
    (root / "hidden-base").rename(base)
    prediction = predict_entities(
        saved,
        {"new": "Paciente con dolor abdominal intenso."},
        root / "predicted",
        device="cpu",
        batch_size=2,
    )
    parquet = pd.read_parquet(prediction.paths["predictions_parquet"])
    tsv = pd.read_csv(prediction.paths["predictions"], sep="\t", keep_default_na=False)
    checks.equal("TSV parquet span identity", tsv.label.tolist(), parquet.label.tolist())
    metadata_path = saved / "encoding.json"
    raw = metadata_path.read_text()
    broken = json.loads(raw)
    broken["num_heads"] += 1
    metadata_path.write_text(json.dumps(broken))
    checks.raises(
        "incompatible mapping rejected", ValueError, load_model, saved, match="head catalogue"
    )
    metadata_path.write_text(raw)
    reordered = json.loads(raw)
    labels = list(reversed(reordered["entity_labels"]))
    reordered["entity_labels"] = labels
    reordered["encoding"]["entity_labels"] = labels
    reordered["model"]["architecture_kwargs"]["entity_labels"] = labels
    reordered["head_to_label"] = dict(enumerate(labels))
    metadata_path.write_text(json.dumps(reordered))
    checks.raises(
        "head reordering rejected even with equal dimensions",
        ValueError,
        load_model,
        saved,
        match="head order",
    )
    metadata_path.write_text(raw)
    from safetensors.torch import load_file, save_file

    weight_path = saved / "model.safetensors"
    weight_bytes = weight_path.read_bytes()
    state = load_file(str(weight_path))
    state["classifier.weight"] = state["classifier.weight"][:-3].contiguous()
    save_file(state, str(weight_path))
    checks.raises(
        "incompatible head dimensions rejected",
        ValueError,
        load_model,
        saved,
        match="Incompatible multihead checkpoint",
    )
    weight_path.write_bytes(weight_bytes)
    from lab.ner.ensemble.io import describe_member

    checks.raises(
        "ensemble boundary explicit", ValueError, describe_member, saved, match="Multihead"
    )
    # Exercise the existing orchestrator and CLI dispatch using actual YAML.
    from lab.cli import main as cli_run

    split = prepare_dataset(
        root / "split", source_parquet=_write_corpus(corpus, root), validation_size=0.25
    )
    split_dir = split.split.paths["train"].parent
    import yaml

    example_dir = Path(__file__).resolve().parents[1] / "docs" / "ner" / "configs"
    configuration = yaml.safe_load((example_dir / "multihead-train.yaml").read_text())
    configuration.update(
        split_dir=str(split_dir),
        output_dir=str(root / "assessment"),
        base_model=str(base),
        max_length=32,
        track_resources=False,
    )
    configuration["training_arguments"].update(num_train_epochs=1, use_cpu=True, disable_tqdm=True)
    path = root / "train.yaml"
    path.write_text(yaml.safe_dump(configuration))
    checks.equal("CLI exit status", cli_run(["run", str(path)]), 0)
    manifest = json.loads((root / "assessment" / "run_manifest.json").read_text())
    checks.equal(
        "CLI macro default",
        manifest["training_arguments"]["metric_for_best_model"],
        "span_strict_macro_f1",
    )
    checkpoints = root / "assessment" / "best_model"
    model2, _, desc2 = load_model(checkpoints)
    checks.equal("orchestrator discovered heads", model2.entity_labels, encoder.entity_labels)
    infer_config = yaml.safe_load((example_dir / "multihead-predict.yaml").read_text())
    texts_dir = root / "texts"
    texts_dir.mkdir()
    (texts_dir / "new.txt").write_text("Paciente con dolor abdominal intenso.")
    infer_config.update(
        model_dir=str(checkpoints),
        documents=str(texts_dir),
        output_dir=str(root / "example-inference"),
        device="cpu",
    )
    infer_path = root / "predict.yaml"
    infer_path.write_text(yaml.safe_dump(infer_config))
    checks.equal("documented inference YAML executes", cli_run(["run", str(infer_path)]), 0)

    checks.equal(
        "fold summary macro",
        json.loads((root / "assessment" / "assessment_summary.json").read_text())[
            "metric_for_best_model"
        ],
        "span_strict_macro_f1",
    )
    from lab.core.split import split_paths

    variants = encode_variants(
        split_paths(split_dir), build_variants([str(base)], max_lengths=[32]), None, "es"
    )
    checks.equal(
        "HPO variants reuse head catalogue",
        next(iter(variants.values())).entity_labels,
        encoder.entity_labels,
    )
    return split_dir


def verify_legacy(checks, base, tokenizer, corpus, root):
    from lab.ner import build_compute_metrics

    encoder = Encoder(tokenizer, "SYMPTOM", "es", max_length=32)
    rows = encoder.encode(corpus)
    for architecture in ["linear", "crf"]:
        model = build_model(
            str(base), encoder.label2id, encoder.id2label, architecture=architecture
        )
        arguments = training_arguments(
            root / architecture, num_train_epochs=1, use_cpu=True, disable_tqdm=True
        )
        result = train(
            model,
            tokenizer,
            rows,
            rows,
            arguments,
            compute_metrics=build_compute_metrics(rows, tokenizer, encoder.id2label),
            save_model=True,
            model_encoding=model_encoding(encoder, str(base), architecture),
        )
        loaded, restored_tokenizer, description = load_model(result.paths["best_model"])
        before = predict_logits(result.model, tokenizer, rows, device="cpu")
        after = predict_logits(loaded, restored_tokenizer, rows, device="cpu")
        checks.check(
            f"{architecture} training checkpoint logits roundtrip",
            np.allclose(before, after, atol=1e-6),
        )
        checks.equal(f"{architecture} keeps 3D logits", after.ndim, 3)
        checks.equal(
            f"{architecture} keeps selection default",
            arguments.metric_for_best_model,
            "span_strict_f1",
        )
        checks.check(
            f"{architecture} checkpoint format unchanged", "format_version" not in description
        )


def verify_folds(checks, base, corpus, root):
    from lab.core.split import split_paths
    from lab.ner import train_model

    source = _write_corpus(corpus, root)
    prepared = prepare_dataset(root / "kfold-split", source_parquet=source, kfolds=3)
    split_dir = next(iter(prepared.split.paths.values())).parent
    paths = split_paths(split_dir, validation_index=1)
    validation = pd.read_parquet(paths["validation"])
    for index in validation.index:
        entities = json.loads(validation.loc[index, "entities_json"])
        if entities:
            entities.append({**entities[0], "label": "UNSEEN"})
            validation.loc[index, "entities_json"] = json.dumps(entities)
            validation.loc[index, "n_entities"] = len(entities)
    validation.to_parquet(paths["validation"], index=False)
    # No test file can be consulted to discover heads.
    Path(paths["test"]).unlink()
    assessment = train_model(
        split_dir,
        root / "fold-assessment",
        str(base),
        language="es",
        architecture="multihead",
        max_length=32,
        track_resources=False,
        save_model=True,
        training_arguments=dict(num_train_epochs=1, use_cpu=True, disable_tqdm=True),
    )
    labels = [summary["metadata"]["entity_labels"] for summary in assessment.summaries]
    checks.check("validation-only type cannot create head", "UNSEEN" not in labels[0])
    checks.check("same type permitted when in train next fold", "UNSEEN" in labels[1])
    checks.equal("two fold rotations", len(assessment.fold_metrics), 2)
    checks.check("macro aggregated across folds", "span_strict_macro_f1" in assessment.aggregate)
    checks.check(
        "unknown contributes false negatives",
        assessment.fold_metrics.iloc[0]["unknown_gold_entities"] > 0,
    )


def verify_hpo(checks, base, split_dir, root):
    import ray

    from lab.ner import search_hyperparameters
    from lab.ner.hpo.trial import top_k_epoch_mean
    from lab.ner.training.multihead import architecture_arguments

    for metric in [
        "span_strict_macro_f1",
        "span_strict_micro_f1",
        "span_strict_f1__SYMPTOM",
        "eval_loss",
    ]:
        settings = architecture_arguments({"metric_for_best_model": metric}, "multihead")
        checks.equal(
            "selection choice retained " + metric, settings["metric_for_best_model"], metric
        )
        checks.equal(
            "selection direction " + metric, settings["greater_is_better"], metric != "eval_loss"
        )
    checks.equal(
        "top epochs use requested macro",
        top_k_epoch_mean(
            pd.DataFrame({"span_strict_macro_f1": [0.1, 0.8, 0.5]}), "span_strict_macro_f1", k=2
        ),
        0.65,
    )
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        result = search_hyperparameters(
            split_dir,
            root / "sweep",
            [str(base)],
            language="es",
            architecture="multihead",
            max_lengths=[32],
            n_trials=2,
            seeds_per_trial=2,
            top_k_epochs=1,
            training_arguments=dict(
                num_train_epochs=1, use_cpu=True, disable_tqdm=True, per_device_train_batch_size=4
            ),
            search_space=dict(
                learning_rate=0.001,
                weight_decay=None,
                warmup_steps=None,
                effective_train_batch_size=4,
                lr_scheduler_type="linear",
            ),
            smoke_test=True,
            track_resources=False,
            report=False,
        )
        checks.equal("real Ray multihead sweep", len(result.trials), 2)
        checks.equal("HPO default macro", result.metric, "span_strict_macro_f1")
        checks.check("two seeds per trial", (result.trials.n_seeds == 2).all())
        checks.check("winner runnable architecture", result.best["architecture"] == "multihead")
        checks.check("winner automatic catalogue", result.best["target_label"] is None)
        from lab.cli import main

        checks.equal("winner YAML runs unchanged", main(["run", str(result.paths["winner"])]), 0)
    finally:
        ray.shutdown()


def _write_corpus(corpus, root):
    path = root / "corpus.parquet"
    corpus.to_parquet(path, index=False)
    return path


def main():
    torch.set_num_threads(2)
    checks = Checks("multihead")
    tokenizer = local_tokenizer()
    corpus = overlap_corpus()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = tiny_base_model(root, tokenizer)
        encoder, rows, batch = verify_encoding(checks, tokenizer, corpus)
        model = verify_model(checks, base, encoder, batch, root)
        verify_metrics(checks, encoder, rows)
        verify_learning(checks, encoder, rows, model)
        split_dir = verify_training(checks, base, encoder, rows, model, root, corpus)
        verify_legacy(checks, base, tokenizer, corpus, root)
        verify_folds(checks, base, corpus, root)
        verify_hpo(checks, base, split_dir, root)
    return checks.report()


if __name__ == "__main__":
    run(main)
