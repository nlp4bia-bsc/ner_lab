"""Offline numeric, BIO, native pipeline and HPO checks for independent binary NER."""

from __future__ import annotations

import itertools
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from _harness import Checks, run
from fixtures import local_tokenizer, tiny_base_model
from torch.nn import functional as F
from verify_multihead import overlap_corpus

from lab.core import prepare_dataset, write_corpus
from lab.ner import (
    Encoder,
    build_compute_metrics,
    build_model,
    evaluate_predictions,
    train_model,
    training_arguments,
)
from lab.ner.encoding.encoder import describe_encoder, encoder_from_description
from lab.ner.evaluation.spans import gold_spans, predicted_spans
from lab.ner.hpo.decoding import fit_binary_decoding
from lab.ner.inference import load_model, predict_entities, predict_logits
from lab.ner.models.binary_decoding import (
    BinaryDecoding,
    adapted_probabilities,
    binary_probabilities,
    decode_binary,
    legal_log_scores,
)
from lab.ner.models.binary_loss import BinaryLoss, binary_loss, estimate_positive_weights
from lab.ner.training import model_encoding, train
from lab.ner.training.binary import binary_architecture_kwargs
from lab.ner.training.collation import classification_collator
from lab.ner.training.dataset import to_dataset


def verify_losses(checks):
    torch.manual_seed(31)
    for heads in (1, 2, 4):
        logits = torch.randn(2, 5, heads, 2, requires_grad=True)
        labels = torch.zeros_like(logits)
        labels[0, 1, 0, 0] = 1
        labels[1, 2, 0, 1] = 1
        labels[:, 0] = -100
        if heads > 1:
            labels[1, :, 1] = -100
        valid = labels != -100
        targets = labels.clamp_min(0)
        reference = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        mean = lambda values: ((values * valid).sum((0, 1, 3)) / valid.sum((0, 1, 3))).mean()
        checks.check(
            "BCE reference",
            torch.allclose(binary_loss(logits, labels, BinaryLoss()), mean(reference)),
        )
        checks.check(
            "focal gamma zero equals BCE",
            torch.allclose(
                binary_loss(logits, labels, BinaryLoss(loss="focal", focal_gamma=0)),
                mean(reference),
            ),
        )
        probability = logits.sigmoid()
        chosen = torch.where(targets.bool(), probability, 1 - probability)
        focal = reference * (1 - chosen).pow(2) * torch.where(targets.bool(), 0.25, 0.75)
        checks.check(
            "focal numerical reference",
            torch.allclose(
                binary_loss(logits, labels, BinaryLoss(loss="focal", focal_alpha=0.25)), mean(focal)
            ),
        )
        weights = torch.full((heads, 2), 3.0)
        weighted = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weights, reduction="none"
        )
        checks.check(
            "weighted BCE reference",
            torch.allclose(
                binary_loss(logits, labels, BinaryLoss(loss="weighted_bce"), weights),
                mean(weighted),
            ),
        )
        negative = (1 - probability + 0.05).clamp(max=1)
        asymmetric = torch.where(
            targets.bool(), -probability.log(), -negative.log() * (1 - negative).pow(4)
        )
        checks.check(
            "asymmetric reference",
            torch.allclose(
                binary_loss(logits, labels, BinaryLoss(loss="asymmetric")),
                mean(asymmetric),
                atol=1e-6,
            ),
        )
        checks.check(
            "asymmetric zero exponents/clip equals BCE",
            torch.allclose(
                binary_loss(logits, labels, BinaryLoss(loss="asymmetric", gamma_neg=0, clip=0)),
                mean(reference),
                atol=1e-6,
            ),
        )
        all_losses = [BinaryLoss(), BinaryLoss(loss="focal"), BinaryLoss(loss="asymmetric")]
        for option in all_losses:
            value = binary_loss(logits, labels, option)
            gradient = torch.autograd.grad(value, logits, retain_graph=True)[0]
            checks.check("finite gradients", torch.isfinite(gradient).all())
            checks.check("ignored gradients zero", not gradient[~valid].any())
            empty = binary_loss(logits, torch.full_like(labels, -100), option)
            checks.equal("empty supervision zero", empty.item(), 0.0)
            extreme = torch.tensor([-10000.0, 10000.0]).reshape(1, 1, 1, 2).requires_grad_()
            loss = binary_loss(extreme, torch.tensor([[[[1.0, 0.0]]]]), option)
            loss.backward()
            checks.check(
                "extreme logits finite", torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
            )
        for dtype in (torch.bfloat16, torch.float16):
            with torch.autocast("cpu", dtype=dtype):
                value = binary_loss(logits, labels, BinaryLoss())
            checks.equal("binary loss FP32", value.dtype, torch.float32)
        negative_only = torch.zeros_like(logits)
        checks.check(
            "negative-only heads contribute", binary_loss(logits, negative_only, BinaryLoss()) > 0
        )
    invalid = torch.tensor([[[[1.0, 1.0]]]])
    checks.raises(
        "incompatible binary gold",
        ValueError,
        binary_loss,
        torch.zeros_like(invalid),
        invalid,
        BinaryLoss(),
        match="simultaneously",
    )
    checks.raises(
        "loss options conditional", ValueError, BinaryLoss, loss="bce", gamma_neg=2, match="only"
    )
    checks.raises("unknown loss", ValueError, BinaryLoss, loss="categorical")

    for option in (BinaryLoss(loss="focal"), BinaryLoss(loss="asymmetric")):
        logits = torch.tensor([[[[0.7, -1.2]]]], requires_grad=True)
        labels = torch.tensor([[[[1.0, 0.0]]]])
        gradient = torch.autograd.grad(binary_loss(logits, labels, option), logits)[0]
        for channel in range(2):
            delta = torch.zeros_like(logits)
            delta[..., channel] = 0.001
            estimate = (
                binary_loss(logits.detach() + delta, labels, option)
                - binary_loss(logits.detach() - delta, labels, option)
            ) / 0.002
            checks.check(
                "binary objective finite difference gradient",
                torch.allclose(gradient[..., channel].sum(), estimate, atol=1e-4),
            )
    checks.raises(
        "partial channel masks rejected",
        ValueError,
        binary_loss,
        torch.zeros(1, 1, 1, 2),
        torch.tensor([[[[-100.0, 0.0]]]]),
        BinaryLoss(),
        match="share",
    )


def verify_decoders(checks):
    settings = BinaryDecoding()
    cases = [
        ([[0.01, 0.01], [0.01, 0.01]], [0, 0]),
        ([[0.9, 0.1], [0.1, 0.9]], [1, 2]),
        ([[0.1, 0.9], [0.01, 0.01], [0.1, 0.9]], [1, 0, 1]),
        ([[0.9, 0.1], [0.9, 0.1]], [1, 1]),
        ([[0.9, 0.9], [0.9, 0.9]], [1, 2]),
        ([[0.5, 0.5]], [1]),
    ]
    for probabilities, expected in cases:
        logits = torch.logit(torch.tensor(probabilities)).reshape(1, -1, 1, 2)
        mask = torch.ones(logits.shape[:2], dtype=torch.bool)
        checks.equal(
            "explicit threshold BIO path",
            decode_binary(logits, mask, settings, ["A"])[0, :, 0].tolist(),
            expected,
        )
    for activation in ("sigmoid", "temperature_sigmoid", "hard_sigmoid"):
        options = BinaryDecoding(
            activation=activation, temperature=2.0 if activation == "temperature_sigmoid" else 1.0
        )
        values = torch.linspace(-1000, 1000, 100).reshape(1, 50, 1, 2)
        scores = binary_probabilities(values, options, ["A"])
        checks.check("bounded sigmoid alternative", scores.ge(0).all() and scores.le(1).all())
        checks.check("monotone scores", (scores.flatten()[1:] >= scores.flatten()[:-1]).all())
        adapted = adapted_probabilities(values, options, ["A"])
        checks.check(
            "legal distribution normalized",
            torch.allclose(adapted.sum(-1), torch.ones(1, 50, 1), atol=2e-5),
        )
        checks.check("legal distribution finite", torch.isfinite(adapted).all())
    hard = binary_probabilities(
        torch.tensor([[[[-10.0, 10.0]]]]), BinaryDecoding(activation="hard_sigmoid"), ["A"]
    )
    checks.equal("hard sigmoid saturation", hard.flatten().tolist(), [0.0, 1.0])
    checks.raises(
        "nonpositive temperature",
        ValueError,
        BinaryDecoding,
        activation="temperature_sigmoid",
        temperature=0,
    )
    checks.raises("irrelevant temperature", ValueError, BinaryDecoding, temperature=2)
    for length in range(1, 5):
        logits = torch.randn(1, length, 2, 2)
        mask = torch.ones(1, length, dtype=torch.bool)
        options = BinaryDecoding(decoder="constrained_viterbi")
        decoded = decode_binary(logits, mask, options, ["A", "B"])
        scores = legal_log_scores(logits, options, ["A", "B"])
        for head in range(2):
            paths = [
                path
                for path in itertools.product(range(3), repeat=length)
                if path[0] != 2 and all(not (a == 0 and b == 2) for a, b in zip(path, path[1:]))
            ]
            best = max(
                paths,
                key=lambda path: sum(
                    scores[0, t, head, state].item() for t, state in enumerate(path)
                ),
            )
            checks.equal(
                "binary Viterbi exhaustive oracle", decoded[0, :, head].tolist(), list(best)
            )
    values = torch.zeros(1, 5, 3, 2)
    mask = torch.tensor([[False, True, False, True, False]])
    tags = decode_binary(values, mask, settings, ["A", "B", "C"])
    checks.equal("three simultaneous tracks", tags[0, 1].tolist(), [1, 1, 1])
    checks.equal("ignored subword not a BIO state", tags[0, 3].tolist(), [2, 2, 2])
    checks.check("absent positions explicit", (tags[~mask] == -100).all())
    for threshold, expected in ((0.0, [1, 2]), (1.0, [0, 0])):
        checks.equal(
            "threshold extremes",
            decode_binary(
                torch.zeros(1, 2, 1, 2),
                torch.ones(1, 2, dtype=torch.bool),
                BinaryDecoding(entity_threshold=threshold),
                ["A"],
            )[0, :, 0].tolist(),
            expected,
        )

    simultaneous = torch.full((1, 1, 1, 2), 1e6)
    conditioned = adapted_probabilities(simultaneous, BinaryDecoding(), ["A"])
    checks.check(
        "extreme simultaneous channels normalize",
        torch.allclose(conditioned.sum(-1), torch.ones(1, 1, 1), atol=1e-6),
    )
    p = torch.full((1, 1, 1, 2), 0.3)
    logits = torch.logit(p)
    checks.check(
        "legal conditioning formula",
        torch.allclose(
            adapted_probabilities(logits, BinaryDecoding(), ["A"]),
            torch.tensor([[[[7 / 13, 3 / 13, 3 / 13]]]]),
            atol=1e-6,
        ),
    )
    checks.equal(
        "independent union differs from conditioned entity mass",
        decode_binary(logits, torch.ones(1, 1, dtype=torch.bool), BinaryDecoding(), ["A"]).item(),
        1,
    )
    for decoder in ("threshold_bio", "constrained_viterbi"):
        empty = decode_binary(
            torch.empty(1, 0, 2, 2),
            torch.empty(1, 0, dtype=torch.bool),
            BinaryDecoding(decoder=decoder),
            ["A", "B"],
        )
        checks.equal("empty sequence decoder", tuple(empty.shape), (1, 0, 2))


def verify_encoding(checks, tokenizer, corpus):
    for strategy in ("greedy", "context"):
        encoder = Encoder(
            tokenizer,
            None,
            "es",
            max_length=24,
            label_schema="multilabel",
            strategy=strategy,
            context_tokens=2 if strategy == "context" else None,
        )
        from lab.ner.encoding.encoder import tokenize_document

        with patch(
            "lab.ner.encoding.encoder.tokenize_document", wraps=tokenize_document
        ) as tokenized:
            rows = encoder.encode(corpus)
        checks.equal("one tokenization per document", tokenized.call_count, len(corpus))
        checks.equal(
            "ordered train discovery",
            encoder.entity_labels,
            ["FINDING", "NESTED", "SYMPTOM", "TOKEN"],
        )
        plain = encoder.encode(corpus.assign(entities_json="[]", n_entities=0))
        for name in ("input_ids", "token_offsets", "word_ids", "word_mask"):
            checks.equal("gold independent structure", rows[name].tolist(), plain[name].tolist())
        for row in rows.itertuples():
            labels = np.asarray(row.labels)
            checks.equal("binary label dimensions", labels.shape, (len(row.input_ids), 4, 2))
            checks.check("specials ignored", (labels[[0, -1]] == -100).all())
            checks.check(
                "channel masks agree",
                np.array_equal(labels[..., 0] == -100, labels[..., 1] == -100),
            )
        oracle = np.asarray([row for row in rows[rows.doc_id != "empty"].labels])
        oracle = np.where(oracle == 1, 8.0, -8.0)
        short = rows[rows.doc_id != "empty"].reset_index(drop=True)
        metrics = evaluate_predictions(short, oracle, tokenizer, encoder.id2label)
        checks.equal("all overlaps reconstructed exactly", metrics["span_strict_macro_f1"], 1.0)
        checks.equal("micro exact", metrics["span_strict_micro_f1"], 1.0)
        checks.equal(
            "gold binary projection", len(gold_spans(rows, tokenizer, encoder.id2label)), 32
        )
        checks.equal(
            "binary score semantics bounded",
            predicted_spans(short, oracle, tokenizer, encoder.id2label, include_scores=True)
            .score.between(0, 1)
            .all(),
            True,
        )
        repeated = pd.concat([short, short], ignore_index=True)
        duplicate_predictions = np.concatenate([oracle * 0.5, oracle])
        deduplicated = predicted_spans(
            repeated, duplicate_predictions, tokenizer, encoder.id2label, include_scores=True
        )
        checks.equal("binary duplicate windows deduplicate exact identities", len(deduplicated), 32)
        checks.check(
            "binary duplicate windows retain greatest score", deduplicated.score.gt(0.99).all()
        )
        checks.equal(
            "encoder schema persisted",
            describe_encoder(encoder_from_description(describe_encoder(encoder), tokenizer)),
            describe_encoder(encoder),
        )
        document = corpus.iloc[0]
        entities = json.loads(document.entities_json)
        entities.append({**entities[0], "start": 14, "end": 34, "text": document.text[14:34]})
        checks.raises(
            "intraclass strict failure",
            ValueError,
            encoder.encode_document,
            document.doc_id,
            document.text,
            entities,
            match="Overlapping",
        )
    from lab.core import build_corpus

    text = "dolor abdominal"
    annotations = pd.DataFrame(
        [
            dict(
                filename="adjacent",
                label="SYMPTOM",
                start_span=start,
                end_span=end,
                text=text[start:end],
            )
            for start, end in ((0, 5), (6, 15))
        ]
    )
    adjacent = encoder.encode(build_corpus({"adjacent": text}, annotations))
    logits = np.where(np.asarray(adjacent.labels.tolist()) == 1, 8.0, -8.0)
    spans = predicted_spans(adjacent, logits, tokenizer, encoder.id2label)
    checks.equal(
        "adjacent same-type mentions stay distinct",
        list(spans[["start_span", "end_span"]].itertuples(index=False, name=None)),
        [(0, 5), (6, 15)],
    )
    shifted = pd.DataFrame(
        [dict(filename="shifted", label="SYMPTOM", start_span=1, end_span=4, text="olo")]
    )
    shifted_rows = encoder.encode(build_corpus({"shifted": "dolor"}, shifted))
    metrics = evaluate_predictions(
        shifted_rows,
        np.where(np.asarray(shifted_rows.labels.tolist()) == 1, 8.0, -8.0),
        tokenizer,
        encoder.id2label,
    )
    checks.equal("word-internal gold remains missed", metrics["span_strict_fn__SYMPTOM"], 1)
    checks.equal(
        "unrepresentable original gold is reported", metrics["gold_unrepresentable_entities"], 1
    )
    weights = estimate_positive_weights(rows, encoder.entity_labels, maximum=3)
    checks.check(
        "positive weighting bounded", all(1 <= v <= 3 for pair in weights.values() for v in pair)
    )
    return encoder, rows


def verify_pipeline(checks, base, tokenizer, corpus, root):
    encoder = Encoder(tokenizer, None, "es", max_length=32, label_schema="multilabel")
    rows = encoder.encode(corpus)
    model_dirs = []
    for loss in ("bce", "weighted_bce", "focal", "asymmetric"):
        settings = binary_architecture_kwargs(
            {"loss": loss, "entity_labels": encoder.entity_labels}, rows, encoder.entity_labels
        )
        if loss == "focal":
            settings.update(
                activation="temperature_sigmoid",
                temperature={"SYMPTOM": 1.3},
                decoder="constrained_viterbi",
                entity_bias={"SYMPTOM": 0.2},
            )
        elif loss == "asymmetric":
            settings.update(activation="hard_sigmoid", entity_threshold={"TOKEN": 0.6})
        model = build_model(str(base), encoder.label2id, encoder.id2label, "multilabel", **settings)
        dataset = to_dataset(rows)
        batch = classification_collator(model, tokenizer)([dataset[0], dataset[-1]])
        calls = []
        hook = model.backbone.register_forward_hook(lambda *_: calls.append(1))
        output = model(**batch)
        hook.remove()
        checks.equal("one shared backbone forward", len(calls), 1)
        checks.equal("logits B L K 2", tuple(output.logits.shape), tuple(batch["labels"].shape))
        checks.equal("config describes two binary channels", model.config.num_labels, 2)
        output.loss.backward()
        checks.check(
            "model finite gradients",
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
        )
        args = training_arguments(
            root / loss,
            architecture="multilabel",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            use_cpu=True,
            fp16=False,
            disable_tqdm=True,
            gradient_accumulation_steps=2,
            learning_rate=0.0 if loss == "bce" else 3e-5,
        )
        result = train(
            model,
            tokenizer,
            rows,
            rows,
            args,
            compute_metrics=build_compute_metrics(
                rows, tokenizer, encoder.id2label, multilabel_decoding=model.decoding
            ),
            save_model=True,
            model_encoding=model_encoding(encoder, str(base), "multilabel", settings),
            early_stopping_patience=1,
        )
        checks.equal("macro default", args.metric_for_best_model, "span_strict_macro_f1")
        checks.check(
            "best checkpoint selected", result.trainer.state.best_model_checkpoint is not None
        )
        if loss == "bce":
            checks.check("binary early stopping executes", result.trainer.state.epoch < 3)
        directory = result.paths["best_model"]
        model_dirs.append(directory)
        losses = []
        hook = model.register_forward_hook(lambda module, args, output: losses.append(output.loss))
        before = predict_logits(model, tokenizer, rows, device="cpu")
        hook.remove()
        checks.check(
            "inference omits dummy binary loss",
            bool(losses) and all(value is None for value in losses),
        )
        loaded, _, description = load_model(directory, device="cpu")
        after = predict_logits(loaded, tokenizer, rows, device="cpu")
        checks.check("raw logits roundtrip", np.allclose(before, after, atol=1e-6))
        checks.equal("binary metadata", description["channels"], {"B": 0, "I": 1})
        checks.equal("loss restored", loaded.loss_options.loss, loss)
        checks.equal(
            "nondefault decoding roundtrip", asdict(loaded.decoding), asdict(model.decoding)
        )
        from lab.ner.encoding.encoder import tokenize_document

        with patch(
            "lab.ner.encoding.encoder.tokenize_document", wraps=tokenize_document
        ) as tokenized:
            a = predict_entities(directory, corpus, root / (loss + "-ann"), device="cpu")
        checks.equal(
            "annotated inference tokenizes documents once", tokenized.call_count, len(corpus)
        )
        b = predict_entities(
            directory,
            corpus.assign(entities_json="[]", n_entities=0),
            root / (loss + "-plain"),
            device="cpu",
        )
        checks.frames_equal("annotated/plain predictions identical", a.spans, b.spans)
        checks.check(
            "canonical TSV and Parquet",
            a.paths["predictions"].exists() and a.paths["predictions_parquet"].exists(),
        )
        parquet = pd.read_parquet(a.paths["predictions_parquet"])
        tsv = pd.read_csv(a.paths["predictions"], sep="\t", keep_default_na=False)
        keys = ["filename", "label", "start_span", "end_span", "text"]
        checks.equal(
            "binary TSV/Parquet identities",
            list(tsv[keys].itertuples(index=False, name=None)),
            list(parquet[keys].itertuples(index=False, name=None)),
        )
        checks.check(
            "binary TSV/Parquet scores",
            np.allclose(tsv.score.to_numpy(dtype=float), parquet.score.to_numpy(dtype=float)),
        )
        tokenizer.padding_side = "left"
        left = predict_logits(loaded, tokenizer, rows.iloc[[0, -1]], device="cpu")
        checks.equal("left padded binary shape", left.shape[-2:], (4, 2))
        tokenizer.padding_side = "right"
        path = directory / "encoding.json"
        raw = path.read_text()
        broken = json.loads(raw)
        broken["channels"] = {"I": 0, "B": 1}
        path.write_text(json.dumps(broken))
        checks.raises(
            "channel mismatch rejected", ValueError, load_model, directory, match="schema"
        )
        path.write_text(raw)
        base.rename(root / "hidden-base")
        independent, _, _ = load_model(directory, device="cpu")
        checks.equal("standalone catalogue", independent.entity_labels, encoder.entity_labels)
        (root / "hidden-base").rename(base)
    return model_dirs, encoder, rows


def verify_fitting(checks, tokenizer, encoder, rows):
    short = rows[rows.doc_id != "empty"].reset_index(drop=True)
    gold = np.asarray(short.labels.tolist())
    logits = np.where(gold == 1, 2.0, -2.0).astype("float32")
    space = {
        "activation": {
            "type": "choice",
            "categories": ["sigmoid", "temperature_sigmoid", "hard_sigmoid"],
        },
        "decoder": {"type": "choice", "categories": ["threshold_bio", "constrained_viterbi"]},
        "temperature": {"type": "uniform", "low": 0.7, "high": 1.5},
        "entity_threshold": {"type": "uniform", "low": 0.2, "high": 0.8},
    }
    fitted = fit_binary_decoding(
        short,
        logits,
        tokenizer,
        encoder.entity_labels,
        space,
        n_trials=8,
        random_state=17,
        threshold_scope="entity",
        temperature_scope="entity",
        min_gold_entities=20,
    )
    checks.equal("cached decoder fitting macro", fitted.metrics["span_strict_macro_f1"], 1.0)
    checks.equal("all decoder trials", len(fitted.trials), 8)
    repeated = fit_binary_decoding(
        short,
        logits,
        tokenizer,
        encoder.entity_labels,
        space,
        n_trials=8,
        random_state=17,
        threshold_scope="entity",
        temperature_scope="entity",
        min_gold_entities=20,
    )
    checks.frames_equal("fixed seed trial trace", fitted.trials, repeated.trials)
    checks.equal("fixed seed reproducibility", asdict(fitted.settings), asdict(repeated.settings))
    checks.check(
        "rare label fallback",
        all(
            not isinstance(value, dict) or all(v == 0.5 for v in value.values())
            for value in fitted.trials.entity_threshold
        ),
    )

    fixed = BinaryDecoding(
        activation="temperature_sigmoid",
        temperature={"SYMPTOM": 1.3},
        entity_threshold={"SYMPTOM": 0.6},
    )
    preserved = fit_binary_decoding(
        short, logits, tokenizer, encoder.entity_labels, {}, baseline=fixed, n_trials=5
    )
    checks.equal("pinned decoder evaluated once", len(preserved.trials), 1)
    checks.equal(
        "named settings are not interpreted as search domains",
        asdict(preserved.settings),
        asdict(fixed),
    )
    constrained = fit_binary_decoding(
        short,
        logits,
        tokenizer,
        encoder.entity_labels,
        {
            "decoder": "constrained_viterbi",
            "activation": {"type": "choice", "categories": ["sigmoid", "temperature_sigmoid"]},
            "temperature": {"type": "uniform", "low": 0.5, "high": 2.0},
        },
        n_trials=2,
    )
    parameters = [json.loads(value) for value in constrained.trials.sampled_parameters_json]
    checks.check(
        "no equivalent Viterbi temperatures searched",
        all(not any(key.startswith("temperature") for key in item) for item in parameters),
    )


def verify_ensembles(checks, paths, base, tokenizer, corpus, root):
    from verify_unified_ensemble import save_member

    from lab.ner.ensemble import fit_ensemble, predict_ensemble
    from lab.ner.ensemble.tables import predict_member

    extras = [
        save_member(root / "legacy-linear", base, tokenizer, "linear", ["SYMPTOM"]),
        save_member(root / "legacy-crf", base, tokenizer, "crf", ["SYMPTOM"]),
        save_member(root / "multihead", base, tokenizer, "multihead", ["TOKEN", "SYMPTOM"]),
    ]
    from safetensors.torch import save_file

    from lab.ner.training.trainer import write_model_encoding

    encoder = Encoder(
        tokenizer,
        None,
        "es",
        max_length=32,
        entity_labels=["TOKEN", "SYMPTOM"],
        include_word_mask=True,
    )
    model = build_model(
        str(base),
        encoder.label2id,
        encoder.id2label,
        "multihead_crf",
        entity_labels=encoder.entity_labels,
    )
    target = root / "mh-crf"
    target.mkdir()
    save_file(model.state_dict(), str(target / "model.safetensors"))
    tokenizer.save_pretrained(target)
    metadata = model_encoding(encoder, str(base), "multihead_crf")
    metadata["backbone_config"] = model.config.to_dict()
    write_model_encoding(metadata, target)
    subset_encoder = Encoder(
        tokenizer,
        None,
        "es",
        max_length=32,
        entity_labels=["TOKEN", "SYMPTOM"],
        label_schema="multilabel",
    )
    subset_model = build_model(
        str(base),
        subset_encoder.label2id,
        subset_encoder.id2label,
        "multilabel",
        entity_labels=subset_encoder.entity_labels,
    )
    subset_dir = root / "subset-binary"
    subset_dir.mkdir()
    save_file(subset_model.state_dict(), str(subset_dir / "model.safetensors"))
    tokenizer.save_pretrained(subset_dir)
    subset_meta = model_encoding(subset_encoder, str(base), "multilabel")
    subset_meta["model"]["architecture_kwargs"] = subset_model.architecture_arguments()
    subset_meta["backbone_config"] = subset_model.config.to_dict()
    write_model_encoding(subset_meta, subset_dir)
    extras.append(subset_dir)
    extras.append(target)
    for method in ("majority", "soft", "weighted"):
        kwargs = (
            {"votes": "soft", "weight_scope": "entity", "threshold_scope": "entity"}
            if method == "weighted"
            else {}
        )
        fitted = fit_ensemble(
            [paths[0], paths[1], *extras],
            method,
            root / ("ensemble-" + method),
            validation=corpus,
            device="cpu",
            n_trials=2,
            compare_methods=False,
            **kwargs,
        )
        result = predict_ensemble(
            fitted.output_dir, corpus, root / ("pred-" + method), device="cpu"
        )
        checks.check("all five architectures combine", "score" in result.spans)
        checks.equal("ensemble macro default", fitted.ensemble["metric"], "span_strict_macro_f1")
    member = predict_member(paths[0], corpus, device="cpu")
    checks.check(
        "adapted soft probabilities normalized",
        np.allclose(member.words[["p_O", "p_B", "p_I"]].sum(1), 1, atol=1e-6),
    )
    checks.check("no negative O probability", member.words.p_O.ge(0).all())


def verify_backbones(checks, tokenizer):
    from transformers import DistilBertConfig

    from lab.ner.models.capacity import check_max_length

    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=24,
        hidden_dim=48,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=32,
    )
    for heads in (1, 2, 4):
        model = build_model(
            "unused",
            {"O": 0, "B": 1, "I": 2},
            {0: "O", 1: "B", 2: "I"},
            "multilabel",
            entity_labels=[f"E{k}" for k in range(heads)],
            backbone_config=config.to_dict(),
        )
        batch = {
            "input_ids": torch.ones(2, 8, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "token_type_ids": torch.zeros(2, 8, dtype=torch.long),
            "word_mask": torch.ones(2, 8, dtype=torch.bool),
            "labels": torch.zeros(2, 8, heads, 2),
        }
        for dtype in (torch.float16, torch.bfloat16):
            model.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=dtype):
                result = model(**batch)
            result.loss.backward()
            checks.check("no token-type backbone under autocast", torch.isfinite(result.loss))
            checks.check(
                "mixed precision model gradients",
                all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
            )
        check_max_length(model, tokenizer, 32, "binary-distilbert", report=False)
        checks.equal("multiple head capacity", result.logits.shape[2], heads)
        batch["labels"][:, :, -1] = -100
        checks.check("unsupervised head safe", torch.isfinite(model(**batch).loss))


def verify_learning(checks, base, tokenizer, corpus):
    encoder = Encoder(tokenizer, None, "es", max_length=32, label_schema="multilabel")
    rows = encoder.encode(corpus)
    model = build_model(
        str(base),
        encoder.label2id,
        encoder.id2label,
        "multilabel",
        entity_labels=encoder.entity_labels,
        dropout=0.0,
    )
    dataset = to_dataset(rows)
    batch = classification_collator(model, tokenizer)([dataset[i] for i in range(len(dataset))])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    initial = model(**batch).loss.item()
    model.train()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        result = model(**batch)
    checks.check("joint binary optimization learns", result.loss.item() < initial * 0.1)
    metrics = evaluate_predictions(rows, result.logits.numpy(), tokenizer, encoder.id2label)
    checks.equal("learns simultaneous overlapping spans", metrics["span_strict_macro_f1"], 1.0)


def verify_hpo(checks, base, tokenizer, corpus, checkpoint, root):
    import ray

    from lab.cli import main as cli_run
    from lab.ner import search_hyperparameters
    from lab.ner.hpo.binary import loss_family_space
    from lab.ner.hpo.space import build_search_space
    from lab.ner.hpo.trial import validate_architecture_dimensions

    checks.raises(
        "conditional parameter rejected for wrong loss",
        ValueError,
        validate_architecture_dimensions,
        build_search_space({"architecture__gamma_pos": 2}, defaults={}),
        "multilabel",
        {},
        match="fixed loss",
    )
    checks.raises(
        "profiles cannot form invalid Cartesian product",
        ValueError,
        loss_family_space,
        build_search_space({"architecture__gamma_pos": 2}, defaults={}),
        [{"loss": "asymmetric"}],
        match="inside profiles",
    )
    source = write_corpus(corpus, root / "hpo-corpus.parquet")
    prepared = prepare_dataset(root / "hpo-split", source_parquet=source, validation_size=0.25)
    split_dir = prepared.split.paths["train"].parent
    from lab.core.split import split_paths
    from lab.ner.hpo.trial import run_trial
    from lab.ner.hpo.variants import build_variants, encode_variants

    variants = encode_variants(
        split_paths(split_dir),
        build_variants([str(base)], max_lengths=[32]),
        None,
        "es",
        label_schema="multilabel",
    )
    arguments = training_arguments(
        root / "loss-trials",
        architecture="multilabel",
        num_train_epochs=1,
        use_cpu=True,
        fp16=False,
        disable_tqdm=True,
        save_strategy="no",
        load_best_model_at_end=False,
    )
    for loss in ("bce", "weighted_bce", "focal", "asymmetric"):
        trial = run_trial(
            {"variant": next(iter(variants)), "loss_variant": json.dumps({"loss": loss})},
            variants,
            arguments,
            root / ("trial-" + loss),
            architecture="multilabel",
            seeds_per_trial=1,
            early_stopping_patience=None,
            track_resources=False,
        )
        checks.check("native HPO scores loss " + loss, np.isfinite(trial["span_strict_macro_f1"]))
        checks.equal("native loss trial seed count", trial["n_seeds"], 1)
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        sweep = search_hyperparameters(
            split_dir,
            root / "hpo",
            [str(base)],
            architecture="multilabel",
            language="es",
            max_lengths=[32],
            n_trials=2,
            seeds_per_trial=2,
            loss_variants=[
                {"loss": "bce"},
                {"loss": "focal", "focal_gamma": 1.5},
                {"loss": "asymmetric", "gamma_neg": 2},
                {"loss": "weighted_bce", "max_pos_weight": 5},
            ],
            search_space={
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "effective_train_batch_size": 4,
                "lr_scheduler_type": "linear",
            },
            training_arguments={
                "num_train_epochs": 1,
                "use_cpu": True,
                "fp16": False,
                "disable_tqdm": True,
            },
            track_resources=False,
            report=False,
            early_stopping_patience=None,
        )
        checks.equal("Ray binary trials", len(sweep.trials), 2)
        checks.check("Ray multiple seeds", (sweep.trials.n_seeds == 2).all())
        checks.equal("Ray macro objective", sweep.metric, "span_strict_macro_f1")
        checks.check(
            "winner preserves loss profile",
            sweep.best["architecture_kwargs"]["loss"]
            in ("bce", "focal", "asymmetric", "weighted_bce"),
        )
        checks.equal("winner YAML executes", cli_run(["run", str(sweep.paths["winner"])]), 0)
        import yaml

        templates = Path(__file__).resolve().parents[1] / "docs" / "ner" / "configs"
        configuration = yaml.safe_load((templates / "multilabel-hpo.yaml").read_text())
        configuration.update(
            split_dir=str(split_dir),
            output_dir=str(root / "hpo-template"),
            base_models=[str(base)],
            max_lengths=[32],
            n_trials=1,
            seeds_per_trial=1,
            smoke_test=False,
            track_resources=False,
            report=False,
            early_stopping_patience=None,
            training_arguments={
                "num_train_epochs": 1,
                "use_cpu": True,
                "fp16": False,
                "disable_tqdm": True,
            },
        )
        configuration["search_space"].update(
            learning_rate=0.001,
            weight_decay=0.0,
            warmup_steps=0,
            lr_scheduler_type="linear",
            effective_train_batch_size=4,
        )
        file = root / "hpo-template.yaml"
        file.write_text(yaml.safe_dump(configuration))
        checks.equal("documented training HPO CLI", cli_run(["run", str(file)]), 0)
    finally:
        ray.shutdown()
    calibration = corpus.copy()
    calibration["doc_id"] = "calibration-" + calibration.doc_id
    calibration["text"] = calibration.text + " Otro."
    calibration_source = write_corpus(calibration, root / "calibration-corpus.parquet")
    calibration_split = prepare_dataset(
        root / "calibration-split", source_parquet=calibration_source, validation_size=0.25
    )
    split_dir = calibration_split.split.paths["train"].parent
    summary_path = checkpoint.parent / "training_summary.json"
    summary_raw = summary_path.read_text()
    summary = json.loads(summary_raw)
    summary["metadata"] = {"train_parquet": str(calibration_split.split.paths["validation"])}
    summary_path.write_text(json.dumps(summary))
    checks.raises(
        "known calibration/training overlap rejected",
        ValueError,
        search_hyperparameters,
        split_dir,
        root / "leaky-decoder",
        model_dir=checkpoint,
        n_trials=1,
        decoder_search_space={"entity_threshold": 0.5},
        match="overlaps training",
    )
    summary_path.write_text(summary_raw)
    from lab.ner.hpo.decoder_search import predict_logits as infer

    with patch("lab.ner.hpo.decoder_search.predict_logits", wraps=infer) as calls:
        selected = search_hyperparameters(
            split_dir,
            root / "decoder-search",
            model_dir=checkpoint,
            n_trials=4,
            random_state=19,
            decoder_search_space={
                "activation": {
                    "type": "choice",
                    "categories": ["sigmoid", "temperature_sigmoid", "hard_sigmoid"],
                },
                "temperature": {"type": "uniform", "low": 0.8, "high": 1.2},
                "entity_threshold": {"type": "uniform", "low": 0.3, "high": 0.7},
            },
            threshold_scope="entity",
            temperature_scope="entity",
            min_gold_entities=1,
        )
    checks.equal("all post-hoc trials reuse one inference", calls.call_count, 1)
    checks.check("post-hoc choices persisted", selected.paths["decoder"].exists())
    prediction = predict_entities(
        checkpoint,
        corpus,
        root / "selected-predictions",
        decoder_path=selected.paths["decoder"],
        device="cpu",
    )
    checks.check("checkpoint-bound decoder inference", prediction.paths["predictions"].exists())
    from lab.ner.inference.binary_settings import read_binary_decoding

    description = json.loads(selected.paths["decoder"].read_text())
    description["model_fingerprint"] = "broken"
    broken = root / "broken-decoder.json"
    broken.write_text(json.dumps(description))
    checks.raises(
        "post-hoc checkpoint binding",
        ValueError,
        read_binary_decoding,
        broken,
        checkpoint,
        json.loads((checkpoint / "encoding.json").read_text())["entity_labels"],
        match="disagrees",
    )
    from verify_multihead import verify_folds

    fold_root = root / "binary-folds"
    fold_root.mkdir()
    with patch(
        "lab.ner.train_model",
        side_effect=lambda *a, **k: train_model(*a, **{**k, "architecture": "multilabel"}),
    ):
        verify_folds(checks, base, corpus, fold_root)


def verify_yaml(checks, base, corpus, paths, root):
    import inspect

    import yaml

    from lab.cli import main as cli_run
    from lab.ner import search_hyperparameters
    from lab.ner.ensemble import fit_ensemble

    directory = Path(__file__).resolve().parents[1] / "docs" / "ner" / "configs"
    functions = {
        "ner.train_model": train_model,
        "ner.predict_entities": predict_entities,
        "ner.search_hyperparameters": search_hyperparameters,
        "ner.fit_ensemble": fit_ensemble,
    }
    names = (
        "multilabel-train",
        "multilabel-focal",
        "multilabel-asymmetric",
        "multilabel-hpo",
        "multilabel-decoder-hpo",
        "multilabel-predict",
        "ensemble-multilabel",
    )
    configs = {name: yaml.safe_load((directory / f"{name}.yaml").read_text()) for name in names}
    for name, config in configs.items():
        inspect.signature(functions[config["task"]]).bind(
            **{k: v for k, v in config.items() if k != "task"}
        )
        checks.check("documented YAML signature " + name, True)
    root.mkdir()
    source = write_corpus(corpus, root / "corpus.parquet")
    prepared = prepare_dataset(root / "split", source_parquet=source, validation_size=0.25)
    for name in names[:3]:
        config = configs[name]
        config.update(
            split_dir=str(prepared.split.paths["train"].parent),
            output_dir=str(root / name),
            base_model=str(base),
            max_length=32,
            track_resources=False,
        )
        config["training_arguments"].update(
            num_train_epochs=1, use_cpu=True, fp16=False, disable_tqdm=True
        )
        file = root / f"{name}.yaml"
        file.write_text(yaml.safe_dump(config))
        checks.equal("documented training CLI " + name, cli_run(["run", str(file)]), 0)
    config = configs["multilabel-predict"]
    config.update(
        model_dir=str(root / "multilabel-train" / "best_model"),
        documents=str(source),
        output_dir=str(root / "predictions"),
        device="cpu",
    )
    file = root / "predict.yaml"
    file.write_text(yaml.safe_dump(config))
    checks.equal("documented binary inference CLI", cli_run(["run", str(file)]), 0)
    config = configs["ensemble-multilabel"]
    config.update(
        model_dirs=[str(path) for path in paths[:2]],
        validation=str(source),
        output_dir=str(root / "ensemble"),
        n_trials=2,
        compare_methods=False,
        device="cpu",
    )
    file = root / "ensemble.yaml"
    file.write_text(yaml.safe_dump(config))
    checks.equal("documented ensemble CLI", cli_run(["run", str(file)]), 0)
    metadata = json.loads((root.parent / "decoder-search" / "run_manifest.json").read_text())
    config = configs["multilabel-decoder-hpo"]
    config.update(
        split_dir=metadata["split_dir"],
        output_dir=str(root / "decoder-cli"),
        model_dir=str(paths[0]),
        n_trials=2,
        min_gold_entities=1,
    )
    file = root / "decoder.yaml"
    file.write_text(yaml.safe_dump(config))
    checks.equal("documented decoder HPO CLI", cli_run(["run", str(file)]), 0)


def main():
    torch.set_num_threads(2)
    checks = Checks("native multilabel sigmoid NER")
    verify_losses(checks)
    verify_decoders(checks)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tokenizer = local_tokenizer()
        corpus = overlap_corpus()
        base = tiny_base_model(root, tokenizer)
        verify_backbones(checks, tokenizer)
        encoder, rows = verify_encoding(checks, tokenizer, corpus)
        verify_learning(checks, base, tokenizer, corpus)
        verify_fitting(checks, tokenizer, encoder, rows)
        paths, encoder, rows = verify_pipeline(checks, base, tokenizer, corpus, root)
        verify_ensembles(checks, paths, base, tokenizer, corpus, root)
        verify_hpo(checks, base, tokenizer, corpus, paths[0], root)
        verify_yaml(checks, base, corpus, paths, root / "yaml")
    if not torch.cuda.is_available():
        checks.skip("CUDA", "CPU-only environment")
    return checks.report()


if __name__ == "__main__":
    run(main)
