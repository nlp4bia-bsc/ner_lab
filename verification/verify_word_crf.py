"""Offline integration checks for compact classic and Multi-Head constrained CRFs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from _harness import Checks, run
from fixtures import local_tokenizer, tiny_base_model
from verify_multihead import overlap_corpus

from lab.core import prepare_dataset, write_corpus
from lab.ner import Encoder, build_compute_metrics, build_model, training_arguments
from lab.ner.encoding.supervision import validate_word_supervision
from lab.ner.ensemble import fit_ensemble, predict_ensemble
from lab.ner.ensemble.tables import predict_member
from lab.ner.inference import load_model, predict_entities, predict_logits
from lab.ner.models.chain import ConstrainedChain
from lab.ner.models.packing import pack_gold, pack_words
from lab.ner.training import model_encoding, train, train_model
from lab.ner.training.collation import classification_collator
from lab.ner.training.dataset import to_dataset
from lab.ner.training.trainer import WordCRFTrainer, WordMarginalTrainer


def verify_packing(checks):
    for heads in (1, 2, 4):
        emissions = torch.randn(3, 8, heads, 3, requires_grad=True)
        mask = torch.zeros(3, 8, heads, dtype=torch.bool)
        mask[0, [1, 3, 6]] = True
        mask[1, [2, 5]] = True
        packed = pack_words(emissions, mask)
        restored = packed.restore(packed.emissions, absent=0)
        checks.check(
            "packing preserves eligible values", torch.equal(restored[mask], emissions[mask])
        )
        checks.check("absent emissions explicit", (restored[~mask] == 0).all().item())
        packed.emissions.sum().backward()
        checks.check("packing gradients", torch.equal(emissions.grad.sum(-1), mask.float() * 3))
        chain = ConstrainedChain({0: "O", 1: "B", 2: "I"}, heads)
        with torch.no_grad():
            for parameter in chain.parameters():
                parameter.normal_()
        whole = chain.log_partition(packed.emissions, packed.mask)
        for head in range(heads):
            independent = ConstrainedChain({0: "O", 1: "B", 2: "I"})
            with torch.no_grad():
                for name, parameter in independent.named_parameters():
                    parameter.copy_(dict(chain.named_parameters())[name][head : head + 1])
            part = independent.log_partition(packed.emissions[:, :, head], packed.mask[:, :, head])
            checks.check(
                "independent chains equal", torch.allclose(whole[:, head], part, atol=2e-6)
            )
            checks.check(
                "independent Viterbi equal",
                torch.equal(
                    chain.decode(packed.emissions, packed.mask)[:, :, head],
                    independent.decode(packed.emissions[:, :, head], packed.mask[:, :, head]),
                ),
            )
        gold = torch.zeros(3, 8, heads, dtype=torch.long)
        gold[0, 3, 0] = -100
        checks.raises(
            "unknown whole word rejected", ValueError, pack_gold, gold, packed, match="Unobserved"
        )
        gold.zero_()
        a = chain(packed.emissions, pack_gold(gold, packed), packed.mask)
        b = chain(packed.emissions, pack_gold(gold, packed), packed.mask, torch.full((heads,), 0.1))
        checks.check("weights normalize below one", torch.allclose(a, b, atol=1e-6))
    zero = torch.randn(1, 4, 1, 3, requires_grad=True)
    packed = pack_words(zero, torch.zeros(1, 4, dtype=torch.bool))
    chain = ConstrainedChain({0: "O", 1: "B", 2: "I"})
    loss = chain(packed.emissions, torch.zeros_like(packed.mask, dtype=torch.long), packed.mask)
    loss.backward()
    checks.check("empty packed batch differentiable", zero.grad is not None and not zero.grad.any())


def verify_encoding(checks, tokenizer, corpus):
    for strategy in ("greedy", "context"):
        encoder = Encoder(
            tokenizer,
            None,
            "es",
            max_length=16,
            strategy=strategy,
            context_tokens=2 if strategy == "context" else None,
            include_word_mask=True,
        )
        rows = encoder.encode(corpus)
        plain = encoder.encode(corpus.assign(entities_json="[]", n_entities=0))
        for key in ("input_ids", "attention_mask", "token_offsets", "word_ids", "crf_word_mask"):
            checks.equal(
                f"{strategy} annotation independent {key}", rows[key].tolist(), plain[key].tolist()
            )
        validate_word_supervision(rows, encoder.entity_labels)
        for row in rows.itertuples():
            checks.check(
                "special tokens absent from chain",
                not row.crf_word_mask[0] and not row.crf_word_mask[-1],
            )
            checks.check(
                "mask equals structural label support",
                np.array_equal(row.crf_word_mask, np.asarray(row.labels)[:, 0] != -100),
            )
        broken = rows.copy(deep=True)
        labels = np.asarray(broken.iloc[0].labels).copy()
        labels[np.flatnonzero(broken.iloc[0].crf_word_mask)[0], 0] = 2
        broken.at[0, "labels"] = labels.tolist()
        checks.raises(
            "invalid gold includes document diagnostic",
            ValueError,
            validate_word_supervision,
            broken,
            encoder.entity_labels,
            match="document=",
        )
    return encoder, rows


def verify_model(checks, base, tokenizer, corpus, root):
    saved = []
    for architecture in ("crf", "multihead_crf"):
        encoder = Encoder(
            tokenizer,
            "SYMPTOM" if architecture == "crf" else None,
            "es",
            max_length=32,
            include_word_mask=True,
        )
        rows = encoder.encode(corpus)
        kwargs = {"entity_labels": encoder.entity_labels} if encoder.target_label is None else {}
        model = build_model(
            str(base), encoder.label2id, encoder.id2label, architecture, dropout=0.0, **kwargs
        )
        dataset = to_dataset(rows)
        batch = classification_collator(model, tokenizer)([dataset[0], dataset[-1]])
        calls = []
        hook = model.backbone.register_forward_hook(lambda *_: calls.append(1))
        output = model(**batch)
        hook.remove()
        checks.equal("one backbone per joint forward", len(calls), 1)
        expected = (
            (*batch["input_ids"].shape, len(model.entity_labels), 3)
            if model.multihead
            else (*batch["input_ids"].shape, 3)
        )
        checks.equal("emissions shape", tuple(output.logits.shape), expected)
        output.loss.backward()
        checks.check(
            "finite gradients",
            all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
        )
        checks.check(
            "shared encoder gradients",
            model.backbone.embeddings.word_embeddings.weight.grad.abs().sum() > 0,
        )
        checks.check(
            "every head gradients",
            model.classifier.weight.grad.reshape(-1, 3, 32).abs().sum((1, 2)).gt(0).all(),
        )
        structured = model.structured_from_emissions(
            output.logits.detach(),
            batch["crf_word_mask"],
            batch["attention_mask"],
            probabilities=True,
        )
        eligible = (
            batch["crf_word_mask"][:, :, None].expand_as(structured.tags)
            if model.multihead
            else batch["crf_word_mask"]
        )
        checks.check("restored absent tags", (structured.tags[~eligible] == -100).all())
        checks.check(
            "real marginals normalized",
            torch.allclose(
                structured.log_marginals.exp().sum(-1)[eligible],
                torch.ones_like(structured.tags[eligible], dtype=torch.float),
                atol=2e-6,
            ),
        )
        checks.check(
            "absent probabilities zero", (structured.log_marginals.exp()[~eligible] == 0).all()
        )
        model.eval()
        with torch.no_grad():
            initial = model(**batch).loss
            model.aux_ce_weight = 0.3
            auxiliary = model(**batch).loss
        checks.check(
            "auxiliary CE adds finite local loss", auxiliary > initial and torch.isfinite(auxiliary)
        )
        model.aux_ce_weight = 0.0
        args = training_arguments(
            root / architecture,
            architecture=architecture,
            num_train_epochs=4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            use_cpu=True,
            fp16=False,
            disable_tqdm=True,
            report_to="none",
        )
        result = train(
            model,
            tokenizer,
            rows,
            rows,
            args,
            compute_metrics=build_compute_metrics(rows, tokenizer, encoder.id2label),
            save_model=True,
            early_stopping_patience=1,
            model_encoding=model_encoding(encoder, str(base), architecture),
        )
        checks.check("Viterbi trainer selected", isinstance(result.trainer, WordCRFTrainer))
        checks.check(
            "best checkpoint chosen", result.trainer.state.best_model_checkpoint is not None
        )
        checks.check("early stopping executed", result.trainer.state.epoch < 4)
        directory = result.paths["best_model"]
        saved.append(directory)
        before = predict_logits(
            model, tokenizer, rows, device="cpu", trainer_class=WordMarginalTrainer
        )
        loaded, _, description = load_model(directory, device="cpu")
        after = predict_logits(
            loaded, tokenizer, rows, device="cpu", trainer_class=WordMarginalTrainer
        )
        checks.check("marginals and Viterbi roundtrip", np.allclose(before, after, atol=1e-6))
        checks.equal("CRF version persisted", description["crf_version"], 2)
        encoding_path = directory / "encoding.json"
        raw = encoding_path.read_text()
        broken = json.loads(raw)
        broken["model"]["architecture_kwargs"]["constrain_transitions"] = False
        encoding_path.write_text(json.dumps(broken))
        checks.raises(
            "changed structural constraints rejected",
            ValueError,
            load_model,
            directory,
            match="disagrees",
        )
        encoding_path.write_text(raw)
        # Loading is standalone, even with the pretrained backbone unavailable.
        base.rename(root / "hidden-backbone")
        standalone, _, _ = load_model(directory, device="cpu")
        checks.equal("standalone catalogue", standalone.entity_labels, model.entity_labels)
        (root / "hidden-backbone").rename(base)
        annotated = predict_entities(
            directory, corpus, root / (architecture + "-ann"), device="cpu"
        )
        unannotated = predict_entities(
            directory,
            corpus.assign(entities_json="[]", n_entities=0),
            root / (architecture + "-plain"),
            device="cpu",
        )
        checks.frames_equal(
            "gold cannot change inference spans or scores", annotated.spans, unannotated.spans
        )
        checks.check("span scores bounded", annotated.spans.score.between(0, 1).all())
        checks.equal(
            "TSV and parquet persisted",
            pd.read_parquet(annotated.paths["predictions_parquet"]).shape[0],
            len(pd.read_csv(annotated.paths["predictions"], sep="\t")),
        )
        member = predict_member(
            directory, corpus.assign(entities_json="[]", n_entities=0), device="cpu"
        )
        checks.check(
            "ensemble marginal probabilities",
            np.allclose(member.words[["p_O", "p_B", "p_I"]].sum(axis=1), 1, atol=2e-6),
        )
        checks.check(
            "ensemble soft stream not one-hot",
            ((member.words.p_O > 0) & (member.words.p_O < 1)).any(),
        )
        tokenizer.padding_side = "left"
        predictions = predict_logits(
            loaded, tokenizer, rows.iloc[[0, -1]], device="cpu", trainer_class=WordMarginalTrainer
        )
        checks.equal("left padding restored shape", predictions.shape[0], 2)
        tokenizer.padding_side = "right"
        # Probability score is read directly from forward-backward, not pseudo logits.
        if len(annotated.spans):
            checks.check("scores not artificial certainty", (annotated.spans.score < 1).any())
        # Frozen zero transition CE is the efficient constrained-decoding baseline.
        baseline = build_model(
            str(base),
            encoder.label2id,
            encoder.id2label,
            architecture,
            loss_type="cross_entropy",
            **kwargs,
        )
        checks.check(
            "CE baseline has no trainable transitions",
            all(not p.requires_grad for p in baseline.chain.parameters()),
        )
        checks.check("CE baseline backward", torch.isfinite(baseline(**batch).loss))
    return saved


def verify_orchestration(checks, base, corpus, root):
    source = write_corpus(corpus, root / "corpus.parquet")
    prepared = prepare_dataset(root / "split", source_parquet=source, validation_size=0.25)
    split_dir = prepared.split.paths["train"].parent
    result = train_model(
        split_dir,
        root / "assessment",
        str(base),
        None,
        language="es",
        architecture="multihead_crf",
        max_length=32,
        training_arguments={
            "num_train_epochs": 1,
            "use_cpu": True,
            "fp16": False,
            "disable_tqdm": True,
        },
        save_model=True,
        track_resources=False,
    )
    saved = result.run_dir / "best_model"
    checks.check("native train_model CRF checkpoint", saved.exists())
    manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
    checks.equal(
        "default strict macro selection",
        manifest["training_arguments"]["metric_for_best_model"],
        "span_strict_macro_f1",
    )
    from lab.core.split import split_paths
    from lab.ner.hpo.space import build_search_space
    from lab.ner.hpo.trial import run_trial, validate_architecture_dimensions
    from lab.ner.hpo.variants import build_variants, encode_variants
    from lab.ner.hpo.winner import winner_configuration

    cases = [
        ({"architecture__aux_ce_weight": 0.1}, "linear", "require version-2"),
        ({"architecture__tag_scheme": "BIOES"}, "crf", "must choose"),
        ({"architecture__aux_ce_weight": -0.1}, "crf", "nonnegative"),
        (
            {"architecture__loss_type": "cross_entropy", "architecture__aux_ce_weight": 0.1},
            "crf",
            "separate sweeps",
        ),
    ]
    for specification, architecture, message in cases:
        checks.raises(
            "invalid architecture search rejected before trials",
            ValueError,
            validate_architecture_dimensions,
            build_search_space(specification, defaults={}),
            architecture,
            match=message,
        )
    variants = encode_variants(
        split_paths(split_dir),
        build_variants([str(base)], max_lengths=[32]),
        None,
        "es",
        include_word_mask=True,
    )
    checks.check(
        "HPO variants retain eligibility",
        "crf_word_mask" in next(iter(variants.values())).train_rows,
    )
    sampled = {
        "variant": next(iter(variants)),
        "architecture__aux_ce_weight": 0.1,
        "architecture__constrain_transitions": True,
    }
    args = training_arguments(
        root / "hpo",
        architecture="multihead_crf",
        num_train_epochs=1,
        fp16=False,
        use_cpu=True,
        disable_tqdm=True,
        report_to="none",
        load_best_model_at_end=False,
        save_strategy="no",
    )
    trial = run_trial(
        sampled,
        variants,
        args,
        root / "trial",
        architecture="multihead_crf",
        seeds_per_trial=2,
        early_stopping_patience=None,
        track_resources=False,
    )
    checks.check("HPO trial finite", np.isfinite(trial["span_strict_macro_f1"]))
    winner = winner_configuration(
        split_dir,
        root / "winner",
        {"base_model": str(base), "strategy": "greedy", "context_tokens": None, "max_length": 32},
        sampled,
        4,
        1,
        None,
        "es",
        "multihead_crf",
        None,
        "error",
        4,
        40,
        None,
        args,
    )
    checks.equal(
        "winner architecture dimension routing", winner["architecture_kwargs"]["aux_ce_weight"], 0.1
    )
    checks.check(
        "no CRF args inside TrainingArguments",
        "architecture__aux_ce_weight" not in winner["training_arguments"],
    )
    return saved


def verify_ensembles(checks, paths, corpus, root):
    # Existing suites cover linear/legacy CRF; these routes add new word CRFs.
    for method in (
        "majority",
        "soft",
        "weighted",
        "span_union",
        "span_consensus",
        "span_intersection",
    ):
        kwargs = (
            {
                "votes": "soft",
                "weight_scope": "entity",
                "threshold_scope": "entity",
            }
            if method == "weighted"
            else {}
        )
        with patch(
            "lab.ner.ensemble.fit.predict_members",
            wraps=__import__(
                "lab.ner.ensemble.tables", fromlist=["predict_members"]
            ).predict_members,
        ) as inferred:
            fitted = fit_ensemble(
                paths,
                method,
                root / method,
                validation=corpus,
                n_trials=2,
                device="cpu",
                compare_methods=False,
                **kwargs,
            )
        checks.equal("Optuna reuses joint inference", inferred.call_count, 1)
        predicted = predict_ensemble(
            fitted.output_dir, corpus, root / (method + "-prediction"), device="cpu"
        )
        checks.check(
            "hybrid structured ensemble canonical",
            set(["filename", "label", "start_span", "end_span", "text", "score"])
            <= set(predicted.spans),
        )
        checks.equal(
            "ensemble multitype macro default", fitted.ensemble["metric"], "span_strict_macro_f1"
        )


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
            "multihead_crf",
            entity_labels=[f"E{k}" for k in range(heads)],
            backbone_config=config.to_dict(),
        )
        batch = {
            "input_ids": torch.ones(2, 8, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "token_type_ids": torch.zeros(2, 8, dtype=torch.long),
            "crf_word_mask": torch.tensor([[0, 1, 0, 1, 1, 0, 0, 0]] * 2, dtype=torch.bool),
            "labels": torch.zeros(2, 8, heads, dtype=torch.long),
        }
        result = model(**batch)
        result.loss.backward()
        checks.check("backbone without token types", torch.isfinite(result.loss))
        check_max_length(model, tokenizer, 32, "word-crf-distil", report=False)
        checks.equal("K dimension supported", result.logits.shape[2], heads)
        for dtype in (torch.bfloat16, torch.float16):
            with torch.autocast("cpu", dtype=dtype):
                value = model(**batch).loss
            checks.equal("CRF loss stays FP32 under autocast", value.dtype, torch.float32)
            checks.check("mixed precision finite", torch.isfinite(value))
            model.zero_grad(set_to_none=True)
            value.backward()
            checks.check(
                "autocast backward finite",
                all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),
            )
        # A head without supervision is represented by a structural empty chain.
        expanded = batch["crf_word_mask"][:, :, None].expand(2, 8, heads).clone()
        expanded[:, :, -1] = False
        batch["crf_word_mask"] = expanded
        batch["labels"][:, :, -1] = -100
        checks.check("inactive head safe", torch.isfinite(model(**batch).loss))


def verify_ray(checks, base, corpus, root):
    import ray

    from lab.cli import main as cli_run
    from lab.ner import search_hyperparameters

    source = write_corpus(corpus, root / "ray-corpus.parquet")
    prepared = prepare_dataset(root / "ray-split", source_parquet=source, validation_size=0.25)
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        result = search_hyperparameters(
            prepared.split.paths["train"].parent,
            root / "ray-hpo",
            [str(base)],
            None,
            language="es",
            architecture="multihead_crf",
            max_lengths=[32],
            n_trials=2,
            seeds_per_trial=2,
            track_resources=False,
            report=False,
            early_stopping_patience=None,
            training_arguments={
                "num_train_epochs": 1,
                "use_cpu": True,
                "fp16": False,
                "disable_tqdm": True,
            },
            search_space={
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "lr_scheduler_type": "linear",
                "effective_train_batch_size": 4,
                "architecture__aux_ce_weight": {"type": "choice", "categories": [0.0, 0.1]},
            },
        )
        checks.equal("real Ray CRF trials", len(result.trials), 2)
        checks.check("Ray multiple seeds", (result.trials.n_seeds == 2).all())
        checks.equal("Ray macro objective", result.metric, "span_strict_macro_f1")
        checks.check(
            "winner contains CRF settings", "aux_ce_weight" in result.best["architecture_kwargs"]
        )
        checks.equal(
            "winner YAML trains unchanged", cli_run(["run", str(result.paths["winner"])]), 0
        )
    finally:
        ray.shutdown()


def verify_yaml(checks, base, corpus, root):
    import inspect

    import yaml

    from lab.cli import main as cli_run
    from lab.ner import search_hyperparameters

    templates = Path(__file__).resolve().parents[1] / "docs" / "ner" / "configs"
    functions = {
        "ner.train_model": train_model,
        "ner.predict_entities": predict_entities,
        "ner.search_hyperparameters": search_hyperparameters,
        "ner.fit_ensemble": fit_ensemble,
    }
    configurations = {}
    for name in (
        "crf-train",
        "multihead-crf-train",
        "crf-auxiliary",
        "crf-predict",
        "crf-hpo",
        "multihead-crf-hpo",
        "ensemble-crf",
    ):
        config = yaml.safe_load((templates / f"{name}.yaml").read_text())
        inspect.signature(functions[config["task"]]).bind(
            **{k: v for k, v in config.items() if k != "task"}
        )
        configurations[name] = config
        checks.check("documented YAML signature " + name, True)
    root.mkdir()
    source = write_corpus(corpus, root / "corpus.parquet")
    prepared = prepare_dataset(root / "split", source_parquet=source, validation_size=0.25)
    for name in ("crf-train", "multihead-crf-train", "crf-auxiliary"):
        config = configurations[name]
        config.update(
            base_model=str(base),
            split_dir=str(prepared.split.paths["train"].parent),
            output_dir=str(root / name),
            max_length=32,
            track_resources=False,
        )
        config["training_arguments"].update(
            num_train_epochs=1, use_cpu=True, fp16=False, disable_tqdm=True
        )
        file = root / f"{name}.yaml"
        file.write_text(yaml.safe_dump(config))
        checks.equal("documented training YAML executes " + name, cli_run(["run", str(file)]), 0)
    config = configurations["crf-predict"]
    config.update(
        model_dir=str(root / "crf-train" / "best_model"),
        documents=str(source),
        output_dir=str(root / "yaml-prediction"),
        device="cpu",
    )
    file = root / "predict.yaml"
    file.write_text(yaml.safe_dump(config))
    checks.equal("documented CRF inference YAML executes", cli_run(["run", str(file)]), 0)


def main():
    torch.manual_seed(23)
    checks = Checks("word-level constrained CRF integration")
    verify_packing(checks)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tokenizer = local_tokenizer()
        base = tiny_base_model(root, tokenizer)
        corpus = overlap_corpus()
        verify_backbones(checks, tokenizer)
        verify_encoding(checks, tokenizer, corpus)
        paths = verify_model(checks, base, tokenizer, corpus, root)
        from verify_unified_ensemble import save_member

        paths.extend(
            [
                save_member(root / "hybrid-linear", base, tokenizer, "linear", ["SYMPTOM"]),
                save_member(
                    root / "hybrid-mh",
                    base,
                    tokenizer,
                    "multihead",
                    ["TOKEN", "SYMPTOM", "NESTED", "FINDING"],
                ),
                save_member(root / "hybrid-legacy-crf", base, tokenizer, "crf", ["SYMPTOM"]),
            ]
        )
        verify_orchestration(checks, base, corpus, root)
        # Independent names prevent known training/calibration overlaps in tuned verification.
        validation = corpus.copy()
        validation["doc_id"] = "cal-" + validation.doc_id
        # Content overlap is intentionally detected; use synthetic direct checkpoints for tuning elsewhere.
        verify_ensembles(checks, paths, validation, root / "ensemble")
        verify_ray(checks, base, corpus, root)
        verify_yaml(checks, base, corpus, root / "yaml")
        from verify_multihead import verify_folds

        fold_root = root / "crf-folds"
        fold_root.mkdir()
        with patch(
            "lab.ner.train_model",
            side_effect=lambda *args, **kwargs: train_model(
                *args, **{**kwargs, "architecture": "multihead_crf"}
            ),
        ):
            verify_folds(checks, base, corpus, fold_root)
    if not torch.cuda.is_available():
        checks.skip("CUDA end-to-end", "CPU-only environment")
    return checks.report()


if __name__ == "__main__":
    run(main)
