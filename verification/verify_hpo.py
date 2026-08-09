"""Verify HPO: search-space forms, trial scoring, OOM handling, and a real mini-sweep."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


def verify_space(checks: Checks) -> None:
    from ray import tune
    from ray.tune.search.sample import Domain

    from ner_lab.hpo import (
        DEFAULT_SEARCH_SPACE,
        build_search_space,
        describe_search_space,
        smoke_configuration,
    )

    space = build_search_space()

    checks.equal("the default space keeps every dimension", sorted(space), sorted(DEFAULT_SEARCH_SPACE))
    checks.check("declarative specs become domains", all(isinstance(domain, Domain) for domain in space.values()))

    overridden = build_search_space({"learning_rate": {"type": "uniform", "low": 1e-5, "high": 2e-5}})

    checks.equal(
        "an override replaces the default domain",
        describe_search_space(overridden)["learning_rate"]["type"],
        "uniform",
    )

    checks.check("None removes a dimension", "warmup_ratio" not in build_search_space({"warmup_ratio": None}))

    pinned = build_search_space({"lr_scheduler_type": "linear"})

    checks.equal(
        "a scalar pins the value as a single choice",
        describe_search_space(pinned)["lr_scheduler_type"]["categories"],
        ["linear"],
    )

    passed_through = tune.loguniform(1e-6, 1e-4)

    checks.check(
        "a Ray domain passes through untouched",
        build_search_space({"learning_rate": passed_through})["learning_rate"] is passed_through,
    )

    checks.raises(
        "an unknown declarative type raises",
        ValueError,
        build_search_space,
        {"learning_rate": {"type": "qlograndint", "low": 1, "high": 2}},
        match="qlograndint",
    )

    described = describe_search_space(space)

    checks.equal(
        "describe inverts the declarative form",
        described["learning_rate"],
        DEFAULT_SEARCH_SPACE["learning_rate"],
    )
    checks.equal(
        "categories survive the round trip",
        described["effective_train_batch_size"]["categories"],
        [16, 32, 64],
    )

    smoke = smoke_configuration(space)

    checks.equal("the smoke configuration takes lower bounds", smoke["learning_rate"], 2e-5)
    checks.equal("the smoke configuration takes first categories", smoke["lr_scheduler_type"], "linear")


def verify_batch_sizes(checks: Checks) -> None:
    from ner_lab.hpo import resolve_batch_sizes

    checks.equal("a fitting batch is not split", resolve_batch_sizes(16, 64), (16, 1))
    checks.equal("an oversized batch accumulates", resolve_batch_sizes(64, 16), (16, 4))
    checks.equal(
        "a non-dividing ceiling drops to the largest divisor",
        resolve_batch_sizes(48, 32),
        (24, 2),
    )
    checks.equal("the degenerate case is 1x", resolve_batch_sizes(1, 64), (1, 1))
    checks.raises("a zero batch size raises", ValueError, resolve_batch_sizes, 0, 64)


def verify_variants(checks: Checks) -> None:
    from ner_lab.hpo import build_variants, describe_variants, variant_key

    variants = build_variants(
        ["/models/roberta-base", "/models/beto"],
        strategies=["greedy", "context"],
        context_tokens=[32, 64],
        max_lengths=[128, 256],
    )

    checks.equal("(greedy + context x2) x2 max_lengths x2 checkpoints", len(variants), 12)
    checks.check(
        "keys name the combination",
        "roberta-base|context|32|128" in variants and "beto|greedy|None|256" in variants,
    )
    checks.equal(
        "the specification carries the full checkpoint path",
        variants["beto|greedy|None|256"]["checkpoint"],
        "/models/beto",
    )

    checks.raises(
        "the context strategy without context_tokens raises",
        ValueError,
        build_variants,
        ["/models/beto"],
        strategies=["context"],
        match="context_tokens",
    )
    checks.raises(
        "context_tokens without the context strategy raises",
        ValueError,
        build_variants,
        ["/models/beto"],
        strategies=["greedy"],
        context_tokens=[32],
        match="context",
    )
    checks.raises(
        "two checkpoints sharing a name collide loudly",
        ValueError,
        build_variants,
        ["/a/model", "/b/model"],
        match="Duplicate",
    )

    def my_windows(sentence_ranges, tokens, entities, text, max_content_length):
        return []

    checks.equal(
        "a callable strategy is named in the key",
        variant_key("/models/beto", my_windows, None, 128),
        "beto|my_windows|None|128",
    )
    checks.equal(
        "describe_variants names callable strategies",
        describe_variants({"k": {"checkpoint": "c", "strategy": my_windows, "context_tokens": None, "max_length": 1}})["k"]["strategy"],
        "my_windows",
    )


def verify_scoring(checks: Checks) -> None:
    from ner_lab.hpo import is_oom_error, metric_greater_is_better, top_k_epoch_mean
    from ner_lab.training import training_arguments

    epochs = pd.DataFrame(
        {
            "split": ["eval", "train", "eval", "eval"],
            "epoch": [1.0, 1.0, 2.0, 3.0],
            "span_strict_f1": [0.2, 0.95, 0.6, 0.4],
        }
    )

    checks.equal("the top-k mean averages the k best epochs", top_k_epoch_mean(epochs, "span_strict_f1", k=2), 0.5)
    checks.check(
        "k larger than the run uses every epoch",
        abs(top_k_epoch_mean(epochs, "span_strict_f1", k=10) - 0.4) < 1e-9,
    )
    checks.equal(
        "an eval_ prefixed metric resolves to the stripped column",
        top_k_epoch_mean(epochs, "eval_span_strict_f1", k=2),
        0.5,
    )
    checks.check(
        "the training pass never contributes",
        top_k_epoch_mean(epochs, "span_strict_f1", k=1) == 0.6,
    )
    checks.check(
        "greater_is_better=False averages the k lowest",
        abs(top_k_epoch_mean(epochs, "span_strict_f1", k=2, greater_is_better=False) - 0.3) < 1e-9,
    )
    checks.raises(
        "an absent metric raises",
        ValueError,
        top_k_epoch_mean,
        epochs,
        "recall",
    )

    import torch

    checks.check("a torch OOM type is recognized", is_oom_error(torch.cuda.OutOfMemoryError("boom")))
    checks.check(
        "an allocator message is recognized",
        is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED during cublasCreate")),
    )
    checks.check("an ordinary error is not", not is_oom_error(ValueError("bad label")))

    with tempfile.TemporaryDirectory() as tmp:
        checks.check(
            "the default metric is maximized",
            metric_greater_is_better(training_arguments(tmp)),
        )
        checks.check(
            "a loss metric with unset direction is minimized",
            not metric_greater_is_better(
                training_arguments(tmp, metric_for_best_model="eval_loss", greater_is_better=None)
            ),
        )


def verify_base_arguments(checks: Checks) -> None:
    from transformers import TrainingArguments

    from ner_lab.hpo import HPO_ARGUMENT_DEFAULTS, resolve_base_arguments
    from ner_lab.training import DEFAULTS

    with tempfile.TemporaryDirectory() as tmp:
        base = resolve_base_arguments(None, tmp)

        checks.check("sweeps write no checkpoints", base.save_strategy == "no")
        checks.check("sweeps skip best-model reloading", not base.load_best_model_at_end)
        checks.equal(
            "the epoch cap comes from the HPO defaults",
            base.num_train_epochs,
            HPO_ARGUMENT_DEFAULTS["num_train_epochs"],
        )
        checks.equal("library defaults fill the rest", base.learning_rate, DEFAULTS["learning_rate"])

        overridden = resolve_base_arguments({"num_train_epochs": 7, "fp16": False}, tmp)

        checks.equal("a mapping overrides the HPO defaults", overridden.num_train_epochs, 7)

        given = TrainingArguments(output_dir="/elsewhere", num_train_epochs=3, learning_rate=7e-5)
        adopted = resolve_base_arguments(given, tmp)

        checks.equal("an instance keeps its own fields", adopted.learning_rate, 7e-5)
        checks.equal(
            "but the sweep defaults are forced onto it",
            adopted.num_train_epochs,
            HPO_ARGUMENT_DEFAULTS["num_train_epochs"],
        )
        checks.check("including no checkpointing", adopted.save_strategy == "no")
        checks.equal("but its output_dir is redirected", adopted.output_dir, tmp)
        checks.equal("random_state sets the seed", resolve_base_arguments(None, tmp, 11).seed, 11)


def verify_winner_configuration(checks: Checks) -> None:
    from ner_lab.hpo import winner_configuration
    from ner_lab.training import training_arguments

    with tempfile.TemporaryDirectory() as tmp:
        block = winner_configuration(
            split_dir="/splits/disease/kfold_5_holdout_0",
            output_dir="/runs",
            variant={"checkpoint": "/models/beto", "strategy": "greedy", "context_tokens": None, "max_length": 128},
            sampled={"variant": "beto|greedy|None|128", "learning_rate": 3e-5, "effective_train_batch_size": 32, "weight_decay": 0.05},
            micro_batch_size=16,
            accumulation_steps=2,
            target_label="DISEASE",
            language="es",
            architecture="crf",
            architecture_kwargs={"dropout": 0.2},
            overlap_policy="merge_same_label_then_keep_longest",
            min_sentence_tokens=4,
            min_overlap_percentage=40.0,
            early_stopping_patience=5,
            base_arguments=training_arguments(tmp, num_train_epochs=40),
            user_overrides={"fp16": False, "save_strategy": "no", "load_best_model_at_end": False},
        )

    checks.equal("the block is a train_model task", block["task"], "train_model")
    checks.equal("the winning variant decides the checkpoint", block["checkpoint"], "/models/beto")
    checks.equal("the winning variant decides the windowing", block["max_length"], 128)

    arguments = block["training_arguments"]

    checks.equal("sampled hyperparameters land in training_arguments", arguments["learning_rate"], 3e-5)
    checks.check("reserved dimensions do not", "effective_train_batch_size" not in arguments and "variant" not in arguments)
    checks.equal("the resolved micro batch is carried", arguments["per_device_train_batch_size"], 16)
    checks.equal("the resolved accumulation is carried", arguments["gradient_accumulation_steps"], 2)
    checks.equal("user overrides are carried", arguments["fp16"], False)
    checks.check(
        "sweep-only settings are dropped",
        "save_strategy" not in arguments and "load_best_model_at_end" not in arguments,
    )
    checks.equal("the epoch cap is carried", arguments["num_train_epochs"], 40)
    checks.equal("architecture kwargs are carried", block["architecture_kwargs"], {"dropout": 0.2})

    verify_winner_config_file(checks, block)


def verify_winner_config_file(checks: Checks, block: dict) -> None:
    from ner_lab.cli import load_config
    from ner_lab.hpo import write_winner_config

    with tempfile.TemporaryDirectory() as tmp:
        path = write_winner_config(block, Path(tmp) / "winner.yaml")

        checks.check("the winner config is written", path.exists())
        checks.equal("the CLI reads it back unchanged", load_config(path), block)


def verify_task_registration(checks: Checks) -> None:
    from ner_lab.hpo import search_hyperparameters
    from ner_lab.tasks import resolve_task

    checks.check(
        "the task name resolves to the search",
        resolve_task("search_hyperparameters") is search_hyperparameters,
    )


def verify_run_trial(checks: Checks) -> None:
    from transformers import AutoTokenizer

    from fixtures import tiny_checkpoint
    from ner_lab.data.split import split_paths
    from ner_lab.hpo import build_variants, encode_variants, resolve_base_arguments, run_trial
    from verify_assessment import prepare_split

    shared = tempfile.TemporaryDirectory()
    root = Path(shared.name)
    checkpoint = tiny_checkpoint(root, AutoTokenizer.from_pretrained("bert-base-uncased"))

    split_dir = prepare_split(root / "data", kfolds=None)
    partitions = split_paths(split_dir)

    variants = build_variants([str(checkpoint)], max_lengths=[64])
    encoded = encode_variants(partitions, variants, "DISEASE", "es")
    key = next(iter(encoded))

    checks.check("the variant is encoded once, up front", len(encoded[key].train_rows) > 0)

    base = resolve_base_arguments({"num_train_epochs": 2, "fp16": False}, root / "arguments")

    report = run_trial(
        {"variant": key, "learning_rate": 3e-5, "effective_train_batch_size": 4},
        variants=encoded,
        base_arguments=base,
        trial_dir=root / "trial",
        max_micro_batch_size=2,
        seeds_per_trial=2,
        top_k_epochs=2,
        early_stopping_patience=3,
        track_resources=False,
    )

    checks.equal("one training per seed", report["n_seeds"], 2)
    checks.equal("per-seed scores are kept", len(report["per_seed_scores"]), 2)
    checks.check("the trial score is their mean", report["span_strict_f1"] == sum(report["per_seed_scores"]) / 2)
    checks.check("the spread across seeds is reported", report["span_strict_f1_std"] >= 0)
    checks.equal("the effective batch is resolved under the ceiling", report["per_device_train_batch_size"], 2)
    checks.equal("accumulation makes up the difference", report["gradient_accumulation_steps"], 2)
    checks.check("the trial did not OOM", not report["oom"])
    checks.check(
        "early stopping ran without checkpoints being written",
        not list((root / "trial").rglob("checkpoint-*")),
    )
    checks.check(
        "no weights survive a trial",
        not list((root / "trial").rglob("best_model")),
    )

    def exploding_architecture(checkpoint, label2id, id2label):
        import torch

        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate everything")

    oom_report = run_trial(
        {"variant": key, "learning_rate": 3e-5},
        variants=encoded,
        base_arguments=base,
        trial_dir=root / "oom_trial",
        architecture=exploding_architecture,
        seeds_per_trial=2,
        track_resources=False,
    )

    checks.check("an OOM trial is reported, not raised", oom_report["oom"])
    checks.equal("an OOM trial scores worst", oom_report["span_strict_f1"], float("-inf"))
    checks.equal("no seed completed", oom_report["n_seeds"], 0)

    def broken_architecture(checkpoint, label2id, id2label):
        raise ValueError("a real bug")

    checks.raises(
        "a non-OOM error propagates and would halt the sweep",
        ValueError,
        run_trial,
        {"variant": key, "learning_rate": 3e-5},
        variants=encoded,
        base_arguments=base,
        trial_dir=root / "bug_trial",
        architecture=broken_architecture,
        seeds_per_trial=1,
        track_resources=False,
        match="a real bug",
    )

    shared.cleanup()


def verify_end_to_end(checks: Checks) -> None:
    from transformers import AutoTokenizer

    from fixtures import tiny_checkpoint
    from ner_lab.hpo import search_hyperparameters
    from ner_lab.training import train_model
    from verify_assessment import prepare_split

    shared = tempfile.TemporaryDirectory()
    root = Path(shared.name)
    checkpoint = tiny_checkpoint(root, AutoTokenizer.from_pretrained("bert-base-uncased"))
    split_dir = prepare_split(root / "data", kfolds=None)

    result = search_hyperparameters(
        split_dir=split_dir,
        output_dir=root / "sweeps",
        checkpoints=str(checkpoint),
        target_label="DISEASE",
        language="es",
        search_space={
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-4},
            "weight_decay": None,
            "warmup_ratio": None,
            "lr_scheduler_type": "linear",
            "effective_train_batch_size": {"type": "choice", "categories": [4, 8]},
        },
        training_arguments={"num_train_epochs": 1, "fp16": False},
        n_trials=2,
        seeds_per_trial=1,
        top_k_epochs=1,
        max_lengths=[64],
        max_micro_batch_size=4,
        gpus_per_trial=0,
        early_stopping_patience=None,
        track_resources=False,
        run_name="mini_sweep",
    )

    checks.check("the sweep succeeded", result.summary["success"])
    checks.equal("every trial ran", result.summary["n_trials"], 2)
    checks.equal("no trial errored", result.summary["num_errors"], 0)
    checks.equal("the trials table has one row per trial", len(result.trials), 2)
    checks.check("sampled dimensions reach the table", "learning_rate" in result.trials.columns)
    checks.check("scores reach the table", result.trials["span_strict_f1"].notna().all())
    checks.check("the run manifest is written", result.paths["run_manifest"].exists())
    checks.check("the trials summary is written", result.paths["trials_summary"].exists())
    checks.check("the hpo summary is written", result.paths["hpo_summary"].exists())
    checks.check("wall time is recorded", result.summary["total_wall_time_sec"] > 0)

    manifest = json.loads(result.paths["run_manifest"].read_text(encoding="utf-8"))

    checks.check("the manifest records the variants", len(manifest["variants"]) == 1)
    checks.equal(
        "the manifest records the declarative space",
        manifest["search_space"]["learning_rate"]["type"],
        "loguniform",
    )
    checks.check("the variant dimension is recorded too", "variant" in manifest["search_space"])
    checks.equal("the objective is recorded", manifest["objective"]["metric"], "span_strict_f1")
    checks.check("the split provenance is embedded", "split" in manifest["data_manifest"])

    best = result.best

    checks.equal("the winner block is a train_model task", best["task"], "train_model")
    checks.equal("it points at the searched split", best["split_dir"], str(split_dir))
    checks.check("it carries a sampled learning rate", 1e-5 <= best["training_arguments"]["learning_rate"] <= 1e-4)

    from ner_lab.cli import load_config

    checks.check("the winner config is written", result.paths["winner"].exists())
    checks.equal(
        "the winner config holds the winner block",
        load_config(result.paths["winner"]),
        best,
    )

    block = {key: value for key, value in best.items() if key != "task"}
    block["training_arguments"]["num_train_epochs"] = 1
    block["output_dir"] = str(root / "winner_run")

    winner = train_model(**block, track_resources=False, run_name="winner")

    checks.check("the winner block runs train_model as-is", len(winner.fold_metrics) == 1)

    shared.cleanup()


def main() -> int:
    checks = Checks("hpo")

    if not torch_available():
        checks.skip("hpo", "torch is not installed in this environment")

        return checks.report()

    verify_space(checks)
    verify_batch_sizes(checks)
    verify_variants(checks)
    verify_scoring(checks)
    verify_base_arguments(checks)
    verify_winner_configuration(checks)
    verify_task_registration(checks)
    verify_run_trial(checks)
    verify_end_to_end(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
