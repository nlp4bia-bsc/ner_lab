"""Verify the training orchestrator: run shape, fold rotation, aggregation, artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run

TRAIN_VALIDATION_MANIFEST = {"split": {"mode": "train_validation", "n_splits": None}}
KFOLD_MANIFEST = {"split": {"mode": "fixed_holdout_kfold", "n_splits": 4, "holdout_fold": 0}}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


def verify_rotations(checks: Checks) -> None:
    from ner_lab.training.assessment import fold_rotations

    checks.equal(
        "a train/validation split has one rotation",
        fold_rotations(TRAIN_VALIDATION_MANIFEST),
        [None],
    )
    checks.raises(
        "folds are meaningless without a holdout",
        ValueError,
        fold_rotations,
        TRAIN_VALIDATION_MANIFEST,
        [1],
        match="train_validation",
    )
    checks.equal(
        "every non-holdout fold rotates as validation",
        fold_rotations(KFOLD_MANIFEST),
        [1, 2, 3],
    )
    checks.equal(
        "folds narrows the rotation",
        fold_rotations(KFOLD_MANIFEST, [2, 3]),
        [2, 3],
    )
    checks.raises(
        "the holdout fold cannot be requested as validation",
        ValueError,
        fold_rotations,
        KFOLD_MANIFEST,
        [0],
        match="fixed holdout",
    )
    checks.raises(
        "an out-of-range fold raises",
        ValueError,
        fold_rotations,
        KFOLD_MANIFEST,
        [9],
        match="not rotatable",
    )
    checks.raises(
        "an unknown split mode raises",
        ValueError,
        fold_rotations,
        {"split": {"mode": "leave_one_out"}},
        match="Unsupported split mode",
    )


def verify_naming(checks: Checks) -> None:
    from ner_lab.training.assessment import architecture_name, run_directory_name

    checks.equal(
        "the run directory names what was trained",
        run_directory_name("DISEASE", "crf", "/models/roberta-base-bne", timestamp="20260807_120000"),
        "DISEASE__crf__roberta-base-bne__20260807_120000",
    )

    def my_architecture(base_model, label2id, id2label):
        return None

    checks.equal(
        "a callable architecture is named by its function",
        architecture_name(my_architecture),
        "my_architecture",
    )
    checks.equal("a builtin architecture keeps its name", architecture_name("linear"), "linear")


def verify_aggregation(checks: Checks) -> None:
    from ner_lab.training.assessment import aggregate_metrics, best_epoch_metrics

    frame = pd.DataFrame(
        {"validation_fold": [1, 2], "best_metric": [0.4, 0.6], "label": ["a", "b"]}
    )
    aggregate = aggregate_metrics(frame, exclude=("validation_fold",))

    checks.equal("the mean is aggregated", aggregate["best_metric"]["mean"], 0.5)
    checks.check("the standard deviation is aggregated", aggregate["best_metric"]["std"] > 0)
    checks.check("the excluded column is left out", "validation_fold" not in aggregate)
    checks.check("non-numeric columns are left out", "label" not in aggregate)

    single = aggregate_metrics(frame.iloc[:1], exclude=("validation_fold",))

    checks.equal("a single run has zero deviation", single["best_metric"]["std"], 0.0)
    checks.equal("an empty frame aggregates to nothing", aggregate_metrics(pd.DataFrame()), {})

    epochs = pd.DataFrame(
        {
            "split": ["eval", "train", "eval"],
            "epoch": [1.0, 1.0, 2.0],
            "span_strict_f1": [0.2, 0.9, 0.7],
            "loss": [0.5, 0.1, 0.9],
        }
    )
    best = best_epoch_metrics(epochs, "span_strict_f1")

    checks.equal("the best epoch is selected", best["epoch"], 2.0)
    checks.equal("its metrics come with it", best["span_strict_f1"], 0.7)
    checks.check("the training pass does not win on score", best["span_strict_f1"] != 0.9)
    checks.check("the split column is dropped", "split" not in best)

    minimized = best_epoch_metrics(epochs, "loss", greater_is_better=False)

    checks.equal("greater_is_better=False minimizes", minimized["epoch"], 1.0)
    checks.check("the training pass does not win on loss", minimized["loss"] != 0.1)
    checks.equal(
        "an eval_ prefixed metric resolves to the stripped column",
        best_epoch_metrics(epochs, "eval_span_strict_f1")["epoch"],
        2.0,
    )
    checks.equal("no epochs means no metrics", best_epoch_metrics(pd.DataFrame(), "f1"), {})
    checks.equal("an absent metric means no metrics", best_epoch_metrics(epochs, "recall"), {})


def verify_arguments(checks: Checks) -> None:
    from transformers import TrainingArguments

    from ner_lab.training import DEFAULTS
    from ner_lab.training.assessment import resolve_training_arguments

    with tempfile.TemporaryDirectory() as tmp:
        defaults = resolve_training_arguments(None, tmp)

        checks.equal("no arguments means the defaults", defaults.learning_rate, DEFAULTS["learning_rate"])
        checks.equal("output_dir is set from the run directory", defaults.output_dir, tmp)

        mapped = resolve_training_arguments({"learning_rate": 1e-4}, tmp)

        checks.equal("a mapping overrides the defaults", mapped.learning_rate, 1e-4)
        checks.equal("other defaults survive a mapping", mapped.weight_decay, DEFAULTS["weight_decay"])

        given = TrainingArguments(output_dir="/somewhere/else", learning_rate=5e-6)
        adopted = resolve_training_arguments(given, tmp)

        checks.equal("an instance is used as given", adopted.learning_rate, 5e-6)
        checks.equal("but its output_dir is redirected", adopted.output_dir, tmp)

        checks.equal("random_state sets the seed", resolve_training_arguments(None, tmp, 7).seed, 7)
        checks.equal(
            "random_state wins over a mapping's seed",
            resolve_training_arguments({"seed": 1}, tmp, 7).seed,
            7,
        )


def verify_task_registration(checks: Checks) -> None:
    from ner_lab.tasks import resolve_task
    from ner_lab.training.assessment import train_model

    checks.check("the task name resolves to the orchestrator", resolve_task("train_model") is train_model)


def verify_manifest_reading(checks: Checks) -> None:
    from ner_lab.training.assessment import read_data_manifest, split_provenance

    with tempfile.TemporaryDirectory() as tmp:
        checks.raises(
            "a directory with no data manifest raises",
            FileNotFoundError,
            read_data_manifest,
            tmp,
            match="prepare_dataset",
        )

    provenance = split_provenance(
        2, {"train": [Path("/a/fold_01.parquet")], "validation": Path("/a/fold_02.parquet")}
    )

    checks.equal("the validation fold is recorded", provenance["validation_fold"], 2)
    checks.equal(
        "a multi-fold train side is recorded as a list",
        provenance["train_parquet"],
        ["/a/fold_01.parquet"],
    )


def fold_directories(run_dir: Path) -> list[Path]:
    """The per-fold subdirectories of a run, ignoring `fold_metrics.parquet` beside them."""
    return sorted(
        path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")
    )


def prepare_split(root: Path, kfolds: int | None) -> Path:
    """Write a synthetic corpus and split it, returning the split directory."""
    from fixtures import synthetic_corpus
    from ner_lab.data import prepare_dataset, write_corpus

    source = write_corpus(synthetic_corpus(24), root / "source" / "documents.parquet")
    prepared = prepare_dataset(
        output_dir=root / "datasets",
        source_parquet=source,
        dataset_name="synthetic",
        kfolds=kfolds,
    )

    return prepared.split_dir


def verify_end_to_end(checks: Checks) -> None:
    from transformers import AutoTokenizer

    from fixtures import tiny_base_model
    from ner_lab.training import train_model

    shared = tempfile.TemporaryDirectory()
    root = Path(shared.name)
    base_model = tiny_base_model(root, AutoTokenizer.from_pretrained("bert-base-uncased"))

    arguments = {"num_train_epochs": 1, "per_device_train_batch_size": 4, "fp16": False}

    kfold_split = prepare_split(root / "kfold", kfolds=3)
    kfold = train_model(
        split_dir=kfold_split,
        output_dir=root / "runs",
        base_model=str(base_model),
        target_label="DISEASE",
        language="es",
        training_arguments=arguments,
        max_length=64,
        early_stopping_patience=None,
        track_resources=False,
        run_name="kfold_run",
    )

    checks.equal("the k-fold mode is detected", kfold.mode, "fixed_holdout_kfold")
    checks.equal("one run per rotatable fold", len(kfold.fold_metrics), 2)
    checks.equal(
        "the folds are the non-holdout ones",
        sorted(kfold.fold_metrics["validation_fold"].tolist()),
        [1, 2],
    )
    checks.equal(
        "each rotatable fold gets its own directory, and only those",
        [path.name for path in fold_directories(kfold.run_dir)],
        ["fold_01", "fold_02"],
    )
    checks.check("the run manifest is written", kfold.paths["run_manifest"].exists())
    checks.check("the fold metrics are written", kfold.paths["fold_metrics"].exists())
    checks.check("the assessment summary is written", kfold.paths["assessment_summary"].exists())
    checks.check("the selection metric is aggregated", "best_metric" in kfold.aggregate)
    checks.check(
        "span metrics reach the fold table", "span_strict_f1" in kfold.fold_metrics.columns
    )
    checks.check(
        "the fold column is not aggregated", "validation_fold" not in kfold.aggregate
    )
    checks.equal("one summary per fold", len(kfold.summaries), 2)
    checks.check(
        "each run records what it trained on",
        kfold.summaries[0]["metadata"]["validation_parquet"].endswith("fold_01.parquet"),
    )
    checks.check(
        "no checkpoints are left behind",
        not list(kfold.run_dir.glob("fold_*/checkpoint-*")),
    )
    checks.check(
        "no weights are saved by default",
        not list(kfold.run_dir.glob("fold_*/best_model")),
    )

    manifest = json.loads(kfold.paths["run_manifest"].read_text(encoding="utf-8"))

    checks.equal("the manifest records the folds", manifest["folds"], [1, 2])
    checks.equal("the manifest records the base_model", manifest["model"]["base_model"], str(base_model))
    checks.check("the manifest embeds the data manifest", "split" in manifest["data_manifest"])
    checks.check("the manifest records the label vocabulary", "B-DISEASE" in manifest["encoding"]["label2id"])
    checks.check(
        "the manifest records the resolved training arguments",
        manifest["training_arguments"]["num_train_epochs"] == 1,
    )
    checks.equal(
        "the manifest records the batch a step really covers",
        manifest["devices"]["effective_train_batch_size"],
        4,
    )
    checks.equal(
        "the manifest records that the device guard was not waived",
        manifest["devices"]["allow_multi_device"],
        False,
    )

    narrowed = train_model(
        split_dir=kfold_split,
        output_dir=root / "runs",
        base_model=str(base_model),
        target_label="DISEASE",
        language="es",
        training_arguments=arguments,
        max_length=64,
        early_stopping_patience=None,
        track_resources=False,
        folds=[2],
        run_name="narrowed_run",
    )

    checks.equal("folds narrows the run to one fold", len(narrowed.fold_metrics), 1)
    checks.equal("a single fold has zero deviation", narrowed.aggregate["best_metric"]["std"], 0.0)

    plain_split = prepare_split(root / "plain", kfolds=None)
    plain = train_model(
        split_dir=plain_split,
        output_dir=root / "runs",
        base_model=str(base_model),
        target_label="DISEASE",
        language="es",
        training_arguments=arguments,
        max_length=64,
        early_stopping_patience=None,
        track_resources=False,
        save_model=True,
        run_name="plain_run",
    )

    checks.equal("the train/validation mode is detected", plain.mode, "train_validation")
    checks.equal("it trains exactly once", len(plain.fold_metrics), 1)
    checks.check("there is no fold subdirectory", not fold_directories(plain.run_dir))
    checks.check("the single run writes into the run directory", plain.paths["run"] == plain.run_dir)
    checks.check("save_model writes weights", (plain.run_dir / "best_model").is_dir())
    checks.check(
        "the run summary is written next to them",
        (plain.run_dir / "training_summary.json").exists(),
    )

    checks.raises(
        "folds against a train/validation split raises",
        ValueError,
        train_model,
        split_dir=plain_split,
        output_dir=root / "runs",
        base_model=str(base_model),
        target_label="DISEASE",
        language="es",
        folds=[1],
        match="train_validation",
    )

    shared.cleanup()


def main() -> int:
    checks = Checks("assessment")

    if not torch_available():
        checks.skip("assessment", "torch is not installed in this environment")

        return checks.report()

    verify_rotations(checks)
    verify_naming(checks)
    verify_aggregation(checks)
    verify_arguments(checks)
    verify_task_registration(checks)
    verify_manifest_reading(checks)
    verify_end_to_end(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
