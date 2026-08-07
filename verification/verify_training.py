"""Verify the torch-free parts of training, and the rest when torch is installed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run


def verify_rows(checks: Checks) -> None:
    from ner_lab.training.dataset import ensure_int_list, is_encoded, validate_rows

    checks.equal("a list is coerced", ensure_int_list([1, 2]), [1, 2])
    checks.equal("a tuple is coerced", ensure_int_list((1, 2)), [1, 2])
    checks.equal("floats are truncated to ints", ensure_int_list([1.0, 2.0]), [1, 2])
    checks.equal("a numpy array is coerced", ensure_int_list(np.array([1, 2])), [1, 2])
    checks.equal("a stringified list is parsed", ensure_int_list("[1, 2]"), [1, 2])
    checks.raises("a scalar raises", TypeError, ensure_int_list, 5)
    checks.raises("a non-list string raises", TypeError, ensure_int_list, "5")

    rows = pd.DataFrame(
        {"input_ids": [[1, 2]], "attention_mask": [[1, 1]], "labels": [[0, 0]], "doc_id": ["d"]}
    )

    checks.check("encoded rows are recognized", is_encoded(rows))
    checks.check("a bare corpus is not", not is_encoded(pd.DataFrame({"doc_id": ["d"]})))

    validate_rows(rows)
    checks.check("valid rows pass validation", True)

    checks.raises(
        "missing columns raise",
        ValueError,
        validate_rows,
        rows.drop(columns=["labels"]),
        match="labels",
    )
    checks.raises("empty rows raise", ValueError, validate_rows, rows.iloc[0:0], match="empty")


def verify_arguments(checks: Checks) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        checks.skip("training_arguments / train", "torch is not installed in this environment")
        return

    import tempfile

    from ner_lab.training import DEFAULTS, training_arguments

    with tempfile.TemporaryDirectory() as tmp:
        arguments = training_arguments(tmp)

        checks.equal("defaults are applied", arguments.learning_rate, DEFAULTS["learning_rate"])
        checks.equal("output_dir is set", arguments.output_dir, tmp)

        overridden = training_arguments(tmp, learning_rate=1e-4, num_train_epochs=3)

        checks.equal("overrides win", overridden.learning_rate, 1e-4)
        checks.equal("other defaults survive an override", overridden.weight_decay, 0.01)


def verify_end_to_end(checks: Checks) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        checks.skip("end-to-end training", "torch is not installed in this environment")
        return

    import tempfile

    from transformers import AutoTokenizer

    from fixtures import synthetic_corpus, tiny_checkpoint
    from ner_lab.encoding import Encoder
    from ner_lab.models import build_model
    from ner_lab.evaluation import build_compute_metrics
    from ner_lab.training import CRFTrainer, resolve_trainer_class, train, training_arguments

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(tokenizer, "DISEASE", "es", max_length=64)

    corpus = synthetic_corpus(12)
    rows = encoder.encode(corpus)

    shared = tempfile.TemporaryDirectory()
    checkpoint = tiny_checkpoint(Path(shared.name), tokenizer)

    model = build_model(str(checkpoint), encoder.label2id, encoder.id2label)
    compute_metrics = build_compute_metrics(rows, tokenizer, encoder.id2label)

    checks.equal("a linear model gets the plain Trainer", resolve_trainer_class(model).__name__, "Trainer")
    checks.equal(
        "the model is configured with the encoder's vocabulary",
        model.config.label2id,
        encoder.label2id,
    )

    with tempfile.TemporaryDirectory() as tmp:
        result = train(
            model=model,
            tokenizer=tokenizer,
            train_rows=rows,
            validation_rows=rows,
            training_arguments=training_arguments(
                tmp, num_train_epochs=1, per_device_train_batch_size=4, fp16=False
            ),
            early_stopping_patience=None,
            compute_metrics=compute_metrics,
        )

        checks.check("training produced a step", result.summary["train_output"]["global_step"] > 0)
        checks.equal("epoch metrics has one eval row", len(result.epoch_metrics), 1)
        checks.check("the summary was written", result.paths["summary"].exists())
        checks.check(
            "span metrics reached the epoch table", "span_strict_f1" in result.epoch_metrics.columns
        )
        checks.equal(
            "model selection tracks the span metric", result.summary["best"]["metric"], "span_strict_f1"
        )
        checks.check("epoch metrics was written", result.paths["epoch_metrics"].exists())
        checks.equal(
            "row counts are recorded",
            result.summary["dataset"]["train_rows"],
            len(rows),
        )
        checks.equal(
            "documents are recorded",
            result.summary["dataset"]["train_documents"],
            len(corpus),
        )
        checks.check(
            "checkpoints are cleaned up when save_model is off",
            not list(Path(tmp).glob("checkpoint-*")),
        )
        checks.check("no best_model directory without save_model", not (Path(tmp) / "best_model").exists())

    with tempfile.TemporaryDirectory() as tmp:
        saved = train(
            model=model,
            tokenizer=tokenizer,
            train_rows=rows,
            validation_rows=rows,
            training_arguments=training_arguments(
                tmp, num_train_epochs=1, per_device_train_batch_size=4, fp16=False
            ),
            early_stopping_patience=None,
            compute_metrics=compute_metrics,
            save_model=True,
        )

        checks.check("save_model writes weights", saved.paths["best_model"].exists())
        checks.check(
            "the tokenizer is saved next to them",
            (saved.paths["best_model"] / "tokenizer_config.json").exists(),
        )

    crf = build_model(str(checkpoint), encoder.label2id, encoder.id2label, architecture="crf")

    checks.equal("a CRF model gets the CRF Trainer", resolve_trainer_class(crf), CRFTrainer)

    with tempfile.TemporaryDirectory() as tmp:
        crf_result = train(
            model=crf,
            tokenizer=tokenizer,
            train_rows=rows,
            validation_rows=rows,
            training_arguments=training_arguments(
                tmp, num_train_epochs=1, per_device_train_batch_size=4, fp16=False
            ),
            early_stopping_patience=None,
            compute_metrics=compute_metrics,
        )

        checks.check("the CRF model trains", crf_result.summary["train_output"]["global_step"] > 0)
        checks.check(
            "the CRF loss is finite", crf_result.summary["train_output"]["training_loss"] == crf_result.summary["train_output"]["training_loss"]
        )

    with tempfile.TemporaryDirectory() as tmp:
        checks.raises(
            "early stopping without load_best_model_at_end raises",
            ValueError,
            train,
            model=model,
            tokenizer=tokenizer,
            train_rows=rows,
            validation_rows=rows,
            training_arguments=training_arguments(
                tmp, num_train_epochs=1, load_best_model_at_end=False, fp16=False
            ),
            compute_metrics=compute_metrics,
            early_stopping_patience=2,
            match="load_best_model_at_end",
        )

    with tempfile.TemporaryDirectory() as tmp:
        checks.raises(
            "a span metric without compute_metrics raises",
            ValueError,
            train,
            model=model,
            tokenizer=tokenizer,
            train_rows=rows,
            validation_rows=rows,
            training_arguments=training_arguments(tmp, num_train_epochs=1, fp16=False),
            early_stopping_patience=None,
            match="compute_metrics",
        )

    shared.cleanup()


def main() -> int:
    checks = Checks("training")

    verify_rows(checks)
    verify_arguments(checks)
    verify_end_to_end(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
