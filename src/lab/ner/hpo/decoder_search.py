"""The decoder-only mode of the existing search_hyperparameters entry point."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from lab.core import read_corpus
from lab.core.provenance import write_manifest
from lab.core.split import split_paths
from lab.ner.encoding.encoder import encoder_from_description
from lab.ner.ensemble.calibration import describe_calibration
from lab.ner.ensemble.io import describe_member
from lab.ner.hpo.decoding import fit_binary_decoding, validate_decoder_space
from lab.ner.hpo.winner import write_winner_config
from lab.ner.inference.binary_settings import model_fingerprint
from lab.ner.inference.loading import load_model
from lab.ner.inference.prediction import predict_logits
from lab.ner.training.assessment import resolve_training_arguments
from lab.ner.training.multihead import architecture_arguments
from lab.ner.training.runs import claim_run_dir, fold_rotations, read_data_manifest

if TYPE_CHECKING:
    from transformers import TrainingArguments

    from lab.ner.hpo.search import HPOResult


def search_checkpoint_decoding(
    split_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    search_space: Mapping[str, Any] | None,
    training_arguments: TrainingArguments | Mapping[str, Any] | None,
    n_trials: int,
    random_state: int | None,
    validation_index: int | None,
    threshold_scope: str,
    temperature_scope: str,
    min_gold_entities: int,
    overwrite: bool,
) -> HPOResult:
    """Load validation once and fit all decoder candidates against one cached inference."""
    from lab.ner.hpo.search import HPOResult

    if search_space is None:
        raise ValueError("model_dir mode requires explicit decoder_search_space.")
    validate_decoder_space(search_space)
    if (
        threshold_scope not in ("global", "entity")
        or temperature_scope not in ("global", "entity")
        or n_trials < 1
        or min_gold_entities < 1
    ):
        raise ValueError("Use valid global/entity scopes and positive trial/support counts.")
    started = time.perf_counter()
    root = claim_run_dir(output_dir, overwrite)
    manifest = read_data_manifest(split_dir)
    requested = None if validation_index is None else [validation_index]
    rotation = fold_rotations(manifest, requested)[0]
    partitions = split_paths(split_dir, validation_index=rotation)
    validation_paths = partitions["validation"]
    validation_paths = (
        validation_paths if isinstance(validation_paths, list) else [validation_paths]
    )
    corpus = pd.concat([read_corpus(path) for path in validation_paths], ignore_index=True)
    member = describe_member(model_dir)
    provenance = describe_calibration([member], corpus, tuned=True)
    arguments = resolve_training_arguments(
        architecture_arguments(training_arguments, "multilabel"), root / "inference", random_state
    )
    if (arguments.metric_for_best_model or "").endswith("loss"):
        raise ValueError("Decoder-only search selects span/character metrics, not training loss.")
    if arguments.greater_is_better is False:
        raise ValueError(
            "Decoder-only search maximizes span/character metrics; use greater_is_better=True."
        )
    model, tokenizer, description = load_model(
        model_dir, device="cpu" if arguments.use_cpu else "auto"
    )
    if getattr(model, "label_schema", None) != "multilabel":
        raise ValueError("Decoder-only sigmoid search requires a multilabel checkpoint.")
    encoder = encoder_from_description(
        description["encoding"], tokenizer, require_target_label=False
    )
    rows = encoder.encode(corpus)
    encoded = time.perf_counter()
    logits = predict_logits(model, tokenizer, rows, training_arguments=arguments)
    inferred = time.perf_counter()
    metric = arguments.metric_for_best_model
    metric = metric.removeprefix("eval_")
    fitted = fit_binary_decoding(
        rows,
        logits,
        tokenizer,
        model.entity_labels,
        search_space,
        model.decoding,
        metric,
        n_trials,
        random_state,
        threshold_scope,
        temperature_scope,
        min_gold_entities,
    )
    fitted_at = time.perf_counter()
    artifact = {
        "format_version": 1,
        "model_dir": str(Path(model_dir).resolve()),
        "entity_labels": model.entity_labels,
        "model_fingerprint": model_fingerprint(model_dir),
        "decoding": asdict(fitted.settings),
        "metric": metric,
        "calibration": provenance,
        "support": fitted.support,
        "requires_revalidation_after_retraining": True,
    }
    decoder_path = write_manifest(artifact, root / "decoding.json")
    kwargs = {**model.architecture_arguments(), **asdict(fitted.settings)}
    kwargs.pop("entity_labels")
    kwargs.pop("pos_weight")
    winner = {
        "task": "ner.train_model",
        "split_dir": str(Path(split_dir).resolve()),
        "output_dir": str(root / "final_train"),
        "base_model": description["model"]["base_model"],
        "architecture": "multilabel",
        "target_label": None,
        "language": encoder.language,
        "max_length": encoder.max_length,
        "strategy": encoder.strategy,
        "context_tokens": encoder.context_tokens,
        "overlap_policy": encoder.overlap_policy,
        "min_sentence_tokens": encoder.min_sentence_tokens,
        "training_arguments": {
            "metric_for_best_model": metric,
            **({"seed": random_state} if random_state is not None else {}),
        },
        **({"folds": [rotation]} if rotation is not None else {}),
        "architecture_kwargs": kwargs,
        "save_model": True,
    }
    trials_path = root / "trials_summary.parquet"
    trials = fitted.trials.copy()
    for name in ("temperature", "entity_threshold", "entity_bias"):
        if trials[name].map(lambda value: isinstance(value, dict)).any():
            trials[name + "_json"] = trials.pop(name).map(json.dumps)
    trials.to_parquet(trials_path, index=False)
    paths = {
        "decoder": decoder_path,
        "trials": trials_path,
        "winner": write_winner_config(winner, root / "winner.yaml"),
    }
    run_manifest = {
        "mode": "cached_decoder_selection",
        "inference_arguments": arguments.to_dict(),
        "split_dir": str(split_dir),
        "validation_paths": [str(path) for path in validation_paths],
        "model_dir": str(model_dir),
        "search_space": dict(search_space),
        "calibration": provenance,
        "transformer_inference_runs": 1,
        "requested_trials": n_trials,
        "executed_trials": len(fitted.trials),
        "timings": {
            "load_encode_seconds": encoded - started,
            "inference_seconds": inferred - encoded,
            "fit_seconds": fitted_at - inferred,
        },
    }
    paths["run_manifest"] = write_manifest(run_manifest, root / "run_manifest.json")
    summary = {
        "best_metric": fitted.metrics[metric],
        "metrics": fitted.metrics,
        "train_model": winner,
        "decoder": artifact,
    }
    paths["summary"] = write_manifest(summary, root / "hpo_summary.json")

    return HPOResult(
        root,
        metric,
        summary,
        winner,
        trials,
        run_manifest,
        paths,
        "Cached decoder selection; revalidate choices after retraining.",
    )
