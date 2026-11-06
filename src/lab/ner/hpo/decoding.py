"""Validation-only decoder selection reusing one array of independent binary logits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import pandas as pd

from lab.core.spans import span_dataframe
from lab.ner.evaluation.multihead import strict_multihead_metrics
from lab.ner.evaluation.multilabel import gold_representation_metrics, multilabel_spans
from lab.ner.models.binary_decoding import BinaryDecoding

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DECODER_FIELDS = ("activation", "decoder", "temperature", "entity_threshold", "entity_bias")


@dataclass(frozen=True)
class DecodingFit:
    """Fixed decoder choices, fitting-set metrics and the complete reproducible trial trace."""

    settings: BinaryDecoding
    metrics: dict
    trials: pd.DataFrame
    support: dict[str, int]


def _suggest(trial: optuna.Trial, name: str, specification: Any) -> Any:
    if not isinstance(specification, Mapping):
        return specification
    kind = specification.get("type")
    if kind == "choice":
        choices = list(specification["categories"])
        return choices[0] if len(choices) == 1 else trial.suggest_categorical(name, choices)
    if kind in ("uniform", "loguniform"):
        return trial.suggest_float(
            name, specification["low"], specification["high"], log=kind == "loguniform"
        )

    raise ValueError(f"Unsupported decoder domain {name!r}: use choice/uniform/loguniform.")


def _named_setting(
    trial: optuna.Trial,
    field: str,
    specification: Any,
    scope: str,
    labels: list[str],
    support: dict[str, int],
    minimum: int,
    fallback: float | dict[str, float],
) -> float | dict[str, float]:
    if scope == "global":
        return _suggest(trial, field, specification)
    default = {"temperature": 1.0, "entity_threshold": 0.5, "entity_bias": 0.0}[field]
    defaults = {
        label: (fallback.get(label, default) if isinstance(fallback, dict) else fallback)
        for label in labels
    }

    return {
        label: (
            _suggest(trial, f"{field}__{label}", specification)
            if support[label] >= minimum
            else defaults[label]
        )
        for label in labels
    }


def _candidate_decoding(
    trial: optuna.Trial,
    space: Mapping[str, Any],
    baseline: BinaryDecoding,
    labels: list[str],
    support: dict[str, int],
    minimum: int,
    threshold_scope: str,
    temperature_scope: str,
) -> BinaryDecoding:
    settings = asdict(baseline)
    settings["decoder"] = _suggest(trial, "decoder", space.get("decoder", baseline.decoder))
    if "activation" in space:
        specification = space["activation"]
        if settings["decoder"] == "constrained_viterbi":
            if isinstance(specification, Mapping):
                choices = [
                    "sigmoid" if value == "temperature_sigmoid" else value
                    for value in specification["categories"]
                ]
                specification = {"type": "choice", "categories": list(dict.fromkeys(choices))}
            elif specification == "temperature_sigmoid":
                specification = "sigmoid"
        settings["activation"] = _suggest(
            trial, f"activation__{settings['decoder']}", specification
        )
    settings["temperature"] = (
        baseline.temperature
        if settings["activation"] == baseline.activation == "temperature_sigmoid"
        else 1.0
    )
    if (
        settings["activation"] == "temperature_sigmoid"
        and settings["decoder"] == "threshold_bio"
        and "temperature" in space
    ):
        settings["temperature"] = _named_setting(
            trial,
            "temperature",
            space["temperature"],
            temperature_scope,
            labels,
            support,
            minimum,
            baseline.temperature,
        )
    settings["entity_threshold"], settings["entity_bias"] = 0.5, 0.0
    field = "entity_threshold" if settings["decoder"] == "threshold_bio" else "entity_bias"
    fallback = (
        getattr(baseline, field)
        if baseline.decoder == settings["decoder"]
        else (0.5 if field == "entity_threshold" else 0.0)
    )
    if (
        field == "entity_bias"
        and baseline.activation == "temperature_sigmoid"
        and baseline.decoder == "constrained_viterbi"
        and settings["activation"] == "sigmoid"
    ):
        fallback = {
            label: (fallback.get(label, 0.0) if isinstance(fallback, dict) else fallback)
            * (
                baseline.temperature.get(label, 1.0)
                if isinstance(baseline.temperature, dict)
                else baseline.temperature
            )
            for label in labels
        }
    settings[field] = (
        _named_setting(
            trial, field, space[field], threshold_scope, labels, support, minimum, fallback
        )
        if field in space
        else fallback
    )

    return BinaryDecoding(**settings)


def fit_binary_decoding(
    rows: pd.DataFrame,
    logits: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    entity_labels: list[str],
    search_space: Mapping[str, Any],
    baseline: BinaryDecoding | None = None,
    metric: str = "span_strict_macro_f1",
    n_trials: int = 40,
    random_state: int | None = None,
    threshold_scope: str = "global",
    temperature_scope: str = "global",
    min_gold_entities: int = 5,
) -> DecodingFit:
    """Search conditional score/decoder settings without any Transformer or training access.

    Temperatures here optimize the selected span metric; they are not a claim of calibrated
    probabilities. Sparse per-entity settings retain baseline defaults. A baseline guard
    rejects a fitting choice worse than the checkpoint's configured decoder.
    """
    validate_decoder_space(search_space)
    if set(search_space) - set(DECODER_FIELDS):
        raise ValueError(f"Decoder search supports only {DECODER_FIELDS}.")
    if threshold_scope not in ("global", "entity") or temperature_scope not in ("global", "entity"):
        raise ValueError("threshold_scope and temperature_scope must be global or entity.")
    if n_trials < 1 or min_gold_entities < 1:
        raise ValueError("n_trials and min_gold_entities must be positive.")
    baseline = baseline or BinaryDecoding()
    gold = rows.attrs.get("multihead_gold")
    if gold is None:
        raise ValueError("Decoder fitting requires original annotated validation gold.")
    gold_frame = span_dataframe(gold)
    gold_frame = gold_frame[gold_frame.filename.isin(rows.doc_id.astype(str))]
    identities = set(
        gold_frame[["filename", "label", "start_span", "end_span"]].itertuples(
            index=False, name=None
        )
    )
    support = {label: sum(record[1] == label for record in identities) for label in entity_labels}
    representation = gold_representation_metrics(rows, tokenizer, entity_labels)

    def score(settings: BinaryDecoding) -> dict:
        spans = multilabel_spans(rows, logits, tokenizer, entity_labels, settings=settings)

        return {**strict_multihead_metrics(gold_frame, spans, entity_labels), **representation}

    best_metrics = score(baseline)
    if metric not in best_metrics:
        raise ValueError(
            f"Unknown decoder metric {metric!r}; select an available span/character metric."
        )
    if not sum(support.values()):
        return DecodingFit(
            baseline,
            best_metrics,
            pd.DataFrame(columns=["trial", metric, *DECODER_FIELDS, "sampled_parameters_json"]),
            support,
        )
    best_settings = baseline
    records = []
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_settings, best_metrics
        selected = _candidate_decoding(
            trial,
            search_space,
            baseline,
            entity_labels,
            support,
            min_gold_entities,
            threshold_scope,
            temperature_scope,
        )
        metrics = score(selected)
        records.append(
            {
                "trial": trial.number,
                **metrics,
                **asdict(selected),
                "sampled_parameters_json": json.dumps(trial.params, sort_keys=True),
            }
        )
        if metrics[metric] > best_metrics[metric]:
            best_metrics, best_settings = metrics, selected
        if not trial.params:
            trial.study.stop()
        return metrics[metric]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return DecodingFit(best_settings, best_metrics, pd.DataFrame(records), support)


def validate_decoder_space(space: Mapping[str, Any]) -> None:
    """Reject invalid domains before fitting; inactive conditional fields are never sampled."""
    allowed = {
        "activation": ("sigmoid", "temperature_sigmoid", "hard_sigmoid"),
        "decoder": ("threshold_bio", "constrained_viterbi"),
    }
    import math

    for name, specification in space.items():
        if name not in DECODER_FIELDS:
            raise ValueError(f"Unknown decoder search field {name!r}.")
        if isinstance(specification, Mapping):
            kind = specification.get("type")
            if kind not in ("choice", "uniform", "loguniform"):
                raise ValueError("Decoder domains support choice, uniform or loguniform only.")
            values = (
                specification.get("categories")
                if kind == "choice"
                else [specification["low"], specification["high"]]
            )
            if not values or kind != "choice" and values[0] > values[1]:
                raise ValueError(f"Empty or reversed decoder domain {name}.")
            if kind == "loguniform" and values[0] <= 0:
                raise ValueError("loguniform requires positive low.")
        else:
            values = [specification]
        if name in allowed:
            if any(value not in allowed[name] for value in values):
                raise ValueError(f"{name} must choose from {allowed[name]}.")
        elif any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or name == "temperature"
            and value <= 0
            or name == "entity_threshold"
            and not 0 <= value <= 1
            for value in values
        ):
            raise ValueError(f"Invalid decoder bounds for {name}.")
