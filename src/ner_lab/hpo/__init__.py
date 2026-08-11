"""Hyperparameter search: Ray Tune + Optuna over a prepared split."""

from __future__ import annotations

from ner_lab.hpo.search import (
    HPO_ARGUMENT_DEFAULTS,
    HPOResult,
    best_trial,
    resolve_base_arguments,
    search_hyperparameters,
    trial_directory_name,
    trials_table,
    user_argument_overrides,
    winner_configuration,
    write_winner_config,
)
from ner_lab.hpo.progress import SweepProgress
from ner_lab.hpo.report import summarize_sweep
from ner_lab.hpo.space import (
    DEFAULT_SEARCH_SPACE,
    build_domain,
    build_search_space,
    describe_search_space,
    resolve_batch_sizes,
    smoke_configuration,
)
from ner_lab.hpo.trial import (
    is_oom_error,
    metric_greater_is_better,
    run_trial,
    top_k_epoch_mean,
    validate_search_space,
)
from ner_lab.hpo.variants import (
    EncodedVariant,
    build_variants,
    describe_variants,
    encode_variants,
    variant_key,
)

__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "HPO_ARGUMENT_DEFAULTS",
    "EncodedVariant",
    "HPOResult",
    "SweepProgress",
    "best_trial",
    "build_domain",
    "build_search_space",
    "build_variants",
    "describe_search_space",
    "describe_variants",
    "encode_variants",
    "is_oom_error",
    "metric_greater_is_better",
    "resolve_base_arguments",
    "resolve_batch_sizes",
    "run_trial",
    "search_hyperparameters",
    "smoke_configuration",
    "summarize_sweep",
    "top_k_epoch_mean",
    "trial_directory_name",
    "trials_table",
    "user_argument_overrides",
    "validate_search_space",
    "variant_key",
    "winner_configuration",
    "write_winner_config",
]
