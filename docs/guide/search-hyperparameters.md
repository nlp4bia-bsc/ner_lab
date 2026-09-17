# Search hyperparameters

Runs a Ray Tune + Optuna sweep against a prepared split and emits the winner as a
ready-to-run [`train_model`](train-model.md) config. Trials train with no checkpointing; each
one is scored as the mean across seeds of each seed's mean-of-top-k epochs — a single lucky
epoch never wins a sweep. Out-of-memory trials score worst instead of failing; any other error
halts the sweep immediately rather than burning the remaining trials on the same bug.

```python
from lab.ner import search_hyperparameters

sweep = search_hyperparameters(
    split_dir="assets/splits/disease/kfold_5_holdout_0",
    output_dir="assets/sweeps",
    base_models=[
        "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "dccuchile/bert-base-spanish-wwm-cased",
    ],
    target_label="DISEASE",
    language="es",
    n_trials=40,
)

sweep.summary["best_metric"]     # winning score, mean over seeds
sweep.best                       # the winner as train_model keyword arguments
sweep.trials                     # DataFrame, one row per trial
```

| Parameter | Default | Meaning |
|---|---|---|
| `split_dir` | *required* | A split directory containing `data_manifest.json`. The fixed holdout is never read. |
| `output_dir` | *required* | Root for sweep directories. |
| `base_models` | *required* | One or more pretrained encoders. Part of the search: each becomes variants. |
| `target_label` | *required* | The entity type to tag. |
| `language` | *required* | Sentence-segmentation language. |
| `architecture` | `"linear"` | As in `train_model`. Fixed per sweep, not searched. |
| `architecture_kwargs` | `None` | Extra keywords for the architecture. |
| `search_space` | `None` | Overrides applied to `lab.ner.hpo.DEFAULT_SEARCH_SPACE` — see below. |
| `training_arguments` | `None` | A `TrainingArguments` or a mapping of overrides, as in `train_model`. Defaults add a 40-epoch cap and no checkpointing. Its `metric_for_best_model` is the sweep objective. |
| `n_trials` | `40` | Configurations sampled by Optuna. |
| `seeds_per_trial` | `5` | Trainings per trial; the score averages across them. |
| `top_k_epochs` | `3` | Epochs averaged per seed. |
| `strategies` | `("greedy",)` | Window strategies searched, as one dimension with the base models. |
| `context_tokens` | `None` | Context sizes searched. Required with, and only with, the context strategy. |
| `max_lengths` | `(256,)` | Token budgets searched. |
| `overlap_policy` | `"merge_same_label_then_keep_longest"` | As in `train_model`. |
| `min_sentence_tokens` | `4` | As in `train_model`. |
| `min_overlap_percentage` | `40.0` | As in `train_model`. |
| `early_stopping_patience` | `5` | Decides actual trial length under the epoch cap. |
| `pad_to_multiple_of` | `8` | As in `train_model`. |
| `max_micro_batch_size` | `64` | VRAM ceiling. The searched `effective_train_batch_size` is split into micro batch × accumulation under it, so sweeps stay comparable across GPUs. |
| `validation_index` | first rotatable fold | Which k-fold rotation to search against. Meaningless for a plain split. |
| `random_state` | `None` | Seeds Optuna and the base training seed (per-seed runs offset from it). |
| `study_name`, `storage` | `None` | Optuna persistence, e.g. `sqlite:///study.db`, for resumable sweeps. |
| `smoke_test` | `True` | A one-epoch, single-seed trial per variant, run before the sweep — a config bug or an unloadable base model costs one epoch, not N parallel hours. |
| `track_resources` | `True` | Energy, emissions and peak VRAM per trial. |
| `report` | `True` | Prints the end-of-sweep summary. It is on `HPOResult.report` either way. |
| `overwrite` | `False` | Replace a run already occupying `output_dir` instead of refusing. |

**Returns** an `HPOResult` with `run_dir`, `metric`, `summary`, `best`, `trials`, `manifest`,
`paths` and `report`.

## GPUs are not a parameter

A trial reserves exactly one GPU when Ray reports any, and runs on CPU when it reports none —
so on an 8-GPU node you get eight trials in parallel, never one trial spread across eight.
That is not tunable on purpose: `per_device_train_batch_size` is per device, so a trial
spanning two GPUs would train at twice the batch size it was scored on, and the winner would
not reproduce on the single GPU that final training uses. The resolved value is recorded in
`run_manifest.json`.

## The search space

Each dimension is a Ray Tune domain, a declarative mapping (the YAML form), a scalar (pinned,
not searched), or `null` (removed). Overrides merge over the defaults:

```yaml
search_space:
  learning_rate: {type: loguniform, low: 1.0e-5, high: 1.0e-4}
  weight_decay: null              # remove the dimension
  lr_scheduler_type: linear       # pin it instead of searching it
```

The defaults search `learning_rate`, `weight_decay`, `warmup_steps`,
`effective_train_batch_size` and `lr_scheduler_type`. `warmup_steps` is searched over
`0.0–0.1`, which is a *fraction* of the run: HuggingFace reads any value below 1 as a
proportion of total training steps and anything from 1 up as an absolute step count, so
`0.06` means 6% of the schedule while `50` means fifty steps. On top of them, every
(base_model, strategy, context_tokens, max_length) combination becomes one `variant`
dimension — one dimension rather than four, because the base model's tokenizer decides the
windowing. Each variant is windowed once, before any trial starts, and shared across all of
them.

## What it writes

```
<output_dir>/                  the sweep directory itself: nothing is derived or nested
    run_manifest.json          before the sweep starts: space, variants, provenance
    trials_summary.parquet     one row per trial: sampled values, score, spread, resources
    hpo_summary.json           the winner, raw and as a train_model config
    winner.yaml                the same winner block, runnable by `lab run`
    smoke_test/, trials/       Ray's own per-trial directories
    final_train/               where winner.yaml sends the final training run
```

`winner.yaml` is the `train_model` block from `hpo_summary.json` as a config file, so the
winning trial reaches the assessment stage without being retyped:

```bash
lab run assets/sweeps/<run>/winner.yaml
```

Its `output_dir` is `<the sweep's directory>/final_train`, so it runs as written and keeps the
final model with the sweep that chose it; `--output-dir` sends it elsewhere. Run it — the
sweep's argmax over noisy scores is mildly optimistic, so the reportable number is that
assessment's k-fold mean ± std, not the sweep's.

## From YAML

```yaml
task: ner.search_hyperparameters
split_dir: assets/splits/disease/kfold_5_holdout_0
output_dir: assets/sweeps
base_models:
  - PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
target_label: DISEASE
language: es
n_trials: 40
seeds_per_trial: 5
training_arguments:
  per_device_eval_batch_size: 32
```
