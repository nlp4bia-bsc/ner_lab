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
| `seeds_per_trial` | `3` | Trainings per trial; the score averages across them. |
| `top_k_epochs` | `3` | Epochs averaged per seed. |
| `strategies` | `("greedy",)` | Window strategies searched, as one dimension with the base models. |
| `context_tokens` | `None` | Context sizes searched. Required with, and only with, the context strategy. |
| `max_lengths` | `(256,)` | Token budgets searched. Before Ray starts, each base model runs one CPU forward pass at each length and prints `[check] <model>: a forward pass at max_length=N passed`; a length past the model's positions raises naming it (D115). |
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
seeds_per_trial: 3
training_arguments:
  per_device_eval_batch_size: 32
```

## Multi-Head search

Use `architecture="multihead"`, `target_label=None` (or omit it in YAML), and an explicit
`language`. There is one joint search over all heads. The training partition of the chosen
rotation fixes a deterministic catalogue; validation and holdout cannot add heads. Each
encoded variant retains that catalogue and shares its rows across trials and seeds.
The existing capacity check, Ray Tune/Optuna sampler, top-k epoch mean, OOM handling and
winner machinery are reused. The default objective becomes `span_strict_macro_f1`;
`training_arguments.metric_for_best_model` overrides it, including per-head F1 or loss.
`winner.yaml` keeps Multi-Head, automatic discovery, window/overlap settings and the selection
metric. Re-running it rediscovers heads from each final-training fold's authorized train;
those catalogues can differ from the searched rotation. See [fold semantics](library.md#strict-metrics).


## CRF search dimensions

`crf` and `multihead_crf` participate in the existing Ray Tune + Optuna sweep, including
cached encoded variants, multiple seeds, OOM handling, epoch scoring and `winner.yaml`.
The latter defaults to strict macro-F1 and uses training-only head discovery.

Namespaced dimensions in the existing `search_space` route to `architecture_kwargs`, never
to `TrainingArguments`: `architecture__dropout`, `architecture__aux_ce_weight`,
`architecture__constrain_transitions`, `architecture__loss_type`,
`architecture__loss_reduction`, `architecture__tag_scheme`. Only `word_mean` and `BIO2`
are accepted for the last two. CRF dimensions are rejected for non-CRF architectures.
Architecture itself remains fixed per sweep; run comparable explicit sweeps to compare
linear/CRF/Multi-Head families, without sampling irrelevant conditional parameters.

[Classic CRF search](configs/crf-hpo.yaml) and
[Multi-Head CRF search](configs/multihead-crf-hpo.yaml) are valid `lab run` templates.
The winning namespaced values are converted back to ordinary `architecture_kwargs` in
`winner.yaml`; replay it with `lab run`. Per-seed `epoch_metrics.parquet` records per-type
metrics while the trial table reports the selected objective and its seed dispersion.

Search validation rejects negative/nonfinite bounds, unsupported schemes and searches that
combine the constrained-CE baseline with nonzero auxiliary CE. Compare those in separate
sweeps to avoid impossible trials. Namespaced options apply to version-2 CRFs only.


## Multi-Label: training search versus cached decoder search

`architecture="multilabel"` uses the same Ray Tune/Optuna training search, multiple seeds,
variant cache, OOM handling and winner replay. Default objective is strict macro-F1.
`loss_variants` is a sequence of the typed `BinaryLoss` schema (YAML mappings use exactly its
fields). Each candidate describes one applicable loss family, e.g. BCE, weighted BCE,
focal with gamma/alpha, or ASL with positive/negative exponents and clipping. One categorical
profile is sampled, avoiding Cartesian products of irrelevant parameters. Place loss options
inside profiles; outside them, `architecture_kwargs` can fix dropout/decoder settings.
`search_space.architecture__dropout` remains independently searchable.

Alternatively fix one loss in `architecture_kwargs` and search its applicable namespaced
fields: `architecture__focal_gamma`, `architecture__focal_alpha`, or ASL's
`architecture__gamma_pos`, `architecture__gamma_neg`, `architecture__clip`; weighted BCE's
`architecture__weight_smoothing`, `architecture__max_pos_weight`,
`architecture__pos_weight_policy`. `architecture__loss` may compare default loss families,
and `architecture__training_temperature` changes the training objective. Incompatible
fields fail before trials. `winner.yaml` retains the exact selected constructor arguments.
[Training HPO YAML](configs/multilabel-hpo.yaml).

For transforms/thresholds/decoders, use the **same entry point** with `model_dir` and
`decoder_search_space`, omitting `base_models` and training loss arguments:

```python
selected = search_hyperparameters(
    split_dir="assets/splits/calibration/train_val_80_20",
    output_dir="assets/sweeps/decoder", model_dir="assets/runs/multilabel/best_model",
    decoder_search_space={
        "activation": {"type": "choice", "categories": ["sigmoid", "temperature_sigmoid", "hard_sigmoid"]},
        "temperature": {"type": "loguniform", "low": 0.5, "high": 2.0},
        "entity_threshold": {"type": "uniform", "low": 0.2, "high": 0.8},
    },
    threshold_scope="entity", temperature_scope="entity", n_trials=40, random_state=42,
)
```

[Transform/decoder HPO YAML](configs/multilabel-decoder-hpo.yaml) also compares constrained
Viterbi and its `entity_bias`. The checkpoint runs **once** over the selected validation
partition; all Optuna trials reuse raw logits. Temperature is sampled only for
`temperature_sigmoid`, threshold only for threshold BIO, bias only for Viterbi. No Transformer
retraining occurs. The pure `lab.ner.hpo.decoding.fit_binary_decoding` accepts precomputed
logits for the same operation without I/O. Positive temperature preserves ranking; this is
post-processing selection, not a claim that an equivalent transform is a better model.

`threshold_scope` and `temperature_scope` accept global/entity. Entity scope fits only labels
with at least `min_gold_entities=5` unique gold mentions, falling back to checkpoint settings
for sparse labels. A checkpoint-decoder baseline guard retains it unless a candidate improves
the selected metric. Temperature selection optimizes span F1 here, **not calibrated NLL**.
A reserved independent calibration/validation set or OOF predictions are required for a
credible experiment. Known training/calibration overlap raises; missing source evidence
warns and records unknown separation. Holdout files are never read to fit choices.

Artifacts retain the existing HPO summary/trial/winner names and add `decoding.json`, with
checkpoint fingerprints, fixed choices, per-type support and calibration provenance.
Dictionary-valued trial fields use `*_json` Parquet columns. `predict_entities` can consume
the decoder artifact explicitly. `winner.yaml` describes retraining with those loss and
prediction settings, preserving the chosen fold and window configuration; **revalidate
thresholds and temperature after retraining**, since new weights or data invalidate their
original fitting evidence. Neither inference nor ensembles optimize them automatically.

For span-metric Viterbi search, a positive temperature with zero transition potentials does
not add a hard-decision degree of freedom: per-position O/B/I log-score differences are
`0,zB/T,zI/T`. With a state bias, T trades with that bias. The search therefore does not
sample temperature for Viterbi and collapses searched temperature-sigmoid to standard sigmoid
in that branch. Configured checkpoint temperatures remain valid for confidence; baseline
guards preserve original choices on ties. `threshold_scope="entity"` also scopes Viterbi's
membership bias. Named checkpoint settings are retained when their field is not searched.
No known gold support retains the baseline without optimization; a fully pinned candidate
is evaluated once. Requested and executed trial counts are recorded separately.

Native Multi-Label training winners set `save_model: true`, so replay also keeps a reloadable
`best_model`. Other architectures retain their previous winner defaults.
