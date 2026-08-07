# ner_lab

A library for end-to-end clinical NER: corpus conversion, entity-stratified splitting,
hyperparameter search, training, inference and evaluation.

**Batteries included, but removable.** One call takes you from a BRAT corpus to a trained
model, and every step of it is a public function you can call, replace or skip. Defaults are
opinionated; none of them are compulsory.

You import a function, pass explicit named arguments, and get an object back. Nothing takes a
config dict, and nothing writes to disk unless you asked it to. A thin CLI runs the same
functions from a YAML file for people who would rather not write Python.

```python
prepare_dataset(output_dir=..., documents=..., annotations=...)   # batteries in

corpus = build_corpus(documents, annotations)                     # batteries out
create_split(corpus, "my/own/dir", ratios=[0.8, 0.2])             # your layout, your names
```

Each stage below documents the one-call version. The pieces it is built from are exported
from the same module, and any policy the high-level call applies — label normalization,
validation, provenance manifests, directory layout — is either a keyword argument or a step
you can leave out by composing the pieces yourself.

## Status

| Stage | Entry point | CLI task | State |
|---|---|---|---|
| Prepare dataset | `ner_lab.data.prepare_dataset` | `prepare_dataset` | available |
| Encode a corpus | `ner_lab.encoding.Encoder` | — | available (library only) |
| Build a model | `ner_lab.models.build_model` | — | available (library only) |
| Train one model | `ner_lab.training.train` | — | available (library only) |
| Evaluate | `ner_lab.evaluation.build_compute_metrics` | — | available (library only) |
| Train against a split | `ner_lab.training.train_model` | `train_model` | available |
| Hyperparameter search | `ner_lab.hpo.search_hyperparameters` | `search_hyperparameters` | available |
| Inference | — | — | not migrated yet |

"Library only" means the piece works and is documented below, but has no YAML task yet: it
takes DataFrames and objects rather than paths, so there is no single artifact for a config
file to point at. `train_model` is the call that joins them — it takes a split directory and
assembles the four for you.

## Install

```bash
git clone git@github.com:nlp4bia-bsc/ner_lab.git
cd ner_lab
uv pip install -e .          # or: pip install -e .
```

Requires Python 3.10 or newer. `torch` is pinned to exactly `2.10.0`.

## Quick start

```python
from ner_lab.data import prepare_dataset

prepared = prepare_dataset(
    output_dir="assets/splits",
    documents="corpora/disease/txt",
    annotations="corpora/disease/ann",
    validation_size=0.2,
)

prepared.corpus                 # DataFrame, one row per document
prepared.split.paths["train"]   # Path to the written train.parquet
```

```bash
ner-lab run prepare.yaml
```

Both routes call the same function and produce identical output.

## The canonical corpus

Every stage after conversion speaks one schema, one row per document:
`doc_id | text | entities_json | n_entities`, where `entities_json` is a JSON list of
`{id, start, end, label, text}`. Offsets are zero-based with an exclusive end, and
`text[start:end] == entity["text"]` is validated on the way in.

Entities are stored **as annotated** — overlaps are preserved, and resolving them is a
modelling choice made at encoding time, so several policies can be compared without
regenerating the corpus. One consequence: `n_entities` and the balance report describe the
corpus, not the entity set any particular training run sees.

---

## 1. Prepare dataset

Converts a source corpus to the canonical parquet, splits it, and records provenance.

```python
from ner_lab.data import prepare_dataset

prepare_dataset(
    output_dir,
    documents=None,
    annotations=None,
    source_parquet=None,
    dataset_name=None,
    split=True,
    validation_size=0.2,
    kfolds=None,
    holdout_fold=0,
    random_state=20260701,
    n_restarts=64,
    reuse_assignments=True,
    corpus_filename="documents.parquet",
    compression="zstd",
) -> PreparedDataset
```

| Parameter | Default | Meaning |
|---|---|---|
| `output_dir` | *required* | Root directory. Output goes to `<output_dir>/<dataset_name>/<split_descriptor>/`. |
| `documents` | `None` | Directory of `.txt` files, or a `{name: text}` mapping. Requires `annotations`. |
| `annotations` | `None` | A DataFrame, a `.tsv` file, a single `.ann` file, or a directory of `.ann` files. Only valid with `documents`. |
| `source_parquet` | `None` | An existing canonical parquet, skipping conversion. Mutually exclusive with `documents`. |
| `dataset_name` | derived | Overrides the derived name (from a sibling `metadata.json`, else the documents directory, else the parquet stem). |
| `normalize_labels` | `False` | Labels are stored exactly as the source annotations write them. Set `True` to map them through `ner_lab.data.LABEL_ALIASES` onto `DISEASE`/`PROCEDURE`/`SYMPTOM`/`MEDICATION`, ignoring case and accents. |
| `split` | `True` | Set `False` to write only the canonical corpus. |
| `validation_size` | `0.2` | Validation fraction for a plain train/validation split. Ignored when `kfolds` is set. |
| `kfolds` | `None` | Number of fixed-holdout cross-validation folds. Minimum 2. |
| `holdout_fold` | `0` | Which fold is the permanently reserved test partition. Only valid with `kfolds`. |
| `random_state` | `20260701` | Seed for stratification. |
| `n_restarts` | `64` | Randomized stratification attempts to keep the best of. |
| `reuse_assignments` | `True` | Reuse an existing split if present, after validating it against the corpus. Set `False` to rebuild. |
| `corpus_filename` | `documents.parquet` | Filename for the converted canonical parquet. |
| `compression` | `zstd` | Parquet compression codec. |

Give either `documents` plus `annotations`, or `source_parquet`. Passing both, or neither,
raises.

**Returns** a `PreparedDataset` with `corpus`, `corpus_path`, `dataset_root` and
`source_manifest`, plus `split`, `split_dir` and `data_manifest` when `split=True`.

**Writes:**

```
<output_dir>/<dataset_name>/
    documents.parquet             canonical corpus
    source_manifest.json          source paths, sha256, upstream metadata
    <split_descriptor>/
        split_assignments.parquet  source of truth for the split
        split_balance_report.parquet / .tsv
        data_manifest.json         split mode, seed, partition summary
        train.parquet, validation.parquet
        or: test.parquet, fold_01.parquet, ...
```

`<split_descriptor>` is `train_val_80_20` for a plain split, `kfold_5_holdout_0` for
cross-validation. With `kfolds`, `test.parquet` is the fixed holdout and is never touched
during model selection.

An existing split is reused rather than regenerated, so reruns stay reproducible even if the
seed changes. Every document is fingerprinted and checked against the manifest first — the
run fails if any document was added, removed or edited since the split was made.

**Label normalization** is off by default: a corpus keeps the labels its annotations declare.
With `normalize_labels=True`, labels are canonicalized (accents stripped, uppercased, `-` and
spaces to `_`) and then looked up in `ner_lab.data.LABEL_ALIASES`, which folds the Spanish and
plural spellings onto `DISEASE`/`PROCEDURE`/`SYMPTOM`/`MEDICATION`. Labels absent from the
table are canonicalized but never renamed. For a corpus with its own vocabulary, normalize the
annotations yourself and pass the result to `build_corpus`:

```python
from ner_lab.data import build_corpus, read_annotations, normalize_annotation_labels

annotations = normalize_annotation_labels(
    read_annotations("corpora/genes/ann"),
    aliases={"GEN": "GENE", "GENES": "GENE"},
)
corpus = build_corpus("corpora/genes/txt", annotations)
```

```yaml
task: prepare_dataset
documents: corpora/disease/txt
annotations: corpora/disease/ann
output_dir: assets/splits
kfolds: 5
holdout_fold: 0
```

---

## 2. Train a model against a split

Takes a split directory produced by stage 1 and scores one configuration against it. The
split's `data_manifest.json` decides the shape of the run: a plain train/validation split
trains once, a fixed-holdout k-fold split trains once per rotatable fold and aggregates. The
fixed holdout is never read.

```python
from ner_lab.training import train_model

assessment = train_model(
    split_dir="assets/splits/disease/kfold_5_holdout_0",
    output_dir="assets/runs",
    checkpoint="PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    target_label="DISEASE",
    language="es",
    architecture="crf",
    training_arguments={"learning_rate": 2e-5, "num_train_epochs": 8},
)

assessment.fold_metrics                    # DataFrame, one row per fold
assessment.aggregate["best_metric"]        # {"mean": ..., "std": ...}
assessment.run_dir                         # where everything was written
```

| Parameter | Default | Meaning |
|---|---|---|
| `split_dir` | *required* | A split directory containing `data_manifest.json`. |
| `output_dir` | *required* | Root for run directories. |
| `checkpoint` | *required* | Pretrained encoder, by hub name or local path. |
| `target_label` | *required* | The entity type to tag. One label, three classes (D43). |
| `language` | *required* | Sentence-segmentation language for `pysbd`. Never guessed (D33). |
| `architecture` | `"linear"` | `"linear"`, `"crf"`, or your own `(checkpoint, label2id, id2label, **kwargs) -> model`. |
| `architecture_kwargs` | `None` | Extra keywords for the chosen architecture, e.g. `{"dropout": 0.2}`. |
| `training_arguments` | `None` | A `TrainingArguments`, or a mapping of overrides applied to `ner_lab.training.DEFAULTS`. The YAML path uses the mapping. |
| `max_length` | `256` | Token budget per window, including special tokens. |
| `strategy` | `"greedy"` | Window strategy, or your own callable. |
| `context_tokens` | `None` | Flanking context per window. Only valid with `strategy="context"`. |
| `overlap_policy` | `"merge_same_label_then_keep_longest"` | How overlapping gold spans are resolved at encoding time. |
| `min_sentence_tokens` | `4` | Sentences shorter than this are merged into their neighbour. |
| `early_stopping_patience` | `5` | Epochs without improvement before stopping. `None` disables it. |
| `pad_to_multiple_of` | `8` | Pad batches to a multiple of this. |
| `metrics_scope` | `"eval"` | `"both"` also scores the training set each epoch, at the cost of a second pass. |
| `include_confusion` | `False` | Add the token confusion matrix to each evaluation. |
| `save_model` | `False` | Keep weights. Off because this scores a configuration; it does not build a deployment artifact. |
| `track_resources` | `True` | Wall clock, energy, emissions and peak VRAM per run, via codecarbon. |
| `folds` | `None` | Narrow a k-fold run to specific validation folds. Invalid without a fixed holdout. |
| `random_state` | `None` | Overrides `TrainingArguments.seed`. |
| `run_name` | derived | Overrides the derived `<target_label>__<architecture>__<checkpoint>__<timestamp>` directory name. |

Everything after `language` configures the `Encoder`, `build_model` and `train` that this
assembles. Build those yourself and call `ner_lab.training.train` if the assembly is in the
way — the orchestrator applies no policy you cannot reach by composing them.

**Returns** an `AssessmentResult` with `run_dir`, `mode`, `fold_metrics`, `aggregate`,
`summaries`, `manifest` and `paths`. It deliberately does **not** return the `Trainer` or the
model: one per fold would keep every fold's weights resident while the next one trains. Use
`save_model=True`, or call `train` directly.

**Writes:**

```
<output_dir>/<target_label>__<architecture>__<checkpoint>__<timestamp>/
    run_manifest.json          written before training starts, so a crash is debuggable
    fold_metrics.parquet       one row per run: selection metric, best epoch, resources
    assessment_summary.json    mean and standard deviation across runs
    fold_01/                   per-fold, or the run directory itself for a plain split
        epoch_metrics.parquet
        training_summary.json
        best_model/            only with save_model=True
    fold_02/
```

`metric_for_best_model` defaults to `span_strict_f1`; the `compute_metrics` that produces it
is built for you from each run's own validation rows.

```yaml
task: train_model
split_dir: assets/splits/disease/kfold_5_holdout_0
output_dir: assets/runs
checkpoint: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
target_label: DISEASE
language: es
architecture: crf
max_length: 256
training_arguments:
  learning_rate: 2.0e-5
  num_train_epochs: 8
  per_device_train_batch_size: 16
```

---

## 3. Search hyperparameters

Runs a Ray Tune + Optuna sweep against a prepared split and emits the winner as a
ready-to-run `train_model` config. Trials train with no checkpointing; each one is scored as
the mean across seeds of each seed's mean-of-top-k epochs — a single lucky epoch never wins a
sweep. Out-of-memory trials score worst instead of failing; any other error halts the sweep
immediately rather than burning the remaining trials on the same bug.

```python
from ner_lab.hpo import search_hyperparameters

sweep = search_hyperparameters(
    split_dir="assets/splits/disease/kfold_5_holdout_0",
    output_dir="assets/sweeps",
    checkpoints=[
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
| `checkpoints` | *required* | One or more pretrained encoders. Part of the search: each becomes variants. |
| `target_label` | *required* | The entity type to tag. |
| `language` | *required* | Sentence-segmentation language. |
| `architecture` | `"linear"` | As in `train_model`. Fixed per sweep, not searched. |
| `architecture_kwargs` | `None` | Extra keywords for the architecture. |
| `search_space` | `None` | Overrides applied to `ner_lab.hpo.DEFAULT_SEARCH_SPACE` — see below. |
| `training_arguments` | `None` | A `TrainingArguments` or a mapping of overrides, as in `train_model`. Defaults add a 40-epoch cap and no checkpointing. Its `metric_for_best_model` is the sweep objective. |
| `n_trials` | `40` | Configurations sampled by Optuna. |
| `seeds_per_trial` | `5` | Trainings per trial; the score averages across them. |
| `top_k_epochs` | `3` | Epochs averaged per seed. |
| `strategies` | `("greedy",)` | Window strategies searched, as one dimension with the checkpoints. |
| `context_tokens` | `None` | Context sizes searched. Required with, and only with, the context strategy. |
| `max_lengths` | `(256,)` | Token budgets searched. |
| `overlap_policy` | `"merge_same_label_then_keep_longest"` | As in `train_model`. |
| `min_sentence_tokens` | `4` | As in `train_model`. |
| `early_stopping_patience` | `5` | Decides actual trial length under the epoch cap. |
| `pad_to_multiple_of` | `8` | As in `train_model`. |
| `max_micro_batch_size` | `64` | VRAM ceiling. The searched `effective_train_batch_size` is split into micro batch × accumulation under it, so sweeps stay comparable across GPUs. |
| `gpus_per_trial` | `1.0` | Ray resource request per trial. Fractions share a GPU; `0` runs on CPU. |
| `validation_index` | first rotatable fold | Which k-fold rotation to search against. Meaningless for a plain split. |
| `random_state` | `None` | Seeds Optuna and the base training seed (per-seed runs offset from it). |
| `study_name`, `storage` | `None` | Optuna persistence, e.g. `sqlite:///study.db`, for resumable sweeps. |
| `smoke_test` | `True` | One fixed-config, single-seed trial run to completion first — a config bug costs one run, not N parallel hours. |
| `track_resources` | `True` | Energy, emissions and peak VRAM per trial. |
| `run_name` | derived | Overrides the derived sweep directory name. |

**The search space.** Each dimension is a Ray Tune domain, a declarative mapping (the YAML
form), a scalar (pinned, not searched), or `null` (removed). Overrides merge over the
defaults:

```yaml
search_space:
  learning_rate: {type: loguniform, low: 1.0e-5, high: 1.0e-4}
  weight_decay: null              # remove the dimension
  lr_scheduler_type: linear       # pin it instead of searching it
```

The defaults search `learning_rate`, `weight_decay`, `warmup_ratio`,
`effective_train_batch_size` and `lr_scheduler_type`. On top of them, every
(checkpoint, strategy, context_tokens, max_length) combination becomes one `variant`
dimension — one dimension rather than four, because the checkpoint's tokenizer decides the
windowing. Each variant is windowed once, before any trial starts, and shared across all of
them.

**Returns** an `HPOResult` with `run_dir`, `metric`, `summary`, `best`, `trials`, `manifest`
and `paths`.

**Writes:**

```
<output_dir>/<target_label>__<architecture>__<checkpoint>__<timestamp>/
    run_manifest.json          before the sweep starts: space, variants, provenance
    trials_summary.parquet     one row per trial: sampled values, score, spread, resources
    hpo_summary.json           the winner, raw and as a train_model config
    smoke_test/, trials/       Ray's own per-trial directories
```

The `train_model` block inside `hpo_summary.json` is copy-pasteable into a YAML file and
runnable as-is — the sweep's argmax over noisy scores is mildly optimistic, so the reportable
number is that assessment's k-fold mean ± std, not the sweep's.

```yaml
task: search_hyperparameters
split_dir: assets/splits/disease/kfold_5_holdout_0
output_dir: assets/sweeps
checkpoints:
  - PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
target_label: DISEASE
language: es
n_trials: 40
seeds_per_trial: 5
training_arguments:
  per_device_eval_batch_size: 32
```

---

## Command line

```
ner-lab --version
ner-lab run <config.yaml> [--random-state N] [--output-dir PATH]
```

There is one command. Every parameter lives in the YAML file; the two flags exist only so a
job array can vary them without writing near-identical config files.

| Flag | Meaning |
|---|---|
| `--random-state N`, `--seed N` | Override the config's `random_state`. |
| `--output-dir PATH` | Override the config's `output_dir`. |

The config is a flat mapping. A `task` key names the stage; every other key is passed to that
stage's function as a keyword argument, so **the YAML keys are exactly the parameter names
documented above**.

```bash
ner-lab run prepare.yaml
ner-lab run prepare.yaml --seed 7 --output-dir assets/splits_seed7
```

Exit codes: `0` on success, `1` when called with no command, `2` on a bad config — unknown
task, missing `task` key, missing file, or an unknown parameter name.

### Running on a cluster

The library ships nothing SLURM-specific. Write your own submit script around the CLI or the
Python API:

```bash
#!/bin/bash
#SBATCH --job-name=prepare_disease
#SBATCH --cpus-per-task=8

ner-lab run prepare.yaml
```

## Development notes

Working documents live in [`docs/`](docs/):

| File | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Governing principle, hard rules, layering, how we work |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Numbered decisions and their rationale |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Migration status, open questions, known baggage |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | What landed, and the evidence for it |

Behavioural checks live in [`verification/`](verification/) — outside `src/`, so they are
never packaged. They are plain scripts, not a test suite:

```bash
.venv-verify/bin/python verification/run_all.py
```

See [`verification/README.md`](verification/README.md) for the environment.
