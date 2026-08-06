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

| Stage | Function | State |
|---|---|---|
| Prepare dataset | `prepare_dataset` | available |
| Hyperparameter search | — | not migrated yet |
| Training | — | not migrated yet |
| Inference | — | not migrated yet |
| Evaluation | — | not migrated yet |

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
| `normalize_labels` | `True` | Map label aliases onto `DISEASE`/`PROCEDURE`/`SYMPTOM`/`MEDICATION`, ignoring case and accents. Set `False` to keep source labels verbatim. |
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

```yaml
task: prepare_dataset
documents: corpora/disease/txt
annotations: corpora/disease/ann
output_dir: assets/splits
kfolds: 5
holdout_fold: 0
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

Design decisions, open questions and the migration log live in [`docs/PLAN.md`](docs/PLAN.md).
