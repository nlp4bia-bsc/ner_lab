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

Any policy a high-level call applies — label normalization, validation, provenance manifests,
directory layout — is either a keyword argument or a step you can leave out by composing the
pieces yourself. [`docs/guide/library.md`](docs/guide/library.md) is the composed version of
the whole pipeline.

## Install

```bash
git clone git@github.com:nlp4bia-bsc/ner_lab.git
cd ner_lab
uv pip install -e .          # or: pip install -e .
```

Requires Python 3.10 or newer. `torch` is pinned to exactly `2.10.0`.

## Quick start

```python
from ner_lab import prepare_dataset

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

## The stages

| Stage | Entry point | CLI task | Reference |
|---|---|---|---|
| Prepare a dataset | `ner_lab.prepare_dataset` | `prepare_dataset` | [guide/prepare-dataset.md](docs/guide/prepare-dataset.md) |
| Train against a split | `ner_lab.train_model` | `train_model` | [guide/train-model.md](docs/guide/train-model.md) |
| Search hyperparameters | `ner_lab.search_hyperparameters` | `search_hyperparameters` | [guide/search-hyperparameters.md](docs/guide/search-hyperparameters.md) |
| Predict with a model | `ner_lab.predict_entities` | `predict_entities` | [guide/predict-entities.md](docs/guide/predict-entities.md) |
| Encode a corpus | `ner_lab.Encoder` | — | [guide/library.md](docs/guide/library.md) |
| Build a model | `ner_lab.build_model` | — | [guide/library.md](docs/guide/library.md) |
| Train one model | `ner_lab.train` | — | [guide/library.md](docs/guide/library.md) |
| Score predictions | `ner_lab.build_compute_metrics` | — | [guide/library.md](docs/guide/library.md) |

The four with no CLI task take DataFrames and objects rather than paths, so there is no single
artifact for a config file to point at. `train_model` is the call that joins them — it takes a
split directory and assembles them for you.

The CLI itself — one command, its two flags, the config format — is
[guide/cli.md](docs/guide/cli.md).

## Importing

The ten functions that start a stage — or that you build to start one — are importable
directly from `ner_lab`:

```python
from ner_lab import (
    prepare_dataset,        # corpus + split
    Encoder,                # corpus -> model-ready rows
    build_model,            # architecture factory
    training_arguments,     # transformers.TrainingArguments with this library's defaults
    train,                  # one training run
    train_model,            # the training orchestrator (single split or k-fold)
    build_compute_metrics,  # span metrics for train(compute_metrics=...)
    evaluate_predictions,   # score predictions against gold
    search_hyperparameters, # HPO sweep
    predict_entities,       # inference with a saved model
)
```

Everything else keeps its subpackage path — `ner_lab.data`, `ner_lab.encoding`,
`ner_lab.models`, `ner_lab.training`, `ner_lab.evaluation`, `ner_lab.hpo` — and so do the ten
above, so `from ner_lab.training import train_model` remains correct.

Names resolve on first use, so `import ner_lab` imports nothing and `ner_lab.data`,
`ner_lab.encoding` and `ner_lab.evaluation` do not require `torch`.

## The canonical corpus

Every stage after conversion speaks one schema, one row per document:
`doc_id | text | entities_json | n_entities`, where `entities_json` is a JSON list of
`{id, start, end, label, text}`. Offsets are zero-based with an exclusive end, and
`text[start:end] == entity["text"]` is validated on the way in.

Entities are stored **as annotated** — overlaps are preserved, and resolving them is a
modelling choice made at encoding time, so several policies can be compared without
regenerating the corpus. One consequence: `n_entities` and the balance report describe the
corpus, not the entity set any particular training run sees.

## Documentation

[`docs/`](docs/) has two halves: [`docs/guide/`](docs/guide/) is the parameter reference for
everything above, and [`docs/dev/`](docs/dev/) is how the library is built —
[DESIGN.md](docs/dev/DESIGN.md) for the governing principle and hard rules,
[DECISIONS.md](docs/dev/DECISIONS.md) for numbered decisions and their rationale,
[ROADMAP.md](docs/dev/ROADMAP.md) for what is open, [PROGRESS.md](docs/dev/PROGRESS.md) for
what landed and the evidence for it.

Behavioural checks live in [`verification/`](verification/) — outside `src/`, so they are
never packaged. They are plain scripts, not a test suite:

```bash
.venv-verify/bin/python verification/run_all.py
```

See [`verification/README.md`](verification/README.md) for the environment.
