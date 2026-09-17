# lab

The unit's NLP library: one repository, one import namespace, one CLI. `lab.core` holds the
data contracts — corpus conversion, entity-stratified splitting, corpus statistics, span
scoring — with no `torch`. Each task lives in its own subpackage on top of it; `lab.ner` is
clinical NER end to end: encoding, hyperparameter search, training, inference and evaluation.

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
uv pip install -e ".[ner]"     # or: pip install -e ".[ner]"
```

Requires Python 3.10 or newer. `lab` alone is torch-free and gives corpus conversion,
splitting, statistics and span scoring. `lab[ner]` adds the NER stack; it pins `torch` to
exactly `2.10.0` through the shared `lab[torch]` extra every tool builds on.

## Quick start

```python
from lab.core import prepare_dataset

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
lab run prepare.yaml
```

Both routes call the same function and produce identical output.

## The stages

| Stage | Entry point | CLI task | Reference |
|---|---|---|---|
| Prepare a dataset | `lab.core.prepare_dataset` | `core.prepare_dataset` | [guide/prepare-dataset.md](docs/guide/prepare-dataset.md) |
| Train against a split | `lab.ner.train_model` | `ner.train_model` | [guide/train-model.md](docs/guide/train-model.md) |
| Search hyperparameters | `lab.ner.search_hyperparameters` | `ner.search_hyperparameters` | [guide/search-hyperparameters.md](docs/guide/search-hyperparameters.md) |
| Predict with a model | `lab.ner.predict_entities` | `ner.predict_entities` | [guide/predict-entities.md](docs/guide/predict-entities.md) |
| Encode a corpus | `lab.ner.Encoder` | — | [guide/library.md](docs/guide/library.md) |
| Build a model | `lab.ner.build_model` | — | [guide/library.md](docs/guide/library.md) |
| Train one model | `lab.ner.train` | — | [guide/library.md](docs/guide/library.md) |
| Score predictions | `lab.ner.build_compute_metrics` | — | [guide/library.md](docs/guide/library.md) |

The four with no CLI task take DataFrames and objects rather than paths, so there is no single
artifact for a config file to point at. `train_model` is the call that joins them — it takes a
split directory and assembles them for you.

Task names carry their subpackage. `lab tasks` lists them and marks any whose extra is not
installed. The CLI itself — one command, its two flags, the config format — is
[guide/cli.md](docs/guide/cli.md).

## Importing

`lab.core` exports its contracts and every function around them directly:

```python
from lab.core import prepare_dataset, read_corpus, build_corpus, create_split, score_spans
```

The nine functions that start a NER stage — or that you build to start one — are importable
directly from `lab.ner`:

```python
from lab.ner import (
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

Everything else keeps its subpackage path — `lab.ner.encoding`, `lab.ner.models`,
`lab.ner.training`, `lab.ner.evaluation`, `lab.ner.hpo` — and so do the nine above, so
`from lab.ner.training import train_model` remains correct. Names on `lab.ner` resolve on
first use, so `import lab.ner` loads nothing.

## Layout

```
lab/
├── core/    data contracts, I/O, provenance, task registry.  No torch.
├── ner/     clinical NER.                                    torch, via lab[ner].
└── cli.py   one entry point, `lab`.
```

Imports point one way: `core` never imports a task subpackage, and task subpackages never
import each other. They compose through `core`'s two tables instead, so a linker or a
cross-lingual comparison works on spans from this NER, from gold annotations, or from a
system outside the library. [`docs/dev/RESTRUCTURE.md`](docs/dev/RESTRUCTURE.md) is the
argument.

## The canonical corpus

Every stage after conversion speaks one schema, one row per document:
`doc_id | text | entities_json | n_entities`, where `entities_json` is a JSON list of
`{id, start, end, label, text}`. Offsets are zero-based with an exclusive end, and
`text[start:end] == entity["text"]` is validated on the way in.

Entities are stored **as annotated** — overlaps are preserved, and resolving them is a
modelling choice made at encoding time, so several policies can be compared without
regenerating the corpus. One consequence: `n_entities` and the balance report describe the
corpus, not the entity set any particular training run sees.

The second table is the span table, one row per mention:
`filename | label | start_span | end_span | text [| score]`. Inference writes it, gold `.ann`
files read into it, and `lab.core.score_spans` scores any two of them against each other.

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
