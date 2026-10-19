# `lab.ner` — clinical NER

**Authors**: Aziz Ouhida · mohamed.ouhida@bsc.es; Fernando Gallego · fernando.gallego@bsc.es

Encoding, hyperparameter search, training, inference and evaluation, end to end. Installed
by `lab[ner]`.

| Page | Covers |
|---|---|
| [train-model.md](train-model.md) | `train_model`: scoring one configuration against a split, single or k-fold |
| [search-hyperparameters.md](search-hyperparameters.md) | `search_hyperparameters`: the Ray Tune + Optuna sweep and its search space |
| [predict-entities.md](predict-entities.md) | `predict_entities`: inference with a saved model, and scoring it |
| [ensemble.md](ensemble.md) | `fit_ensemble` and `predict_ensemble`: combining saved models by word-level voting |
| [library.md](library.md) | The pieces the orchestrators are built from: `Encoder`, `build_model`, `training_arguments`, `train`, the metrics builders |

| CLI task | Function |
|---|---|
| `ner.train_model` | `lab.ner.train_model` |
| `ner.search_hyperparameters` | `lab.ner.search_hyperparameters` |
| `ner.predict_entities` | `lab.ner.predict_entities` |
| `ner.fit_ensemble` | `lab.ner.fit_ensemble` |
| `ner.predict_ensemble` | `lab.ner.predict_ensemble` |

The stages feed each other through `lab.core`'s tables: `train_model` takes a split
directory from [`prepare_dataset`](../core/prepare-dataset.md), `predict_entities` writes a
span table, and [`link_entities`](../nel/link-entities.md) takes that table as it is.

## Importing

The eleven functions that start a stage — or that you build to start one — are importable
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
    fit_ensemble,           # combine saved models into an ensemble
    predict_ensemble,       # inference with a fitted ensemble
)
```

Everything else keeps its subpackage path — `lab.ner.encoding`, `lab.ner.models`,
`lab.ner.training`, `lab.ner.evaluation`, `lab.ner.hpo`, `lab.ner.ensemble` — and so do the eleven above, so
`from lab.ner.training import train_model` remains correct. Names on `lab.ner` resolve on
first use, so `import lab.ner` loads nothing.
