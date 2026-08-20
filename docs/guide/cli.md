# Command line

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

`--random-state` reaches whichever task is named, so it fails on `predict_entities`, which has
no seed to set — inference is deterministic.

## The config

A flat mapping. A `task` key names the stage; every other key is passed to that stage's
function as a keyword argument, so **the YAML keys are exactly the parameter names** the guide
pages document.

| `task` | Function | Parameters |
|---|---|---|
| `prepare_dataset` | `ner_lab.prepare_dataset` | [prepare-dataset.md](prepare-dataset.md) |
| `train_model` | `ner_lab.train_model` | [train-model.md](train-model.md) |
| `search_hyperparameters` | `ner_lab.search_hyperparameters` | [search-hyperparameters.md](search-hyperparameters.md) |
| `predict_entities` | `ner_lab.predict_entities` | [predict-entities.md](predict-entities.md) |

The [pieces underneath](library.md) have no task: they take DataFrames and objects rather than
paths, so there is no single artifact for a config file to point at.

```bash
ner-lab run prepare.yaml
ner-lab run prepare.yaml --seed 7 --output-dir assets/splits_seed7
```

Exit codes: `0` on success, `1` when called with no command, `2` on a bad config — unknown
task, missing `task` key, missing file, or an unknown parameter name.

## Running on a cluster

The library ships nothing SLURM-specific. Write your own submit script around the CLI or the
Python API:

```bash
#!/bin/bash
#SBATCH --job-name=prepare_disease
#SBATCH --cpus-per-task=8

ner-lab run prepare.yaml
```
