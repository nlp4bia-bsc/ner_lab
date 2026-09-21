# Command line

```
lab --version
lab tasks
lab run <config.yaml> [--random-state N] [--output-dir PATH]
```

There is one command that does work. Every parameter lives in the YAML file; the two flags
exist only so a job array can vary them without writing near-identical config files.

| Flag | Meaning |
|---|---|
| `--random-state N`, `--seed N` | Override the config's `random_state`. |
| `--output-dir PATH` | Override the config's `output_dir`. |

`--random-state` reaches whichever task is named, so it fails on `ner.predict_entities`, which
has no seed to set — inference is deterministic.

## The config

A flat mapping. A `task` key names the stage; every other key is passed to that stage's
function as a keyword argument, so **the YAML keys are exactly the parameter names** the guide
pages document. The CLI knows nothing about any task's keys.

Task names are namespaced by subpackage: `core.` for the torch-free contracts, `ner.` for
clinical NER, `nel.` for entity linking.

| `task` | Function | Parameters |
|---|---|---|
| `core.prepare_dataset` | `lab.core.prepare_dataset` | [prepare-dataset.md](core/prepare-dataset.md) |
| `ner.train_model` | `lab.ner.train_model` | [train-model.md](ner/train-model.md) |
| `ner.search_hyperparameters` | `lab.ner.search_hyperparameters` | [search-hyperparameters.md](ner/search-hyperparameters.md) |
| `ner.predict_entities` | `lab.ner.predict_entities` | [predict-entities.md](ner/predict-entities.md) |
| `nel.link_entities` | `lab.nel.link_entities` | [link-entities.md](nel/link-entities.md) |

`lab tasks` prints the same list, and marks any task whose extra is not installed in the
current environment. Running such a task fails with the `pip install lab[<extra>]` that fixes
it.

The [pieces underneath](ner/library.md) have no task: they take DataFrames and objects rather than
paths, so there is no single artifact for a config file to point at.

```bash
lab run prepare.yaml
lab run prepare.yaml --seed 7 --output-dir assets/splits_seed7
```

Exit codes: `0` on success, `1` when called with no command, `2` on a bad config — unknown
task or namespace, missing `task` key, missing file, missing extra, or an unknown parameter
name.

## Running on a cluster

The library ships nothing SLURM-specific. Write your own submit script around the CLI or the
Python API:

```bash
#!/bin/bash
#SBATCH --job-name=prepare_disease
#SBATCH --cpus-per-task=8

lab run prepare.yaml
```
