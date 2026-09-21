# lab

The unit's NLP library: one repository, one import namespace, one CLI. `lab.core` holds the
data contracts with no `torch`; each task lives in its own subpackage on top of it —
`lab.ner` for clinical NER, `lab.nel` for linking mentions to an ontology.

Every step is a public function you call with explicit named arguments. Nothing takes a
config dict, and nothing writes to disk unless you asked it to. A thin CLI runs the same
functions from a YAML file.

## Install

```bash
git clone git@github.com:nlp4bia-bsc/ner_lab.git
cd ner_lab
uv pip install -e ".[ner]"     # or: pip install -e ".[ner]"; ".[nel]" for linking, ".[all]" for both
```

Python 3.10 or newer. `lab` alone is torch-free; `lab[ner]` and `lab[nel]` both pin
`torch==2.10.0` through the shared `lab[torch]` extra.

## One example

```python
from lab.core import prepare_dataset

prepared = prepare_dataset(
    output_dir="assets/splits",
    documents="corpora/disease/txt",
    annotations="corpora/disease/ann",
    validation_size=0.2,
)

prepared.split.paths["train"]   # Path to the written train.parquet
```

```bash
lab run prepare.yaml            # task: core.prepare_dataset, the same keys as above
```

Both routes call the same function and produce identical output.

## Documentation

| | | |
|---|---|---|
| [`docs/core/`](docs/core/) | `lab.core` | the two tables, `prepare_dataset`, splitting, scoring |
| [`docs/ner/`](docs/ner/) | `lab.ner` | `train_model`, `search_hyperparameters`, `predict_entities`, and the pieces underneath |
| [`docs/nel/`](docs/nel/) | `lab.nel` | `link_entities`, the candidate methods, reranking |
| [`docs/cli.md`](docs/cli.md) | `lab` | `lab run`, `lab tasks`, the YAML config |
| [`docs/dev/`](docs/dev/) | | how the library is built: design, decisions, roadmap, progress |

Each subpackage index names its maintainer. Behavioural checks live in
[`verification/`](verification/), outside `src/`; see its README for the environment.
