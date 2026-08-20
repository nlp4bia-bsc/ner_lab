# Documentation

Two audiences, two folders.

## [`guide/`](guide/) — using the library

| Page | Covers |
|---|---|
| [prepare-dataset.md](guide/prepare-dataset.md) | `prepare_dataset`: corpus conversion, disagreeing sources, label normalization, entity-stratified splitting |
| [train-model.md](guide/train-model.md) | `train_model`: scoring one configuration against a split, single or k-fold |
| [search-hyperparameters.md](guide/search-hyperparameters.md) | `search_hyperparameters`: the Ray Tune + Optuna sweep and its search space |
| [predict-entities.md](guide/predict-entities.md) | `predict_entities`: inference with a saved model, and scoring it |
| [library.md](guide/library.md) | The pieces the orchestrators are built from: `Encoder`, `build_model`, `training_arguments`, `train`, the metrics builders |
| [cli.md](guide/cli.md) | `ner-lab run`, the YAML config, exit codes, cluster submission |

## [`dev/`](dev/) — building the library

Working documents for the migration of `bsc/NER-API` into `ner_lab`. Four files, four jobs.

| File | Answers | Changes |
|---|---|---|
| [DESIGN.md](dev/DESIGN.md) | Why the library is shaped this way. The governing principle, the hard rules, the layering, how we work. | Rarely. |
| [DECISIONS.md](dev/DECISIONS.md) | What was decided and why. Numbered `D1`–`Dn`, superseded entries struck through in place. | Append-only. |
| [ROADMAP.md](dev/ROADMAP.md) | What is done, what is next, what is still open, what we refuse to port by accident. | Every stage. |
| [PROGRESS.md](dev/PROGRESS.md) | What actually landed and the evidence for it. One entry per subsystem. | Every stage. |

[CLUSTER_TESTING_NOTES.md](dev/CLUSTER_TESTING_NOTES.md) holds the notes from running against
the cluster.

Rules of thumb when updating:

- A decision goes in **DECISIONS.md** and nowhere else. Other files link to it by number.
- Superseding a decision means striking the old row and pointing at the new one, never
  deleting it — the reasoning that was wrong is worth keeping.
- **PROGRESS.md** records outcomes and evidence, not narrative. If an entry is growing a
  blow-by-blow account, it belongs in the commit message.
- Anything not yet decided is an open question in **ROADMAP.md**, not an assumption in code.
- A parameter that changes name, default or meaning changes **guide/** in the same commit.
  The guide is the parameter reference; drift there is a bug.
