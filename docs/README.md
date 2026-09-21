# Documentation

Two audiences: the subpackage folders are for using the library, `dev/` is for building it.

## Using the library

One folder per subpackage. Each index names its maintainer, its CLI tasks and its pages.

| Folder | Subpackage | Pages |
|---|---|---|
| [`core/`](core/) | `lab.core` — the data contracts, no `torch` | [prepare-dataset](core/prepare-dataset.md), [library](core/library.md) |
| [`ner/`](ner/) | `lab.ner` — clinical NER | [train-model](ner/train-model.md), [search-hyperparameters](ner/search-hyperparameters.md), [predict-entities](ner/predict-entities.md), [library](ner/library.md) |
| [`nel/`](nel/) | `lab.nel` — entity linking | [link-entities](nel/link-entities.md) |

[cli.md](cli.md) is the one page that cuts across them: `lab run` and `lab tasks`, the YAML
config, namespaced task names, exit codes, cluster submission.

## [`dev/`](dev/) — building the library

| File | Answers | Changes |
|---|---|---|
| [DESIGN.md](dev/DESIGN.md) | Why the library is shaped this way: the governing principle, the hard rules, the layering and the two tables, verification, how we work. | Rarely. |
| [DECISIONS.md](dev/DECISIONS.md) | What was decided, one line each, numbered `D1`–`Dn`; the argument under the same number for the ones that need it. | Append-only. |
| [ROADMAP.md](dev/ROADMAP.md) | Where things stand, what is next, what is still open. | Every stage. |

Rules of thumb when updating:

- A decision goes in **DECISIONS.md** and nowhere else. Other files link to it by number.
- A decision row is one line: the verdict, and the why in a clause. Reasoning longer than
  two sentences goes in the notes section of the same file under the same number, and the
  row points there (`→ notes`).
- Superseding a decision means collapsing the old row to `Dn — superseded by Dm`, never
  renumbering — citations elsewhere must keep resolving.
- Anything not yet decided is an open question in **ROADMAP.md**, not an assumption in code.
- A parameter that changes name, default or meaning changes its subpackage's page in the
  same commit. Those pages are the parameter reference; drift there is a bug.
