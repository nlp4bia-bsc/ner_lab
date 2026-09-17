# Restructure: from `ner_lab` to the group library

The plan for turning `ner_lab` into one subpackage of a library shared by the NLP unit, with
normalisation (NEL) and cross-lingual transfer (XLT) as sibling subpackages. Self-contained:
a reader who has never seen the code should be able to follow it. The decisions it makes are
D82–D90 in [DECISIONS.md](DECISIONS.md); this document is their rationale.

`lab` is a placeholder for the umbrella name throughout. It is open (§8).

## 1. Goal

One library for the unit's NLP tooling, such that:

- there is one repository, one import namespace, one documentation site and one CLI;
- each tool can be installed and used without the others' dependencies;
- tools compose through data, not through imports, so a tool works on artifacts produced by
  anything — another tool here, gold annotations, a system from outside the library.

"One library" and "separate environments" were the two positions on the table. They are not
in conflict, because they answer different questions. One library is a statement about the
**repository and namespace**. Separate environments is a statement about **distributions**.
The layout below gives both: one repo and namespace, several installable extras (§6).

What the alternative — separate libraries that "work together well" — would have cost: the
working-together part needs a shared data model, so a `core` package would exist anyway, and
every change to it becomes a coordinated multi-repository release. With one maintainer per
tool that is where projects rot: tool A pins `core==0.3`, tool B moved to `0.5`, nobody
reconciles. In one repository a change to `core` and its consumers is one commit, and the
verification run says immediately what it broke.

## 2. Layout and the dependency rule

```
lab/
├── core/        data contracts, I/O, provenance, task registry.  No torch.
├── ner/         everything ner_lab does today.                   torch.
├── nel/         normalisation.
├── xlt/         cross-lingual transfer.
└── cli.py       one entry point, `lab`.
```

Imports point one way:

```
core  ←  { ner, nel, xlt }
```

- `core` imports nothing from the task subpackages.
- A task subpackage imports `core` and its own modules. **It never imports another task
  subpackage.** `nel` does not know `ner` exists; it consumes a span table (§4).
- Composition across tools — run NER, then link its output — happens at the edge: in an
  `examples/` script, or in a user's own pipeline. Not in a subpackage that imports both.

The rule that decides where a module lives: **a subpackage is defined by the artifacts it
consumes and produces, not by its topic.** Something goes into `core` when a second
subpackage needs it, and not before — otherwise `core` becomes the drawer where everything
generic-looking ends up, and every tool pays its import cost.

This is why HPO and the training scaffold stay under `ner`. Both look generic; both have one
consumer. They move to a shared layer on the day XLT or NEL calls them, and on that day it
will be known exactly which parts are shared, rather than guessed now. The same rule, in the
other direction, moves sentence segmentation to `core` today: `Encoder` and `stats` already
both call it.

Enforceable in CI with an import-linter contract (`core` may not import `lab.ner`,
`lab.nel`, `lab.xlt`; no task subpackage may import another). Cheap, and it turns the rule
from a convention into a check.

## 3. Module mapping

Everything currently in `src/ner_lab/` and where it goes. Mostly a rename; the four splits are
marked.

| Today | Becomes | Note |
|---|---|---|
| `data/corpus.py`, `data/brat.py`, `data/io.py`, `data/labels.py` | `core/corpus.py`, `core/brat.py`, `core/io.py`, `core/labels.py` | The corpus contract (§4). |
| `data/split.py`, `data/assignments.py`, `data/stratification.py` | `core/split.py`, `core/assignments.py`, `core/stratification.py` | |
| `data/dataset.py` (`prepare_dataset`) | `core/dataset.py` | Gains the `stats` handling the CLI currently does for it (§5). |
| `data/stats.py` | `core/stats.py` | Needs a tokenizer (`transformers`, tokenizer only — no torch) and segmentation. |
| `encoding/segmentation.py` | `core/segmentation.py` | **Moves out of encoding**: `stats` is its second consumer. `pysbd` becomes a `core` dependency. |
| `evaluation/spans.py` | **split** | `SPAN_COLUMNS`, `SCORED_SPAN_COLUMNS`, `span_dataframe` → `core/spans.py`. `gold_spans`, `predicted_spans`, `bio_to_spans`, `align_offsets`, `expand_to_word_extent`, `strip_bio_prefix`, `entity_tags`, `softmax` → `ner/evaluation/spans.py` (they know BIO tags and token rows). |
| `evaluation/scoring.py` | **split** | `score_spans`, `flatten`, `normalize_spans`, `group_by_document`, `safe_f1`, `to_nervaluate` → `core/scoring.py`. `span_metrics` (the `id2label` convenience) → `ner/evaluation/scoring.py`. `nervaluate` becomes a `core` dependency. |
| `evaluation/metrics.py`, `evaluation/tokens.py`, `evaluation/multiclinner.py` | `ner/evaluation/…` | Token-level metrics and the MultiClinNER official scorer are NER's. |
| `encoding/{encoder,overlaps,rows,tagging,windowing}.py` | `ner/encoding/…` | IOB2, windowing, overlap policy: model-input construction. |
| `models/` | `ner/models/` | |
| `training/` | `ner/training/` | Includes `devices.py`, `tracking.py`. Shared layer only when a second consumer exists. |
| `hpo/` | `ner/hpo/` | Same. Its import of `training.assessment.encode_partition` is NER calling NER and stays. |
| `inference.py` | `ner/inference.py` | |
| `provenance.py` | `core/provenance.py` | |
| `tasks.py` | `core/tasks.py` + one `tasks.py` per subpackage | §5. |
| `cli.py` | `lab/cli.py` | §5. Loses its `prepare_dataset` branch. |
| `__init__.py`, `__init__.pyi`, `py.typed` | `lab/ner/__init__.py` (+ stub), `lab/py.typed` | D65's ten lazy exports become `lab.ner`'s; `lab/__init__.py` exports nothing but `__version__`. |

Dependencies after the split. `core` needs `numpy`, `pandas`, `pyarrow`, `pyyaml`, `pysbd`,
`nervaluate`, and `transformers` for tokenizers — `transformers` does not depend on
`torch`, and D73 already verified that the torch-free half stays torch-free with it imported
under `TYPE_CHECKING`; `stats.py` imports `AutoTokenizer` at module level and is the one
place `core` loads it at runtime.

## 4. The contracts

Two tables. Every subpackage reads and writes them; nothing else crosses a subpackage
boundary.

**Corpus** — `core.corpus`, one row per document, parquet on disk.

```
doc_id | text | entities_json | n_entities
```

`entities_json` holds the source annotations verbatim, overlaps intact (D23). Produced by
`core.prepare_dataset` from BRAT; consumed by `ner` (encoding), `nel` (context for a
mention), `xlt` (the corpora being compared).

**Spans** — `core.spans`, one row per mention, TSV on disk.

```
filename | label | start_span | end_span | text [| score]
```

Produced by `ner.predict_entities` (with `score`), by `core.brat` from gold `.ann` files, or
by anything else that emits the columns. Consumed by `core.scoring` and by `nel`.

The rule for a subpackage that adds information: **append columns, never invent a table.**
NEL takes a span table and returns the same rows with `code` and `code_score` added. The
span columns are preserved, so the result is still a span table — it scores with
`core.scoring`, it round-trips through the same TSV reader, and a downstream tool that does
not care about codes ignores two columns.

Two levels of interaction, same schema at both:

*In process*, as DataFrames. The entry points accept a DataFrame or a path, as
`predict_entities` does today.

```python
from lab.core import read_corpus
from lab.ner import predict_entities
from lab.nel import link_entities

corpus = read_corpus("data/cantemist/corpus.parquet")
ner = predict_entities("runs/ner/best", corpus, output_dir="runs/ner/pred")
nel = link_entities(ner.spans, corpus, ontology="snomed-es", output_dir="runs/nel")
```

*On disk*, as files. This is the level that matters on the cluster, where each stage is its
own SLURM job (D22): `runs/ner/pred/predictions.tsv` is the `spans:` input of the NEL
config. Each stage's manifest records the `sha256` of what it consumed (D15, D52), so
provenance chains through the files: NEL's manifest names the predictions file, whose
manifest names the model, whose manifest names the data. The chain that exists today,
extended one link per tool.

Because `nel` depends on the schema and not on `lab.ner`, it links spans from gold
annotations, from a model outside this library, or from a colleague's system, unchanged.
That is what makes it a unit tool rather than a NER add-on.

## 5. Tasks and the CLI

The CLI keeps its shape (D11–D13): one command, one YAML per run, the `task:` key names the
operation, four flags. Two changes.

**Task names are namespaced.**

```yaml
task: ner.train_model        # was: train_model
task: core.prepare_dataset   # was: prepare_dataset
task: nel.link_entities
```

Each subpackage owns its task table. It is strings only, so importing it never loads torch:

```python
# lab/ner/tasks.py
TASKS = {
    "train_model":            ("lab.ner.training.assessment", "train_model"),
    "search_hyperparameters": ("lab.ner.hpo.search",          "search_hyperparameters"),
    "predict_entities":       ("lab.ner.inference",           "predict_entities"),
}
```

`core.tasks.resolve_task` splits the name on the first dot, imports the namespace's table,
then imports the target — and turns a missing extra into a message that names the fix:

```python
def resolve_task(name: str) -> Callable[..., Any]:
    namespace, _, task = name.partition(".")
    table = importlib.import_module(f"lab.{namespace}.tasks").TASKS
    module_name, function_name = table[task]
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"{name} needs the '{namespace}' extra: pip install lab[{namespace}]") from error
    return getattr(module, function_name)
```

`lab tasks` lists every task across namespaces and marks the ones whose extra is not
installed. That, plus the message above, answers the one support question a multi-extra
library otherwise generates.

**`cli.run` becomes fully generic.** Today it special-cases `prepare_dataset` to pop
`stats`, `base_model`, `language`, `normalize_labels` and call the stats functions on the
result (D81). Once the CLI dispatches for several subpackages it cannot know any one task's
keys; the task is the only place that knows its own. So `prepare_dataset` accepts
`stats="none" | "text" | "both"` (with `base_model`, `language`) itself and calls the stats
functions on its own result, and `run` is `task(**parameters)` and nothing else. `dataset.py`
importing `stats.py` is the cost; D81's alternative — the CLI as the one place allowed to
combine stages — does not survive a CLI that serves three subpackages.

Not doing: `lab ner train …`-style subcommands. They would say what the `task:` key already
says, and every task would need its own argparse surface — the flag explosion D11 closed.
The YAML is the argument surface.

## 6. Distributions

```toml
[project]
name = "lab"
dependencies = ["numpy", "pandas", "pyarrow", "pyyaml", "pysbd", "nervaluate", "transformers"]

[project.optional-dependencies]
torch = ["torch==2.10.0"]
ner   = ["lab[torch]", "datasets", "accelerate", "ray[tune]", "optuna", "pytorch-crf", "codecarbon", "nvidia-ml-py"]
nel   = ["lab[torch]", ...]
xlt   = ["lab[torch]", ...]
all   = ["lab[ner,nel,xlt]"]

[project.scripts]
lab = "lab.cli:main"
```

- `pip install lab` gives corpus conversion, splitting, stats and span scoring with no
  torch. Useful on its own, including to people outside the unit.
- `pip install lab[nel]` gives the linker without CRF, Ray or pysbd's NER-side uses.
- `lab[torch]` is one place for the exact pin (D7), referenced by every tool, so the unit
  stays on one `torch` and a tool cannot drift without editing the shared extra. Extras and
  subpackages are separate axes: sharing a pin does not require sharing code.

Self-referential extras (`lab[torch]` inside `lab`'s own extras) are supported by pip since
21.2 and by uv.

This supersedes D5, whose measurement — extras save nothing for HPO users — is still true
for HPO users. It stops being the whole picture once a tool exists that needs none of ray,
CRF or pysbd.

## 7. Sequencing

Two changes, done in order, each verified before the next starts. **Step 1 landed on
2026-09-17**: 686 checks passing before and after the move, plus the 68 layering checks
added with it (`verification/verify_layering.py`, D90). No compatibility shim (D89).

1. **Restructure.** `ner_lab` → `lab.core` + `lab.ner`; namespaced tasks; extras; the four
   splits in §3; `cli.run` made generic. Verified by `verification/run_all.py` and
   `examples/full_pipeline.py` passing unchanged in behaviour, and by `docs/guide/` updated
   in the same change (the guide is the parameter reference; drift there is a bug).
2. **Bring NEL in.** Adapt the existing NEL code to consume `core.spans` / `core.corpus`,
   register `nel.*` tasks, write its extra, add its verification scripts.

The two fail differently. A rename is verified by the checks that already exist; an
integration has no checks yet and involves judgment about someone else's conventions. Doing
both at once means a failure has two possible causes, and the commit that lands is
"everything changed". One commit boundary each keeps the failure attributable, and matches
how this project migrates — file by file, equivalence before improvement.

The NEL code is **read** during step 1, not merged. It answers questions the restructure has
to guess at otherwise: what shape of span it wants, whether it needs document context,
whether it has its own sentence splitter (a third consumer for `core.segmentation`), and
what it pins for `torch`/`transformers`. Read it before drawing the `core` boundary, not
after.

Migration is by module, in the dependency order of §2: `core` first (so `ner` has something
to import), then `ner`, then the CLI and registry, then docs and examples.

## 8. Open questions

- **The umbrella name.** `lab` is a placeholder, now also the directory `src/lab`, the
  distribution name, the CLI command and every import in `docs/guide/`. Renaming is a
  find-and-replace; cheapest before NEL exists. Supervisor's call.
- **What XLT consumes.** Assumed: corpora (`core.corpus`) and model identifiers, producing a
  report table and a manifest. To be confirmed before its subpackage is drawn.

Resolved by reading `bsc/nlp4bia-linking` during step 1 (D88):

- **NEL does not need the training stack.** It is retrieval (bi-encoder + FAISS, lexical
  matching) plus cross-encoder reranking and rank fusion; the cross-encoder is trained by a
  standalone script with its own torch loop, not the HF `Trainer`. `training/` keeps one
  consumer and stays under `ner`. §6's extras are right as written.
- **Its span shape is the span table.** `MentionAnnotation` is `filename, label, start_span,
  end_span, text, code` plus three optional flags — `core.spans` with `code` appended, which
  is §4's rule exactly.
- **It does not use document context today.** A text-directory loader exists but nothing
  calls it; `need_context` is a flag read from TSV. Consuming `core.corpus` is optional for it.
- **It has no sentence splitter.** `core.segmentation` keeps two consumers.
- **Pins to reconcile in step 2:** it declares `torch>=2.1,<3` and `transformers>=4.40,<5`,
  and hard-depends on `faiss`, `networkx` and `scikit-learn`. This repo's lock is on
  `transformers` 5.x. Which side moves is decided when `lab[nel]` is written.
- **`DESIGN.md`'s "Layering" section** now states §2–§3 in short and points here.
