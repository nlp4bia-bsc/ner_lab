# Design

Why `lab` is shaped the way it is. Decisions are numbered in [DECISIONS.md](DECISIONS.md);
what is open is in [ROADMAP.md](ROADMAP.md).

## Design intent

Model the library on `transformers`: you import a function, pass explicit named arguments,
and get an object back. `Trainer` has no CLI; nobody trains a model by typing 25 flags.

Two user surfaces, no more:

```
python   from lab.core import prepare_dataset
         prepared = prepare_dataset(output_dir=..., documents=..., annotations=...)

yaml     lab run config.yaml
```

The YAML path is for people who don't want to touch Python. It reads a file into kwargs and
calls the same function. It is not a separate implementation, and it never reaches the
library as a config object.

**The governing principle: batteries included, but removable.** One call goes end to end, and
every step of it is a public function that can be called, replaced or skipped. Defaults are
opinionated; none are compulsory. Concretely, a high-level function may only apply policy that
is either (a) a keyword argument with the battery as its default, or (b) reachable by
composing the exported pieces yourself. Policy that can only be had by calling the whole thing
is a bug.

Three hard rules:

1. **No opaque bags.** A function never accepts `config: dict` — an untyped, undiscoverable
   grab-bag is the rigid black box this design exists to avoid. A *typed, documented,
   single-concern* arguments object is fine, and past roughly a dozen parameters it is better
   than the signature; `transformers.TrainingArguments` is the precedent (D38). The test is
   whether the object has a schema you can discover and covers one concern.
2. **No mandatory disk writes on the pure path.** Functions return objects; persistence is a
   separate, explicit call. One carve-out: training and HPO take a required `output_dir`
   because their artifacts cannot live in memory, and they still return their result (D14).
3. **No welded-in policy.** Anything that transforms or rejects the user's data by default
   carries a keyword to turn it off, and the switch defaults to the strict behaviour.

Corollaries that recur in the register: one name per concept across the library (D27, D66);
implicit behaviour is a public function the orchestrator defaults to, never buried inside it
(D26); a saved artifact describes itself (D62); stages are siblings with no shared pipeline
config and nothing forcing them to chain (D22).

---

## One library, one namespace

`lab` is the unit's NLP tooling in one repository, one import namespace, one CLI, so that a
change to the shared contracts and to every consumer is one commit that the verification
run judges at once. Separate libraries "working together" would need the same shared data
model, and every change to it would become a coordinated multi-repository release — with one
maintainer per tool, that is where projects rot (D82).

"One library" and "separate environments" answer different questions. One library is about
the repository and namespace; separate environments is about distributions, and extras give
that: `lab` alone is torch-free, `lab[torch]` is the one place the exact `torch` pin lives,
and `lab[ner]`, `lab[nel]` build on it, so the unit stays on one `torch` and a tool cannot
drift without editing the shared extra (D85). Extras and subpackages are separate axes;
sharing a pin does not require sharing code.

---

## Layering

Two axes. Within a subpackage, three tiers, so the caller picks how much is done for them:

```
tier 1  pure           objects in -> objects out. No I/O. This is the API users want.
tier 2  persistence    thin readers/writers. Explicit paths. Nothing else.
tier 3  orchestration  paths in, directory out, + manifests + provenance. What `lab run` calls.
```

Across subpackages, imports point one way:

```
lab/
├── core/    data contracts, I/O, provenance, task registry.  No torch.
├── ner/     clinical NER: encoding, models, training, HPO, inference.
├── nel/     entity linking: matching, retrieval, reranking, fusion.
└── cli.py   one entry point, `lab`.

core  <-  { ner, nel, xlt }
```

`core` imports nothing from a task subpackage, and a task subpackage never imports another
(D83). `nel` does not know `ner` exists. Composition across tools — run NER, then link its
output — happens at the edge, in an `examples/` script or the user's own pipeline, not in a
subpackage that imports both. `verification/verify_layering.py` checks the rule statically,
and that importing `core` and the CLI loads neither torch nor transformers (D90).

**A subpackage is defined by the artifacts it consumes and produces, not by its topic.**
Something moves to `core` when a second subpackage needs it, and not before — otherwise
`core` becomes the drawer where everything generic-looking ends up, and every tool pays its
import cost (D84). This is why HPO and the training scaffold stay under `ner`: both look
generic, both have one consumer. Sentence segmentation is in `core` because `Encoder` and
`stats` both call it.

### The two tables

Every subpackage reads and writes them; nothing else crosses a subpackage boundary.

**Corpus** — `core.corpus`, one row per document, parquet on disk:

```
doc_id | text | entities_json | n_entities
```

`entities_json` holds the source annotations verbatim, overlaps intact; resolving them is a
modelling choice made at encoding time (D23). Produced by `core.prepare_dataset`, consumed by
`ner` for encoding and by anything that needs document context.

**Spans** — `core.spans`, one row per mention, TSV on disk:

```
filename | label | start_span | end_span | text [| score]
```

Produced by `ner.predict_entities` (with `score`), by `core.brat` from gold `.ann` files, or
by anything else that emits the columns. Consumed by `core.scoring` and by `nel`.

The rule for a subpackage that adds information: **append columns, never invent a table.**
`nel.link_entities` takes a span table and returns the same rows with `code`, `code_term`,
`code_score`, `candidates_json` added (D92). The span columns are preserved, so the result is
still a span table — it scores with `core.scoring`, it round-trips through the same reader,
and a downstream tool that does not care about codes ignores the extra columns.

Both levels of interaction use the same schema: in process, as DataFrames, every entry point
accepting a frame or a path; on disk, as files, which is what matters on the cluster where
each stage is its own SLURM job and one stage's output file is the next stage's YAML input.
Each stage's manifest records the `sha256` of what it consumed, so provenance chains through
the files. Because `nel` depends on the schema and not on `lab.ner`, it links spans from gold
annotations, from a model outside this library, or from a colleague's system, unchanged.

### Tasks and the CLI

One command, one YAML per run, a `task:` key naming the operation, two override flags
(D11–D13). Task names are namespaced — `core.prepare_dataset`, `ner.train_model` — and each
subpackage owns a string-only `tasks.py`, so listing tasks never loads torch;
`core.tasks.resolve_task` turns a missing extra into a message naming the `pip install` that
fixes it (D86). `cli.run` is `task(**parameters)` and nothing else: the task is the only place
that knows its own keys (D87). No `lab ner train …` subcommands — they would say what the
`task:` key already says, and every task would need its own argparse surface, the flag
explosion D11 closed.

---

## Verification

There is no test suite in the package and no CI (D8). Behavioural checks live in
[`verification/`](../../verification/), outside `src/` so they are never packaged: plain
scripts, run directly, read the output. They run on synthetic fixtures — a randomly
initialised miniature BERT for anything that trains — so a machine with no corpora gets a
full green run, and real-data checks skip loudly rather than pass silently. The suite is what
runs before a stage is called done; it is what gives the rules above teeth.

Every migrated piece was diffed against its NER-API original on real data before it was
allowed to behave differently (`core`, `Encoder`, every metric family, exact), and `nel`
against `nlp4bia-linking` call for call. The one exception is `ner/inference.py`: its script
cannot run against a checkpoint both implementations can read, so it was verified through
the pieces it is built from plus direct checks on what is new in it.

---

## How we work

- **Findings are not decisions.** Noticing that the source code does something odd is not
  permission to change it. Surface it, then ask. This rule exists because it was broken once,
  during the encoding migration: ten findings were listed, one was raised, nine were
  implemented as if agreed.
- **Equivalence before improvement.** A migrated piece is diffed against its original on
  real data first; only then is it allowed to behave differently, and the difference gets a
  decision number.
- **Deliberate behaviour changes are written down**, in DECISIONS.md, with what breaks.
- **One commit boundary per kind of change.** A rename and an integration fail differently;
  landing them together makes a failure have two causes (D88).
- **No explanatory comments in library code.** Rationale goes in the register (D20).
