# Design

Why `lab` is shaped the way it is. Decisions are numbered in [DECISIONS.md](DECISIONS.md).

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
is a bug — see the 2026-08-06 audit for one that got through.

Three hard rules:

1. **No opaque bags.** A function never accepts `config: dict` — an untyped, undiscoverable
   grab-bag is the rigid black box this design exists to avoid. A *typed, documented,
   single-concern* arguments object is fine, and past roughly a dozen parameters it is better
   than the signature; `transformers.TrainingArguments` is the precedent. The test is whether
   the object has a schema you can discover and covers one concern. NER-API's
   `NERTrainingConfig` fails the second half — see D29.
2. **No mandatory disk writes on the pure path.** Functions return objects; persistence is a
   separate, explicit call. (One carve-out — see D13.)
3. **No welded-in policy.** Anything that transforms the user's data by default carries a
   keyword to turn it off.

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
└── cli.py   one entry point, `lab`.

core  <-  { ner, nel, xlt }
```

`core` imports nothing from a task subpackage, and a task subpackage never imports another.
Tools compose through `core`'s two tables — the corpus (`doc_id | text | entities_json |
n_entities`) and the span table (`filename | label | start_span | end_span | text [| score]`)
— not through imports; a subpackage that adds information appends columns. A module moves to
`core` when a second subpackage needs it, and not before. `verification/verify_layering.py`
checks the rule. [RESTRUCTURE.md](RESTRUCTURE.md) §2–§4 is the full argument.

---

## How we work

- **Findings are not decisions.** Noticing that NER-API does something odd is not permission
  to change it. Surface it, then ask. This rule exists because it was broken once: ten
  encoding "findings" were listed, one was raised, and the other nine were implemented as if
  agreed.
- **Migration is file by file, with modifications.** Not a bulk tree move. NER-API stays as
  the research history.
- **Equivalence before improvement.** A migrated piece is diffed against its NER-API
  original on real data first; only then is it allowed to behave differently, and the
  difference gets a decision number.
- **Deliberate behaviour changes are written down**, in DECISIONS.md, with what breaks.
