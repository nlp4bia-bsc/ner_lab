# Design

Why `ner_lab` is shaped the way it is. Decisions are numbered in [DECISIONS.md](DECISIONS.md).

## Design intent

Model the library on `transformers`: you import a function, pass explicit named arguments,
and get an object back. `Trainer` has no CLI; nobody trains a model by typing 25 flags.

Two user surfaces, no more:

```
python   from ner_lab.preprocessing import build_split
         df = build_split(documents=..., validation_size=0.2)

yaml     ner-lab run config.yaml
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

Three tiers, so the caller picks how much is done for them:

```
tier 1  pure           objects in -> objects out. No I/O. This is the API users want.
tier 2  persistence    thin readers/writers. Explicit paths. Nothing else.
tier 3  orchestration  paths in, directory out, + manifests + provenance. What `ner-lab run` calls.
```

Where NER-API's existing functions land:

| Existing function | Tier | Note |
|---|---|---|
| `create_entity_stratified_partition_assignments()` | 1 | already pure |
| `build_partition_balance_report()` | 1 | already pure |
| `load_document_level_parquet()` | 2 | already a reader |
| `write_document_partition()` | 2 | already a writer |
| `create_entity_stratified_split(parquet_path, output_dir, …)` | 3 | path-in/dir-out; should also accept a DataFrame |
| `build_document_level_parquet(documents, annotations, output_path)` | 3 only | **gap** — no way to get the canonical DataFrame without writing a file |
| `scripts/00_generate_data.py::main()` | 3 | manifest building and name derivation live in the script, unreachable from the library |

Work implied:

1. Split `build_document_level_parquet` into a tier-1 core returning the canonical
   `doc_id | text | entities_json | n_entities` DataFrame, plus a tier-2 writer.
2. Let `create_entity_stratified_split` take an in-memory DataFrame as well as a path.
3. Lift `00_generate_data.py`'s manifest building, `sha256` provenance, dataset-name
   derivation and split-descriptor naming into the library, opt-in per D15.

Code may be changed freely during migration — this is not a faithful port.


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
