# `lab.core` — the data contracts

Maintainer: <author> · <email>

Everything the task subpackages share, with no `torch`: corpus conversion, entity-stratified
splitting, corpus statistics, span scoring, provenance. `pip install lab` alone gives you
this.

| Page | Covers |
|---|---|
| [prepare-dataset.md](prepare-dataset.md) | `prepare_dataset`: corpus conversion, disagreeing sources, label normalization, entity-stratified splitting, corpus statistics |
| [library.md](library.md) | The pieces it is built from: the annotation readers, `build_corpus`, `create_split`, `score_spans`, the manifests |

| CLI task | Function |
|---|---|
| `core.prepare_dataset` | `lab.core.prepare_dataset` |

## The two tables

Every stage after conversion speaks one schema, one row per document:
`doc_id | text | entities_json | n_entities`, where `entities_json` is a JSON list of
`{id, start, end, label, text}`. Offsets are zero-based with an exclusive end, and
`text[start:end] == entity["text"]` is validated on the way in.

Entities are stored **as annotated** — overlaps are preserved, and resolving them is a
modelling choice made at encoding time, so several policies can be compared without
regenerating the corpus. One consequence: `n_entities` and the balance report describe the
corpus, not the entity set any particular training run sees.

The second table is the span table, one row per mention:
`filename | label | start_span | end_span | text [| score]`. Inference writes it, gold `.ann`
files read into it, `score_spans` scores any two of them against each other, and
`lab.nel.link_entities` takes one in and hands it back with `code` columns appended.

Task subpackages compose through these two tables and never import each other, so a linker
or a cross-lingual comparison works on spans from this NER, from gold annotations, or from a
system outside the library.

## Importing

```python
from lab.core import prepare_dataset, read_corpus, build_corpus, create_split, score_spans
```

`lab.core` exports its contracts and every function around them directly; there are no
lazy names and no submodule paths to remember.
