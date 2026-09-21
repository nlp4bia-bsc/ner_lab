# The pieces underneath

[`prepare_dataset`](prepare-dataset.md) is an assembly of three things: an annotation reader,
`build_corpus`, and `create_split`, with provenance manifests written around them. Each is
public, works on DataFrames and mappings as readily as on paths, and applies no policy the
orchestrator adds behind your back. Use them directly when the orchestrator's directory
layout, dataset naming or manifests are in the way.

None of this imports `torch`. The same module holds `score_spans` and `score_characters`,
the scorers every task's evaluation is built on, so a span table from any system can be
scored without the NER stack installed.

A complete pipeline built from them:

```python
from lab.core import build_corpus, create_split, read_corpus, score_spans, flatten

corpus = build_corpus("corpora/disease/txt", "corpora/disease/ann", normalize_labels=True)

split = create_split(corpus, "my/own/dir", ratios=[0.8, 0.2])
split.paths["train"]                       # my/own/dir/train.parquet

train = read_corpus(split.paths["train"])
```

---

## Reading annotations

```python
from lab.core import read_annotations, read_ann, read_annotation_tsv, resolve_documents

read_annotations(annotations, normalize_labels=False) -> DataFrame
read_ann(path, pattern="*.ann", normalize_labels=False) -> DataFrame
read_annotation_tsv(path, normalize_labels=False) -> DataFrame
resolve_documents(documents, pattern="*.txt", encoding="utf-8") -> dict[str, str]
```

`read_annotations` is the dispatcher: a DataFrame, a `.tsv` file, a single `.ann` file or a
directory of `.ann` files all come back as the canonical annotation frame
`filename | label | start_span | end_span | text`, offsets zero-based with an exclusive end.
The other two are what it dispatches to. A discontinuous BRAT annotation becomes one row per
fragment.

`resolve_documents` is the document-side counterpart: a directory of `.txt` files or an
existing `{name: text}` mapping, returned as the mapping `build_corpus` consumes.

---

## `build_corpus`

```python
from lab.core import build_corpus

build_corpus(
    documents,
    annotations,
    normalize_labels=False,
    on_mismatch="error",
    on_conflict="raise",
    validate=True,
) -> DataFrame
```

| Parameter | Default | Meaning |
|---|---|---|
| `documents` | *required* | Directory of `.txt` files, or a `{name: text}` mapping. |
| `annotations` | *required* | Anything `read_annotations` accepts. |
| `normalize_labels` | `False` | Map labels through `LABEL_ALIASES`. Source strings are kept verbatim otherwise. |
| `on_mismatch` | `"error"` | When the two sets name different files: `"error"`, `"documents"` (the document set wins), `"annotations"` (the annotation set wins). |
| `on_conflict` | `"raise"` | When an annotation's text is not what its offsets select: `"raise"`, `"rewrite"` (take the document text), `"drop"` (drop the document). |
| `validate` | `True` | Run `validate_corpus` on the result. Off only for a corpus you trust. |

Returns the canonical corpus, one row per document — `doc_id | text | entities_json |
n_entities`, with `entities_json` a JSON list of `{id, start, end, label, text}`. Every
entity's offsets are round-tripped against its document text on the way in, so an offset bug
surfaces here rather than as a mislabelled token later. Overlapping entities are preserved
as annotated; resolving them is a modelling choice made at encoding time.

The two policies are public functions, for when you want to know what they did:

```python
from lab.core import resolve_mismatch, resolve_conflicts

documents, annotations, mismatch = resolve_mismatch(documents, annotations, on_mismatch="documents")
mismatch.dropped_documents, mismatch.dropped_annotation_files

documents, annotations, conflict = resolve_conflicts(documents, annotations, on_conflict="rewrite")
conflict.rewritten_documents, conflict.n_rewritten_entities, conflict.dropped_documents
```

`build_corpus` calls both in that order and discards the reports. The policies themselves are
described in [prepare-dataset.md](prepare-dataset.md#disagreeing-sources).

Around the corpus frame:

| Call | Does |
|---|---|
| `validate_corpus(df)` | Check the schema, unique `doc_id`, non-empty text and every entity's offsets; return the frame reduced to the canonical columns. |
| `count_labels(df)` | `{label: count}` over the corpus as annotated. |
| `document_fingerprints(df)` | A stable sha256 per document over the canonical columns — what the split manifest records. |
| `read_corpus(path, validate=True)` | Read a corpus parquet, validated by default. |
| `write_corpus(df, path, compression="zstd")` | Write one, creating parent directories. Returns the path. |

---

## Labels

```python
from lab.core import CANONICAL_LABELS, LABEL_ALIASES, normalize_label

normalize_label("Enfermedades")          # "DISEASE"
normalize_label("Enfermedades", aliases) # your own vocabulary
```

`CANONICAL_LABELS` is `DISEASE`, `PROCEDURE`, `SYMPTOM`, `MEDICATION`; `LABEL_ALIASES` maps
the Spanish and English spellings and plurals onto them, ignoring case and accents. A label
absent from the aliases is upper-cased, stripped of accents and kept, never renamed. `normalize_entity_labels`
applies it to one document's entity list and `normalize_annotation_labels` to an annotation
frame's label column; every `normalize_labels=True` in the library routes through them, and
all three take an `aliases` mapping for a corpus with a different vocabulary.

---

## `create_split`

```python
from lab.core import create_split

create_split(
    corpus,
    output_dir,
    ratios,
    holdout_index=None,
    partition_names=None,
    random_state=20260701,
    n_restarts=64,
    reuse_assignments=True,
    compression="zstd",
) -> SplitResult
```

| Parameter | Default | Meaning |
|---|---|---|
| `corpus` | *required* | A corpus DataFrame or the path to one. |
| `output_dir` | *required* | Where the partitions, manifest and balance report are written. Your layout: nothing is nested under it. |
| `ratios` | *required* | One fraction per partition; normalised to sum to one. |
| `holdout_index` | `None` | Which partition is a permanently reserved test set. Turns the rest into rotatable folds. |
| `partition_names` | `None` | Output names. Defaults to `train`/`validation` for two ratios, or `test` and `fold_00`, `fold_01`, … with a holdout. Required for three or more ratios without one. |
| `random_state` | `20260701` | Seed for the stratification. |
| `n_restarts` | `64` | Randomised stratification attempts to keep the best of. |
| `reuse_assignments` | `True` | Reuse a manifest already in `output_dir`, after validating it against the corpus. `False` rebuilds. |
| `compression` | `"zstd"` | Parquet compression codec. |

The split is stratified by per-document entity label counts, so every partition sees each
label in proportion. **Returns** a `SplitResult` with `assignments` (the manifest),
`summary` (one row per partition), `balance_report` and `paths` (partition name to parquet).

What it writes to `output_dir`:

```
split_assignments.parquet          doc_id, fold, partition_name, document_fingerprint, plus the ratios and holdout it was made with
split_balance_report.parquet / .tsv
<partition>.parquet                one per partition name
```

The manifest is the source of truth. It is self-describing, so a later call has to be told
nothing: `create_split` on the same directory reuses it, and fails if any document was added,
removed or edited since — the fingerprints are checked against the corpus first.

Reading a split back:

| Call | Returns |
|---|---|
| `split_paths(output_dir, validation_index=None, holdout_index=None)` | Partition name to path. With a holdout, `validation_index` picks the validation fold and `train` becomes the list of every remaining fold. |
| `read_split(...)` | The same, loaded, and asserted pairwise disjoint. |
| `split_documents(corpus, assignments)` | Partition a corpus in memory against a manifest. |
| `build_balance_report(corpus, assignments)` | Per-partition document, entity and label counts against their expected values. |

Counts everywhere reflect the corpus as annotated; overlaps are resolved at encoding time,
so they are not the counts any particular training run sees.

---

## `score_spans` and `score_characters`

```python
from lab.core import score_spans, score_characters, flatten

results = score_spans(gold, predicted, tags, min_overlap_percentage=40.0)
flatten(results)                    # {"span_strict_f1": ..., "span_partial_recall": ..., ...}
score_characters(gold, predicted)   # {"char_precision": ..., "char_recall": ..., "char_f1": ..., ...}
```

| Parameter | Default | Meaning |
|---|---|---|
| `gold` | *required* | A span table: `filename | label | start_span | end_span | text`. |
| `predicted` | *required* | Another. Extra columns, such as `score`, are ignored. |
| `tags` | *required* | The labels to score. |
| `min_overlap_percentage` | `40.0` | Character overlap a predicted span needs against a gold one to count in the partial scenario. Strict and exact ignore it. |

SemEval 2013 Task 9.1 scoring via `nervaluate`, across four scenarios — `strict`, `exact`,
`partial`, `ent_type`. `score_spans` returns nervaluate's per-scenario objects; `flatten`
turns them into scalars keyed `span_strict_precision`, `span_strict_f1`,
`span_strict_correct`, `span_strict_missed` and so on, which is the form every task writes.
Offsets are converted to nervaluate's inclusive end internally, so both tables use the
library's half-open convention like everything else.

`score_characters` takes the same two tables and makes the character the unit: a character
is correct when a gold span and a predicted span of the same label both cover it, precision is
correct over predicted characters, recall correct over gold characters. There is no matching
step, so overlapping spans on either side and the order predictions arrive in cannot change
the result. Keys are `char_precision`, `char_recall`, `char_f1`, `char_correct`,
`char_missed`, `char_spurious`.

`span_dataframe(spans, scored=None)` builds a span table from a list of dicts, dropping
duplicate spans and keeping the highest-scoring copy when they carry a `score`.
`spans_from_corpus(corpus, label=None)` goes the other way — a corpus frame or parquet to
a span table, one row per entity as annotated, overlaps intact, optionally one label only;
it is how the gold for scoring comes out of a corpus.

---

## Statistics

`compute_text_stats`, `compute_annotation_stats` and `write_stats` measure a corpus frame
and are described with the orchestrator that runs them, under
[corpus statistics](prepare-dataset.md#corpus-statistics).

---

## Provenance

```python
from lab.core import write_manifest, read_manifest, file_sha256
```

`write_manifest(payload, path)` writes a JSON manifest; `read_manifest(path)` reads one back;
`file_sha256(path)` is the hash every manifest records for its inputs. They are what
`prepare_dataset` and every task orchestrator use to write `*_manifest.json`.
