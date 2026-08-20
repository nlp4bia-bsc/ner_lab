# Prepare a dataset

Converts a source corpus to the canonical parquet, splits it, and records provenance.

```python
from ner_lab import prepare_dataset

prepare_dataset(
    output_dir,
    documents=None,
    annotations=None,
    source_parquet=None,
    dataset_name=None,
    normalize_labels=False,
    on_mismatch="error",
    on_conflict="raise",
    split=True,
    validation_size=0.2,
    kfolds=None,
    holdout_fold=0,
    random_state=20260701,
    n_restarts=64,
    reuse_assignments=True,
    corpus_filename="documents.parquet",
    compression="zstd",
) -> PreparedDataset
```

| Parameter | Default | Meaning |
|---|---|---|
| `output_dir` | *required* | Root directory. Output goes to `<output_dir>/<dataset_name>/<split_descriptor>/`. |
| `documents` | `None` | Directory of `.txt` files, or a `{name: text}` mapping. Requires `annotations`. |
| `annotations` | `None` | A DataFrame, a `.tsv` file, a single `.ann` file, or a directory of `.ann` files. Only valid with `documents`. |
| `source_parquet` | `None` | An existing canonical parquet, skipping conversion. Mutually exclusive with `documents`. |
| `dataset_name` | derived | Overrides the derived name (from a sibling `metadata.json`, else the documents directory, else the parquet stem). |
| `normalize_labels` | `False` | Labels are stored exactly as the source annotations write them. Set `True` to map them through `ner_lab.data.LABEL_ALIASES` onto `DISEASE`/`PROCEDURE`/`SYMPTOM`/`MEDICATION`, ignoring case and accents. |
| `on_mismatch` | `"error"` | What to do when the document set and the annotation set name different files. See below. |
| `on_conflict` | `"raise"` | What to do when an annotation's text disagrees with the document it points into. See below. |
| `split` | `True` | Set `False` to write only the canonical corpus. |
| `validation_size` | `0.2` | Validation fraction for a plain train/validation split. Ignored when `kfolds` is set. |
| `kfolds` | `None` | Number of fixed-holdout cross-validation folds. Minimum 2. |
| `holdout_fold` | `0` | Which fold is the permanently reserved test partition. Only valid with `kfolds`. |
| `random_state` | `20260701` | Seed for stratification. |
| `n_restarts` | `64` | Randomized stratification attempts to keep the best of. |
| `reuse_assignments` | `True` | Reuse an existing split if present, after validating it against the corpus. Set `False` to rebuild. |
| `corpus_filename` | `documents.parquet` | Filename for the converted canonical parquet. |
| `compression` | `zstd` | Parquet compression codec. |

Give either `documents` plus `annotations`, or `source_parquet`. Passing both, or neither,
raises.

**Returns** a `PreparedDataset` with `corpus`, `corpus_path`, `dataset_root` and
`source_manifest`, plus `split`, `split_dir` and `data_manifest` when `split=True`.

## What it writes

```
<output_dir>/<dataset_name>/
    documents.parquet             canonical corpus
    source_manifest.json          source paths, sha256, label inventory, upstream metadata
    <split_descriptor>/
        split_assignments.parquet  source of truth for the split
        split_balance_report.parquet / .tsv
        data_manifest.json         split mode, seed, partition summary
        train.parquet, validation.parquet
        or: test.parquet, fold_01.parquet, ...
```

`<split_descriptor>` is `train_val_80_20` for a plain split, `kfold_5_holdout_0` for
cross-validation. With `kfolds`, `test.parquet` is the fixed holdout and is never touched
during model selection.

An existing split is reused rather than regenerated, so reruns stay reproducible even if the
seed changes. Every document is fingerprinted and checked against the manifest first — the
run fails if any document was added, removed or edited since the split was made.

## Disagreeing sources

Two things can be wrong with a source corpus before any offset is even checked, and each has
its own policy because each has a different correct answer.

`on_mismatch` covers the two sets naming different files:

| Value | Behaviour |
|---|---|
| `"error"` | Raise when an annotated filename has no document. The default: a missing document is usually a broken export, not an intentional subset. |
| `"documents"` | The document set wins. Annotations naming no document are dropped. |
| `"annotations"` | The annotation set wins. Annotations naming no document are still dropped, and documents carrying no annotation at all are dropped too. |

An annotated filename with no document is dropped under both non-raising policies — there is
no text for its offsets to refer to.

`on_conflict` covers an annotation whose surface form is not what its own offsets select:

| Value | Behaviour |
|---|---|
| `"raise"` | Raise. The default, for the same reason. |
| `"rewrite"` | The document is authoritative: the annotated text is replaced with `text[start:end]`, offsets kept as annotated. |
| `"drop"` | The document is removed from the corpus, along with all of its annotations. |

The document is never edited to match an annotation, so a rewritten corpus always trains on
what its documents actually say. A dropped document takes its clean annotations with it:
keeping them would leave a document that looks fully annotated while a real entity is silently
missing. `"rewrite"` and `"drop"` both warn once, naming the affected documents.
`ner_lab.data.resolve_mismatch` and
`ner_lab.data.resolve_conflicts` are the same resolvers, callable directly, and each returns
a report of what it dropped or rewrote.

## Label normalization

Off by default: a corpus keeps the labels its annotations declare. With
`normalize_labels=True`, labels are canonicalized (accents stripped, uppercased, `-` and
spaces to `_`) and then looked up in `ner_lab.data.LABEL_ALIASES`, which folds the Spanish and
plural spellings onto `DISEASE`/`PROCEDURE`/`SYMPTOM`/`MEDICATION`. Labels absent from the
table are canonicalized but never renamed. For a corpus with its own vocabulary, normalize the
annotations yourself and pass the result to `build_corpus`:

```python
from ner_lab.data import build_corpus, read_annotations, normalize_annotation_labels

annotations = normalize_annotation_labels(
    read_annotations("corpora/genes/ann"),
    aliases={"GEN": "GENE", "GENES": "GENE"},
)
corpus = build_corpus("corpora/genes/txt", annotations)
```

**Which labels a corpus holds** is recorded in `source_manifest.json` as
`n_entities_by_label`, alongside `n_documents` and `n_entities`:

```json
"n_documents": 1200,
"n_entities": 15064,
"n_entities_by_label": {"DISEASE": 12043, "MEDICATION": 3021}
```

`ner_lab.data.count_labels(corpus)` returns the same mapping for any corpus DataFrame, which
is the quickest way to find the `target_label` values a new corpus supports. Like the balance
report, it counts entities **as annotated** — overlaps are resolved at encoding time, so these
describe the corpus rather than the entity set a training run sees.

## From YAML

```yaml
task: prepare_dataset
documents: corpora/disease/txt
annotations: corpora/disease/ann
output_dir: assets/splits
kfolds: 5
holdout_fold: 0
```

```bash
ner-lab run prepare.yaml
```
