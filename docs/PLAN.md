# ner_lab — packaging plan & decision log

Working document. Tracks what we want, what we decided, what's still open, and what's
actually been done. Update in place as we go; append to the log, don't rewrite it.

**Goal:** package the NER pipeline developed in `bsc/NER-API` as an editable, installable
Python package (`ner_lab`) that the NLP4BIA group can `pip install -e .` and use.

Source repo: `bsc/NER-API` (personal research repo, `A6-z/NER-API`)
Target repo: `bsc/ner_lab` (group repo, `nlp4bia-bsc/ner_lab`)

Migration mode: **files are copied over one at a time and modified in the process** — not a
bulk tree move. NER-API stays as the research history.

---

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

1. **Library functions take explicit named parameters.** No function ever accepts
   `config: dict`. The moment one does, the library becomes the rigid black box this design
   exists to avoid.
2. **No mandatory disk writes on the pure path.** Functions return objects; persistence is a
   separate, explicit call. (One carve-out — see D13.)
3. **No welded-in policy.** Anything that transforms the user's data by default carries a
   keyword to turn it off.

---

## Settled decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Package name / import namespace: `ner_lab`; distribution `ner-lab` | Matches repo. Flat `preprocessing`/`models`/`training`/`metrics` would squat 4 generic names in a shared venv. |
| D2 | `src/` layout: `src/ner_lab/{preprocessing,models,training,metrics}/` | Prevents accidental imports from cwd; standard for installable packages. |
| D3 | ~~Single CLI `ner-lab <subcommand>`, one subcommand per stage~~ **Superseded by D11.** | The per-stage subcommand design assumed a flag surface. It doesn't survive the "library first" intent. |
| D4 | ~~CLI is a thin wrapper over the library~~ **Restated as D11/D12.** | Still true, but the shape changed. |
| D5 | All dependencies required — no optional extras | Measured: torch+nvidia = 4.0 GB, ray = 200 MB (3.6%). Extras buy nothing on size, and the first users *are* the HPO users. |
| D6 | Drop deps `dataclasses`, `pathlib`, `typing` | stdlib since 3.7 / 3.4 / 3.5; floor is 3.10. Installed as real redundant files (`site-packages/pathlib.py` is the 2014 version, `from collections import Sequence`). Inert only because stdlib precedes site-packages in `sys.path`. |
| D7 | `torch==2.10.0` stays an exact pin | All known users need exactly 2.10, not a floor. |
| D8 | No tests in the package or the repo for now. No `pytest` dep. | User decision. Consequence accepted: no regression guard, notably on `metrics.py` (offset-sensitive) and `multiclinner_eval.py` (equivalence to the official scorer rests on a one-off diff). |
| D9 | Build backend: hatchling | uv-native, no `setup.py`, minimal config. |
| D10 | Nothing MN5- or SLURM-specific ships. No `sh/`. Pure Python. | Users write their own submit script around the library. Closes Q4. |
| D11 | **One CLI command: `ner-lab run <config.yaml>`.** No per-stage subcommands. | ~9 stages × ~25 flags = ~200 argparse declarations, each duplicating a function parameter and needing edits in two places on every signature change. |
| D12 | YAML is flat, with a `task:` key naming the operation. One file fully describes one run. | Dispatch is a dict lookup; the file is self-contained and readable on its own. |
| D13 | **~4 named flags total:** `--seed`, `--output-dir`, plus `--version`. No generic `--set key=value`. Everything else lives in the YAML. | Covers the SLURM job-array case (sweep seeds without writing N near-identical YAML files) without reopening a per-stage flag surface. |
| D14 | HPO and training may write to disk; `output_dir` stays an explicit required parameter, and the function still **returns** its result. | Ray writes per-trial dirs and Trainer writes checkpoints — a 40-trial GPU-day sweep can't live in memory. Precedent: `Trainer` requires `output_dir` and `train()` returns `TrainOutput`. What we kill is "the only way to see a result is to read a file afterward". |
| D15 | Manifests (`sha256`, `run_manifest.json`, dataset-name derivation) become opt-in behaviour of the persistence layer. | Today they are unconditional side effects inside `00_generate_data.py`. The pure path must not touch them. |
| D16 | Augmentation is cut from the first release. | `00b_augment_data.py` is broken in NER-API anyway (expects `--input-jsonl`; nothing emits jsonl since the parquet migration). Closes the `augment-data` row. |
| D17 | Python floor `>=3.10`. | Most clusters are at this version. Closes Q7. |
| D18 | Drop deps `nltk`, `pydantic`, `psutil` (zero imports). Keep `accelerate` and `optuna` despite zero direct imports. | `accelerate` is required by `Trainer`; `optuna` by `ray.tune.search.optuna.OptunaSearch`. Both would otherwise look dead and get pruned by someone later. |
| D19 | `ray` must be `ray[tune]`. | Code imports `ray.tune`, `ray.air`, `ray.tune.search.optuna` — none ship in the base wheel. Works in NER-API's venv only because something else dragged them in. Latent bug on a clean install. |
| D20 | No explanatory comments in library code. Rationale goes in this document. | Comments that narrate what the next line does read as machine-written and add maintenance surface. Comments are for the non-obvious only (e.g. why a dependency with no imports is required). |
| D21 | No `quickstart()` function. A runnable script in `examples/` instead. | By the "no functions whose only value is calling three others" rule, a quickstart is exactly that. As an example it is something to read and copy; as an API it is dead weight. |
| D22 | No shared multi-stage pipeline config, and no blocking `run_pipeline()`. | Stages have incompatible runtime shapes: seconds/CPU vs GPU-days/Ray/multi-GPU vs GPU-hours/single-GPU. Chaining them in-process either camps on a login node for days or reimplements SLURM dependencies badly. Parameter drift between stages is instead solved *optionally* — point two runs at the same YAML if you want to; nothing forces it. |
| D23 | **Entity overlap resolution stays at encoding time, not corpus build time.** The canonical parquet holds raw source annotations, overlaps intact. | Keeps the corpus artifact faithful to the source; overlap handling is a modelling choice. Lets several policies be compared without regenerating the corpus. Consequence, accepted and to be documented: `n_entities` and the split balance report count *unresolved* entities, so they describe the corpus rather than any run's training set. |
| D25 | `n_entities` is pinned to `int32`, dropping NER-API's `downcast="integer"`. | A data-dependent dtype is a footgun: a corpus fitting `int8` silently changes schema when a densely-annotated document is added. Stable schema over a few saved bytes. |
| D26 | Implicit behaviour is exposed as public functions that the orchestrator defaults to, never buried inside it. | `derive_dataset_name`, `split_descriptor`, `read_dataset_metadata` are callable and overridable. The `<output_dir>/<dataset_name>/<split_descriptor>/` layout stays in the library because the training stage reads it — it is a contract, not a convenience. |
| D27 | One name per concept across the library: `random_state`, not `seed`. CLI accepts `--random-state`/`--seed` as aliases for one destination. | The script used `--seed`, the splitter used `random_state`. Two names for one thing is worse than either. Default unified on `DEFAULT_RANDOM_STATE` (the script's 42 was an accident); a default-seeded rerun therefore won't reproduce an old script-made split, and the seed is recorded in the manifest. |
| D28 | `target_label` is validated against the labels present in the corpus, not against `CANONICAL_LABELS`. To be applied when `encoding/` is migrated. | The data layer is already entity-agnostic (verified on a `GENE/protein/cell-line/Chemical` corpus). Only `label_builder.py` and `dataset_loader.py` hard-raise on the four clinical labels, which would stop the group using this on any other corpus. `CANONICAL_LABELS` stays as an alias table for normalization, not as a whitelist. **Separate, still open:** multi-label IOB2 (`target_labels`, `2n+1` classes) — NER-API's deferred Phase 5. D28 does not depend on it. |
| D24 | Folder structure is created as needed, not defined up front. | Agreed shape: `data/` (document-granularity, persisted) vs `encoding/` (window-granularity, per-run), plus `models/`, `training/`, `evaluation/`, and top-level `cli.py`, `tasks.py`, `provenance.py`, `inference.py`. Details settle during migration. |

---

## Open questions

- **Q1 — Scope.** Which files move, exactly. Deliberately deferred: decided file by file
  during migration, not up front.
- **Q5 — Task naming.** Numeric script prefixes (`00_`, `01b_`) encoded pipeline order, lost
  in a flat `task:` vocabulary. Names to settle as each stage lands.
- **Q6 — Stray CLI in library code.** `entity_stratified_holdout_kfold.py` has its own
  `_parse_args()`/`main()`. Drop it — presumed yes under D11, confirm on migration.
- **Q8 — Public API surface.** What each subpackage's `__init__` re-exports. `metrics.py`
  alone has ~30 public functions; most are internals.

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

## Stages (draft)

Pipeline order. Confirmed against the code as each one is migrated.

| NER-API script | Input | Output |
|---|---|---|
| `00_generate_data.py` | BRAT dir + annotations, or canonical parquet | `documents.parquet`, `split_assignments.parquet`, `train`/`validation` (or `test` + `fold_XX`) `.parquet`, balance report, `source_manifest.json`, `data_manifest.json` |
| `01_HPO_ner.py` | `train.parquet` + `validation.parquet`, model id, target label | `run_manifest.json`, `trials_summary.parquet`, `best_config.json` |
| `01b_analyze_hpo_trials.py` | `trials_summary.parquet` | trial analysis |
| `02_train_assessment.py` | `best_config.json` + split dir | `run_manifest.json`, `best_config.json`, `epoch_metrics.parquet`, `best_model/`, k-fold summary |
| `03_infer_model.py` | trained model dir + input parquet or BRAT dir | `predictions.tsv` |
| `04_official_eval.py` | `predictions.tsv` + gold (TSV or canonical parquet) | `gold_eval_metrics.json`, `multiclinner_eval.json` |
| `aux_download_baseline_model.py` | HF model tag | local model dir |
| `build_gold_test_parquets.py` | gold corpus | gold parquets |
| ~~`00b_augment_data.py`~~ | — | cut for now (D16) |

---

## Known baggage inherited from NER-API

Flagged so we decide consciously rather than porting it by accident.

- `docs/architecture.md` in NER-API is **stale** (snapshot 2026-07-20; describes
  `01_optuna_ner.py`, `02_final_training.py`, `02_quick_train.py` — none exist now). Do not
  port as-is.
- `src/preprocessing/dataset_loader.py` exports a class named `DataLoader` — collides
  conceptually with `torch.utils.data.DataLoader`. Rename.
- `src/preprocessing/load_data.py::BratLoader` was itself renamed from `DataLoader`; two
  confusable loaders remain.
- `03_infer_model.py` carries **copy-pasted** `SentenceSpan`/`TokenSpan`/`ensure_int_list`
  instead of importing them. Deduplicate.
- `ensure_int_list` exists in both `metrics.py` and `models/token_level_dataset.py`.
- `metrics.py` is 1118 lines with ~30 public functions — the module most in need of a
  curated public surface (Q8).
- `CLAUDE.md` in NER-API is 70 KB. Not package content.

---

## Progress log

### 2026-08-05
- Read `bsc/NER-API` end to end. 12.3k lines across 5 flat top-level modules, **not
  installable** — no `[build-system]`, scripts reach the library via `sys.path.insert`.
- Measured dependency weight to settle the extras question (D5); verified the three backport
  deps are real redundant files on disk (D6).
- Audited imports against declared deps: found 3 dead (D18) and the `ray[tune]` gap (D19).
- Design conversation settled the library-first shape: D11–D22 replace the original
  subcommand plan (D3/D4).
- Written: `pyproject.toml`, `.gitignore`, `src/ner_lab/__init__.py`, `src/ner_lab/cli.py`.
  Verified: wheel builds with only `ner_lab/` inside, `ner-lab` entry point runs, config
  error paths exit 2, `uv lock` resolves 123 packages across `>=3.10`, and `torch==2.10.0`
  publishes cp310–cp314 wheels so D7 and D17 don't conflict.

### 2026-08-06 — `data/` migrated

First migration. NER-API `preprocessing/{label_normalization,load_data,brat_to_parquet}.py`
became `ner_lab/data/{labels,brat,corpus,io}.py`.

Changes made on the way over, beyond a plain port:

- **`BratLoader` is gone; it is module-level functions now.** The class had no `__init__`, no
  instance state and one class constant — every method was a function carrying a `self`.
- **All JSONL handling dropped** (~150 lines): `load_jsonl_records`,
  `load_documents_from_jsonl`, `load_annotations_from_jsonl`, `jsonl_to_annotations_tsv`,
  `_validate_jsonl_record`, and the JSONL branches of both dispatchers. Nothing has emitted
  jsonl since the parquet migration and the only consumer was the cut augmentation script.
- **`normalize_labels_in_record`/`_records` dropped** — used only by `00b_augment_data.py`
  (cut, D16). `strip_accents` made private; nothing outside the module used it.
- **The tier-1 gap is closed.** `build_document_level_parquet` was write-only; it is now
  `build_corpus()` (pure, returns the DataFrame) plus `write_corpus()`/`read_corpus()`.
- **Dependency inverted.** `brat_to_parquet.py` imported `DOCUMENT_COLUMNS`,
  `DEFAULT_PARQUET_COMPRESSION` and `load_document_level_parquet` *from the splitter*. Those
  are corpus-schema concerns; they now live in `data/corpus.py` and `data/io.py`, and
  `data/split.py` will depend on them rather than the reverse.
- **`load_document_level_parquet` split** into `validate_corpus(df)` (works in memory) and
  `read_corpus(path)` (read + validate). The `document_fingerprint` column it used to append
  is now an explicit `document_fingerprints(df)` call, so the reader no longer silently adds a
  column outside the canonical schema. **Open:** how `split.py` consumes it — decide when
  that module is migrated.

Verified against NER-API on the real `MultiClinNER-es-train-disease` sample (1258 documents,
26,296 entities): `build_corpus` returns a DataFrame `.equals()` the old
`build_document_level_parquet` output, from both `.ann` and `.tsv` sources. Plus parquet
round-trip, fingerprint stability/uniqueness, label aliasing, discontinuous BRAT spans, and
7 rejection paths. 19 checks, all passing.

**Caveat, per D8:** that verification was a throwaway script in a scratch directory, not a
committed test. The evidence does not survive the session — the same footgun already noted
for `multiclinner_eval.py` in NER-API.

### 2026-08-06 — splitting migrated

`entity_stratified_holdout_kfold.py` (1115 lines, one module) became three, split at the
seams rather than by size:

```
data/stratification.py   the algorithm. pure numpy, knows nothing about manifests.
data/assignments.py      the manifest: build, validate, integrity, accessors.
data/split.py            using a split: partition, report, persist, resolve paths.
```

Changes made on the way over:

- **`assignments_df` is now self-describing.** `holdout_index` and `partition_name` used to be
  bolted on by the orchestrator *after* the pure function returned, leaving its output
  incomplete. They are set inside `assign_partitions` now, so
  `build_partition_balance_report(documents_df, assignments_df, n_partitions, holdout_index,
  ratios)` collapses to `build_balance_report(documents_df, assignments_df)` — the other three
  are columns in the manifest, and passing them separately was a way to pass inconsistent
  values. Same for `assert_partition_integrity`.
- **The restart loop moved into `stratify_documents`**, so "find the best assignment vector"
  is entirely inside `stratification.py`; `assign_partitions` just wraps the result in a
  DataFrame. Seeding (`default_rng(random_state + restart_index)`) is unchanged.
- **Tier-1 gap closed for splitting**: `split_documents(documents_df, assignments_df)` returns
  the partitions as DataFrames, no directory required.
- **`write_document_partition` deleted** — it was `write_corpus` with a column selection.
- **The module's `_parse_args`/`main` dropped** (closes Q6).
- Renames: `create_entity_stratified_split` -> `create_split`,
  `create_entity_stratified_partition_assignments` -> `assign_partitions`,
  `validate_partition_assignments` -> `validate_assignments`, `get_split_paths` ->
  `split_paths`, `load_split` -> `read_split`. Manifest accessors take a `manifest_` prefix
  (`manifest_holdout_index`, not `holdout_index`) so they don't shadow the identically named
  parameter in half the functions that use them.
- `create_split` accepts an in-memory DataFrame as well as a path, and returns a `SplitResult`
  dataclass (assignments, summary, balance report, written paths) instead of a 2-tuple.

**One real bug found by the equivalence test.** Passing a DataFrame to `create_split`
bypassed the dtype normalization the path route gets from `validate_corpus`, so the *same*
corpus serialized as `int32` via a DataFrame and `int8` via a path. Fixed by making
`validate_corpus` the single normalization point: `build_corpus` ends with it, and
`create_split` applies it to DataFrame input.

**Deliberate divergence from NER-API:** `build_corpus` now returns `n_entities` downcast
(`int8` on the sample) where `build_document_level_parquet` returned `int32`. Values are
identical. This is the cost of having one normalization point. Note the downcast makes the
dtype data-dependent — a corpus with >127 entities in a document gets `int16`. **Open:** worth
dropping the downcast for a stable schema? Deferred, not silently changed.

Verified against NER-API on the real sample for both an 80/20 split and 5-fold with a fixed
holdout: assignments, fold vectors, balance reports, summaries and every partition parquet
are identical. Plus in-memory splitting, manifest reuse, tamper detection (edited/removed
document), path resolution, and 7 argument-validation paths. 38 checks; corpus suite still
green at 21.

Same D8 caveat: scratch scripts, nothing committed.

### 2026-08-06 — generate-data finished, first task wired

`scripts/00_generate_data.py` became `ner_lab/data/dataset.py` + `ner_lab/provenance.py`, and
`prepare_dataset` is the first task reachable from YAML.

- **`n_entities` pinned to `int32`** (D25). The inherited
  `pd.to_numeric(downcast="integer")` made the dtype data-dependent — a corpus that happened
  to fit `int8` would silently change schema once a densely-annotated document was added.
- **No class.** `prepare_dataset` is a function returning a `PreparedDataset` dataclass;
  nothing holds state between calls.
- **Derivation is exposed, not buried** (D26). `derive_dataset_name`, `split_descriptor` and
  `read_dataset_metadata` are public functions that `prepare_dataset` merely defaults to, so a
  Python caller can inspect, override or skip any of it. The
  `<output_dir>/<dataset_name>/<split_descriptor>/` layout stays in the library because it is
  a contract with the training stage, not cosmetics.
- **`prepare_dataset` returns the corpus and both manifests**, not just paths — the script's
  only output channel was the filesystem.
- **One name for the seed** (D27): the library uses `random_state` throughout (already the
  manifest column, sklearn convention); the CLI accepts `--random-state` and `--seed` as
  aliases for the same destination. The script's two names for one concept are gone.
- **Default `random_state` unified** on `DEFAULT_RANDOM_STATE` (20260701). The script
  defaulted to 42 while the splitter defaulted to 20260701 — an accident worth removing, but
  note a default-seeded rerun will not reproduce an old script-made split. The seed is
  recorded in the manifest either way.
- `tasks.py` maps task names to `(module, function)` and imports lazily, so `ner-lab run`
  doesn't pay for torch or ray to prepare a dataset.

Verified against `scripts/00_generate_data.py` run as a subprocess on the real sample, for
both 80/20 and 5-fold + holdout: identical directory layout, identical file lists, and
byte-identical canonical parquet, partitions, assignments and balance report. Plus the
returned object, no-split mode, parquet input with a sibling `metadata.json` (CARMEN-I),
every derivation helper, 6 input-validation paths, and the YAML route through the CLI
including `--seed`/`--output-dir` overrides and 4 CLI error paths. 63 checks; corpus 21 and
split 45 still green.

Same D8 caveat: scratch scripts, nothing committed.

### 2026-08-06 — removability audit

"Batteries included, but removable" adopted as the governing principle (see Design intent).
Audited the data layer against it and found one real violation:

- **Label normalization was welded into the BRAT reader.** `read_annotations` and everything
  above it rewrote label strings unconditionally, with no opt-out. `Anatomy` -> `ANATOMY`,
  `cell-line` -> `CELL_LINE`. Harmless for the four clinical labels this library defaults to,
  silently destructive for any corpus with a different vocabulary — which a *group* library
  will meet. Fixed: `normalize_labels: bool = True` on `read_annotations`, `read_ann`,
  `read_annotation_tsv`, `build_corpus` and `prepare_dataset`.
- **`build_corpus` forced a whole-corpus validation pass.** `read_corpus` already had
  `validate=`; the builder now matches.

Judged acceptable as-is, because they are removable by composition rather than by keyword:
`create_split` always writing a balance report (use `assign_partitions` + `split_documents`),
`prepare_dataset` always writing manifests (use `create_split`), and `split_documents`
asserting partition integrity (cheap set operations on doc_ids, and a footgun to skip).

Verified the battery comes out at every level and that aliasing still works when it is left
in. All three suites still green: 21 / 45 / 63.

### 2026-08-06 — data generation closed out

Feature parity with `scripts/00_generate_data.py` confirmed flag by flag; all 12 CLI flags map
onto a `prepare_dataset` parameter.

Fixed one provenance gap introduced by the removability work: `normalize_labels` changes the
stored data but was not recorded anywhere, so a verbatim-label corpus was indistinguishable
from a normalized one. It is now in `source_manifest.json` (and `None` for parquet input,
where no conversion happened).

**Data generation is complete.** Remaining items are decisions, not missing work:

- `scripts/build_gold_test_parquets.py` is **not** being migrated as library surface — it is
  MultiClinNER-specific glue (fixed `<entity>/tsv|txt` directory convention, a doc_id mapping
  file, text reconciliation). It becomes a user script calling `prepare_dataset`. Its
  `reconcile_annotation_text` is the only part that might deserve generalizing later.
- No tests (D8). The equivalence evidence for all three suites is disposable.
- `language` is captured into `source_manifest.upstream_metadata` but not surfaced as a
  parameter. `encoding/` needs it for pysbd sentence segmentation — wire it there.
- `CANONICAL_LABELS` is exported but unused until `encoding/` (see D28).

**Next:** `encoding/` (the tokenizer-facing half of NER-API's `preprocessing/`), then
`models/`, `training/`, `evaluation/`.
