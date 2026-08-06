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

Two hard rules:

1. **Library functions take explicit named parameters.** No function ever accepts
   `config: dict`. The moment one does, the library becomes the rigid black box this design
   exists to avoid.
2. **No mandatory disk writes on the pure path.** Functions return objects; persistence is a
   separate, explicit call. (One carve-out — see D13.)

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
- Written: `pyproject.toml`, `src/ner_lab/__init__.py`. Nothing migrated from NER-API yet.
