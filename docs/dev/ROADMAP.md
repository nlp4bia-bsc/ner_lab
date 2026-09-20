# Roadmap

What is done, what is next, what is still open.

## Migration status

Migration status. Updated as each stage lands.

| Stage | `lab` | From | Status |
|---|---|---|---|
| Corpus + split | `core/` | `00_generate_data.py`, `src/preprocessing/{load_data,label_normalization,entity_stratified_holdout_kfold}.py` | **done** — task `core.prepare_dataset` |
| Encoding | `ner/encoding/` (segmentation in `core/`) | `src/preprocessing/{dataset_loader,sentence_splitter,window_*,label_builder,row_builder}.py` | **done** — library only, no task (D-note below) |
| Architectures | `ner/models/` | `src/models/{token_classification_base,simple_ner,crf_transformer}.py` | **done** — `build_model` |
| Training | `ner/training/` | `src/training/*`, `src/models/token_level_dataset.py`, the training half of `token_classification_base.py` | **done** — `train` |
| Evaluation | `ner/evaluation/` (span table and generic scoring in `core/`) | `src/metrics/{metrics,multiclinner_eval}.py` | **done** — `build_compute_metrics`, `multiclinner` |
| Training orchestrator | `ner/training/assessment.py` | `02_train_assessment.py` | **done** — task `ner.train_model` (D48) |
| HPO | `ner/hpo/` | `01_HPO_ner.py` (13 of its functions live in the script) | **done** — task `ner.search_hyperparameters`. Not "a loop over the orchestrator" as this file once claimed: trials call `train()` directly, with the corpus windowed once per variant (D54) |
| Inference | `ner/inference.py` | `03_infer_model.py` | **done** — task `ner.predict_entities` (D61) |
| Analysis / setup scripts | — | `01b_analyze_hpo_trials.py`, `04_official_eval.py`, `build_gold_test_parquets.py`, `aux_download_baseline_model.py` | not library surface; candidates for `examples/` or dropped |
| Augmentation | — | `00b_augment_data.py` | cut (D16) |
| Restructure | `core/` + `ner/`, `cli.py` | `ner_lab` | **done** 2026-09-17 — D82–D90, [RESTRUCTURE.md](RESTRUCTURE.md) |
| NEL | `nel/` | `bsc/nlp4bia-linking` (library surface) | **done** 2026-09-17 — task `nel.link_entities` (D91–D95) |

`Encoder` is deliberately **not** a task: it produces in-memory rows with no artifact worth
persisting. Training, HPO and inference construct and call it.

Every library stage is migrated. What remains is not library surface:

```
01b_analyze_hpo_trials.py   04_official_eval.py
build_gold_test_parquets.py aux_download_baseline_model.py
```

`hpo/report.py` already answers most of what `01b` printed, and `evaluation.multiclinner
.evaluate_tsv` is `04`'s entry point, so both are candidates for `examples/` (Q10) rather
than migration.

---

## Next

- **Gold codes in `prepare_dataset`.** Extend the corpus contract so annotated codes travel
  with the entities: `core.brat` reads `N` lines, `entities_json` entries carry the code,
  `core.corpus` validates it, and `nel.link_entities` takes gold from the span table
  instead of its own readers. One commit with its own checks (D93). Blocked on Q14 and Q15.

---

## Open questions

Nothing here blocks the next stage. Each is decided when the stage that needs it lands.

- **Q10 — `examples/`.** D21 promised a runnable script instead of a `quickstart()`. Nothing
  written yet. The four unmigrated NER-API scripts are candidates for it.
- **Q12 — Gold codes in the corpus contract.** NEL's benchmark reads a per-document parquet
  whose mention list carries `code`; ours drops BRAT `N` lines, so a gold span table with
  codes cannot be built from `prepare_dataset`'s output (D93). Direction settled 2026-09-19:
  codes belong *inside* the annotations — an optional field on each `entities_json` entry,
  not a new column alongside it — and `prepare_dataset` writes them whenever the source
  carries them. What remains open is Q14 and Q15; the work is tracked under *Next*.
- **Q14 — How does a reader know a corpus parquet carries codes?** `entities_json` is an
  opaque string column, so the parquet schema is the same with or without codes. Options:
  a corpus-level flag (parquet key-value metadata, or a column in the split manifest),
  a `has_codes` column per document, or presence of the field on the entries themselves,
  with `core.corpus.validate` deciding whether "some entries have a code and some do not"
  is a valid corpus or an error. Decide with Q15.
- **Q15 — How is a code stored on an entry?** BRAT `N` lines carry a resource name and an
  id (`N1 Reference T1 SCTID:123456 term`); NEL's TSV carries a bare `code` and, for
  composite mentions, `+`-joined codes. Options: a single `code` string, `code` plus
  `code_source`, or a list. Whatever is chosen must round-trip through `core.brat` and
  `core.corpus.read_corpus`, and be what `nel.link_entities` reads as gold instead of its
  own TSV/`.ann` readers (D92, D93).
- **Q13 — NEL's research code.** `scripts/` (retrieval benchmark, triplet generation,
  cross-encoder training), `triplets/`, `retrieval/experiments.py`, `utils/profiling.py`,
  `utils/io.py` and the `rerank_candidates` TSV helper stayed in `bsc/nlp4bia-linking`
  (D91). Candidates for `examples/` or for a training task once the cross-encoder trainer
  is needed from the library; the trainer would be `training/`'s second consumer (§2).
- **Q11 — Provenance of the caller's code.** D52 dropped NER-API's `git rev-parse HEAD`,
  which recorded the commit of the repo the *script* lived in. What identifies a run's code
  when the library is an installed wheel is undecided: `lab.__version__` is one answer,
  a `run_metadata` field the caller fills is another. Decide when someone needs to reproduce
  a run, not before.

*Closed:* Q1 (scope — settled file by file, as intended), Q4 (cluster scripts — D10),
Q5 (task naming — D48), Q6 (stray CLI in `entity_stratified_holdout_kfold.py` — dropped on
migration), Q7 (Python floor — D17), Q8 (public API surface — settled per subpackage as each
`__init__` was written, and closed by `inference`'s), Q9 (README depth — a section per stage
as it gains a task, established by `train_model`).

---

## Known baggage from NER-API

Flagged so it is not ported by accident. Struck through once handled.

- ~~`dataset_loader.py` exports a class named `DataLoader`, colliding conceptually with
  `torch.utils.data.DataLoader`.~~ Now `Encoder`.
- ~~`load_data.py::BratLoader` — two confusable loaders.~~ Now module functions in
  `data/brat.py`.
- ~~`ensure_int_list` duplicated in `metrics.py` and `token_level_dataset.py`.~~ Still two
  copies, in `training/dataset.py` and `evaluation/spans.py`, but each is now a local
  coercion helper for its own column types rather than a shared utility pretending to be one.
- ~~`metrics.py` is 1118 lines with ~30 public functions.~~ Split into four modules with a
  curated `__init__`.
- ~~`03_infer_model.py` carries copy-pasted `SentenceSpan`/`TokenSpan`/`ensure_int_list`
  instead of importing them.~~ Deduplicated on migration: its private BIO walk existed only
  because the shared one carried no score or text, which D64 fixed at the source. `inference.py`
  now decodes through `evaluation.predicted_spans` like everything else.
- `docs/architecture.md` in NER-API is stale (snapshot 2026-07-20, describes three scripts
  that no longer exist). Do not port.
- `CLAUDE.md` in NER-API is 70 KB. Not package content.
