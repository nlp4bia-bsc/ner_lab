# Roadmap

What is next, and what is still open. Decisions are in [DECISIONS.md](DECISIONS.md).

## Status

Every library stage of `bsc/NER-API` is migrated: `core.prepare_dataset`, the `Encoder`,
`build_model`, `train`, `ner.train_model`, `ner.search_hyperparameters`,
`ner.predict_entities`. The restructure into `lab.core` + `lab.ner` (D82–D90) and the
integration of `bsc/nlp4bia-linking` as `lab.nel` (D91–D95) both landed on 2026-09-17.
Verification stands at 878 checks across eleven scripts, all passing; see
[`verification/`](../../verification/).

What remains of NER-API is not library surface — `01b_analyze_hpo_trials.py`,
`04_official_eval.py`, `build_gold_test_parquets.py`, `aux_download_baseline_model.py` —
and is a candidate for `examples/` (Q10) rather than migration. `hpo/report.py` already
answers most of what `01b` printed; `04`'s job is `predict_entities` with a `reference`.

`lab.xlt` (cross-lingual transfer) is a planned subpackage; nothing is drawn until what it
consumes is confirmed (Q16).

## Next

- **Gold codes in `prepare_dataset`.** Extend the corpus contract so annotated codes travel
  with the entities: `core.brat` reads `N` lines, `entities_json` entries carry the code,
  `core.corpus` validates it, and `nel.link_entities` takes gold from the span table
  instead of its own readers. One commit with its own checks (D93). Blocked on Q14 and Q15.

## Open questions

Nothing here blocks the next stage. Each is decided when the stage that needs it lands.

- **Q10 — `examples/`.** D21 promised runnable scripts instead of a `quickstart()`.
  `examples/full_pipeline.py` exists; the four unmigrated NER-API scripts and NEL's research
  code (Q13) are candidates for more.
- **Q11 — Provenance of the caller's code.** D52 dropped `git rev-parse HEAD`, which
  recorded the commit of the repo the *script* lived in. What identifies a run's code when
  the library is an installed wheel is undecided: `lab.__version__` is one answer, a
  `run_metadata` field the caller fills is another. Decide when someone needs to reproduce a
  run, not before.
- **Q12 — Gold codes in the corpus contract.** Direction settled 2026-09-19: codes belong
  *inside* the annotations — an optional field on each `entities_json` entry, not a new
  column alongside it — and `prepare_dataset` writes them whenever the source carries them.
  What remains is Q14 and Q15; the work is tracked under *Next*.
- **Q13 — NEL's research code.** `scripts/` (retrieval benchmark, triplet generation,
  cross-encoder training), `triplets/`, `retrieval/experiments.py`, the profiler and the
  `rerank_candidates` TSV helper stayed in `bsc/nlp4bia-linking` (D91). Candidates for
  `examples/`, or for a training task once the cross-encoder trainer is needed from the
  library; that trainer would be `training/`'s second consumer and the first reason to move
  any of it to `core`.
- **Q14 — How does a reader know a corpus parquet carries codes?** `entities_json` is an
  opaque string column, so the parquet schema is the same with or without codes. Options: a
  corpus-level flag (parquet key-value metadata, or a column in the split manifest), a
  `has_codes` column per document, or presence of the field on the entries themselves, with
  `validate_corpus` deciding whether "some entries have a code and some do not" is a valid
  corpus or an error. Decide with Q15.
- **Q15 — How is a code stored on an entry?** BRAT `N` lines carry a resource name and an
  id (`N1 Reference T1 SCTID:123456 term`); NEL's TSV carries a bare `code` and, for
  composite mentions, `+`-joined codes. Options: a single `code` string, `code` plus
  `code_source`, or a list. Whatever is chosen must round-trip through `core.brat` and
  `read_corpus`, and be what `nel.link_entities` reads as gold (D92, D93).
- **Q16 — What XLT consumes.** Assumed: corpora (`core.corpus`) and model identifiers,
  producing a report table and a manifest. To be confirmed before its subpackage is drawn.
- **Q18 — The umbrella name.** `lab` was chosen as a placeholder and is now the directory
  `src/lab`, the distribution name, the CLI command and every import in the docs. Renaming
  is a find-and-replace, cheapest before anyone depends on it. Supervisor's call.

*Closed:* Q1–Q9 during the migration (scope, cluster scripts → D10, task naming → D48,
Python floor → D17, public API → D65); Q17 (`lab.core` loading torch — the tokenizer import
moved inside `stats._load_tokenizer`, 7.0s → 0.3s, checked by `verify_layering.py`).
