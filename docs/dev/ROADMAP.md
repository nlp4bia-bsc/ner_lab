# Roadmap

What is next, and what is still open. Decisions are in [DECISIONS.md](DECISIONS.md).

## Status

Every library stage of `bsc/NER-API` is migrated: `core.prepare_dataset`, the `Encoder`,
`build_model`, `train`, `ner.train_model`, `ner.search_hyperparameters`,
`ner.predict_entities`. The restructure into `lab.core` + `lab.ner` (D82–D90) and the
integration of `bsc/nlp4bia-linking` as `lab.nel` (D91–D95) both landed on 2026-09-17.
Gold codes entered the corpus contract on 2026-09-25 (D98–D106): `prepare_dataset` takes a
linking TSV, `spans_from_corpus` carries the codes, and `link_entities` either evaluates
against them or, with `keep_gold`, completes around them.
Verification stands at 957 checks across eleven scripts, all passing, plus one deliberate
skip — the NER-API equivalence no longer applies to byte-level tokenizers, whose token
offsets `tokenize_document` trims and NER-API's `DataLoader` does not; see
[`verification/`](../../verification/).

What remains of NER-API is not library surface — `01b_analyze_hpo_trials.py`,
`04_official_eval.py`, `build_gold_test_parquets.py`, `aux_download_baseline_model.py` —
and is a candidate for `examples/` (Q10) rather than migration. `hpo/report.py` already
answers most of what `01b` printed; `04`'s job is `predict_entities` with a `reference`.

`lab.xlt` (cross-lingual transfer) is a planned subpackage; nothing is drawn until what it
consumes is confirmed (Q16).

## Next

- **Persisting the gazetteer's representations.** `link_entities` rebuilds the index over
  the gazetteer on every run. Scope not yet discussed.

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
- **Q13 — NEL's research code.** `scripts/` (retrieval benchmark, triplet generation,
  cross-encoder training), `triplets/`, `retrieval/experiments.py`, the profiler and the
  `rerank_candidates` TSV helper stayed in `bsc/nlp4bia-linking` (D91). Candidates for
  `examples/`, or for a training task once the cross-encoder trainer is needed from the
  library; that trainer would be `training/`'s second consumer and the first reason to move
  any of it to `core`.
- **Q16 — What XLT consumes.** Assumed: corpora (`core.corpus`) and model identifiers,
  producing a report table and a manifest. To be confirmed before its subpackage is drawn.
- **Q18 — The umbrella name.** `lab` was chosen as a placeholder and is now the directory
  `src/lab`, the distribution name, the CLI command and every import in the docs. Renaming
  is a find-and-replace, cheapest before anyone depends on it. Supervisor's call.
- **Q19 — Leading punctuation in decoded spans.** EuroBERT's Llama-3 pre-tokenizer regex
  attaches one preceding punctuation character to a word, so a single token covers
  `+cocaína` and the decoded span opens on the `+`. A few dozen spans, mostly drug mentions.
  Unlike leading whitespace this is not safely trimmable: a gold span may legitimately begin
  with punctuation, so trimming relocates the errors rather than removing them. Decide by
  counting gold spans that start on punctuation against predictions carrying a spurious one.
  Trimming would be a no-op for tokenizers whose punctuation is already its own token.

*Closed:* Q1–Q9 during the migration (scope, cluster scripts → D10, task naming → D48,
Python floor → D17, public API → D65); Q12, Q14, Q15 (gold codes → D98–D106); Q17
(`lab.core` loading torch — the tokenizer import moved inside `stats._load_tokenizer`,
7.0s → 0.3s, checked by `verify_layering.py`).
