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
The encoder methods of `link_entities` keep the gazetteer's embeddings on disk since
2026-10-04 (D107–D111).
Words became Unicode UAX #29 words for every tokenizer on 2026-10-09 (D114).
Sentences are split by pysbd, Punkt or ICU according to `language` since 2026-10-22
(D128–D131).
`lab.summarization` landed on 2026-11-05 with its first stage, MultiClinSum2's summary
evaluation as `summarization.evaluate_summaries` (D132–D139): ROUGE, BERTScore and the
faithfulness, completeness and fluency judges, on a `transformers` or vLLM judge.
`verify_summarization.py` adds 143 checks, including side-by-side equivalence with the
MultiClinSum2 code.
Verification stands at 1,036 checks across eleven scripts, all passing, plus one deliberate
skip — the NER-API equivalence no longer applies to byte-level tokenizers, whose token
offsets `tokenize_document` trims and NER-API's `DataLoader` does not, and for the others it
no longer compares the word-dependent `labels` and `word_ids` (D114); see
[`verification/`](../../verification/).

What remains of NER-API is not library surface — `01b_analyze_hpo_trials.py`,
`04_official_eval.py`, `build_gold_test_parquets.py`, `aux_download_baseline_model.py` —
and is a candidate for `examples/` (Q10) rather than migration. `hpo/report.py` already
answers most of what `01b` printed; `04`'s job is `predict_entities` with a `reference`.

`lab.xlt` (cross-lingual transfer) is a planned subpackage; nothing is drawn until what it
consumes is confirmed (Q16).

## Next

- **Checking ensembling on real models.** D116–D126 are verified on miniature BERTs only
  (`verify_ensemble.py`). On a GPU machine, with two or three trained backbones: a one-member
  `majority` ensemble must reproduce `predict_entities`' spans, and `ensemble_metrics.json`
  shows whether the ensemble beats its best member.
- **Checking persisted embeddings on a real encoder.** D107–D111 are verified with a tiny
  local model only. On a GPU machine, with the SympTEMIST gazetteer and test mentions: recall@k
  and MRR from a loaded index must equal a fresh build's.
- **Checking the judges on a real model.** D132–D139 are verified against the MultiClinSum2
  code side by side and with a canned and a miniature judge. On a GPU, in vLLM's environment,
  with EuroLLM-9B-Instruct over one baseline language: the judge scores must match
  `evaluation_results/eurollm_agent/`, up to vLLM's batch-dependent numerics.
- **Summarization training.** The second stage of `lab.summarization`, over the same
  summaries table.

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
- **Q20 — `mean` pooling averages over padding.** `transformer_faiss`'s default pooling,
  kept from HERBERT, is `last_hidden_state.mean(dim=1)` without the attention mask, so a
  term's vector depends on the longest term in its batch, and therefore on `batch_size`.
  `attention_mask_mean` exists. Changing the default changes every `transformer_faiss` result.
- **Q21 — `dense` does not collapse duplicate codes.** `transformer_faiss` and the lexical
  methods keep each code once at its best term; `dense` returns the top-k terms, so one code
  can fill several of the `top_k` slots and recall@k is not comparable across methods.
- **Q22 — A cross-lingual consistency judge.** MultiClinSum2's prompt compares one
  document's summaries across languages, but its runner only ever asked about one (D137). A
  real one scores per `doc_id` across the languages present, so the summaries table needs a
  key shared across languages — MultiClinSum's `case_id`s embed the language
  (`multiclinsum2_test_1_es`) — and yields one score per document, not per summary.
- **Q23 — BERTScore is not exactly reproducible.** Against MultiClinSum2's stored Greek
  scores, rescoring the same chunk of 32 differs by a mean of 1.3e-3 and at most 1.6e-2. Two
  causes were found. The batch a summary is padded in moves its score by up to about 1e-3 in
  one environment. And transformers 5 reads mdeberta's full-width punctuation (`，（）`) as
  `[UNK]`, where sentencepiece, which MultiClinSum2's transformers 4 used, NFKC-normalises it
  to ASCII; the Qwen baseline's Greek summaries hold Chinese. NFKC-normalising text before
  BERTScore would restore the old tokens, at the cost of a behaviour change.
- **Q24 — Unreadable answers score 0.** D136 keeps MultiClinSum2's rule, so a judge that
  fails to answer lowers the mean as if it had judged the summary worthless. Leaving the
  score missing instead would report the mean over answers that were read, beside the count.
- **Q25 — Completeness reports no confidence.** Its prompt does not ask for one, so every
  completeness `confidence` is 0. Asking would change the prompt, and with it every score.

*Closed:* Q1–Q9 during the migration (scope, cluster scripts → D10, task naming → D48,
Python floor → D17, public API → D65); Q12, Q14, Q15 (gold codes → D98–D106); Q17
(`lab.core` loading torch — the tokenizer import moved inside `stats._load_tokenizer`,
7.0s → 0.3s, checked by `verify_layering.py`).
