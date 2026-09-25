# Decisions

Numbered, append-only. A superseded decision keeps its number and points at the one that
replaced it, so citations still resolve; its old text is in git history.

A row is one line: what was decided, and why in a clause. Where the reasoning needs more, the
row says `→ notes` and the section under the same number at the bottom of this file has it.
The row is the verdict; the note is the argument.

## Register

| # | Decision | Rationale |
|---|---|---|
| D1 | Superseded by D82. | |
| D2 | Superseded by D83. The `src/` layout itself stands. | Prevents accidental imports from cwd; standard for installable packages. |
| D3 | Superseded by D11. | |
| D4 | Restated as D11/D12. | |
| D5 | Superseded by D85. | |
| D6 | Drop deps `dataclasses`, `pathlib`, `typing` | stdlib since 3.7 / 3.4 / 3.5; floor is 3.10. Installed as real redundant files, inert only because stdlib precedes site-packages in `sys.path`. |
| D7 | `torch==2.10.0` stays an exact pin | All known users need exactly 2.10, not a floor. |
| D8 | No tests in the package. No `pytest` dep. Behavioural checks live in `verification/`, outside `src/`. | User decision. Consequence accepted: no regression guard beyond the verification run. |
| D9 | Build backend: hatchling | uv-native, no `setup.py`, minimal config. |
| D10 | Nothing MN5- or SLURM-specific ships. No `sh/`. Pure Python. | Users write their own submit script around the library. |
| D11 | **One CLI command: `lab run <config.yaml>`.** No per-stage subcommands. | ~9 stages × ~25 flags = ~200 argparse declarations, each duplicating a function parameter and needing edits in two places on every signature change. |
| D12 | YAML is flat, with a `task:` key naming the operation. One file fully describes one run. | Dispatch is a dict lookup; the file is self-contained and readable on its own. |
| D13 | **Two override flags:** `--random-state`/`--seed` and `--output-dir`, plus `--version`. No generic `--set key=value`. Everything else lives in the YAML. | Covers the SLURM job-array case (sweep seeds without writing N near-identical YAML files) without reopening a per-stage flag surface. |
| D14 | HPO and training may write to disk; `output_dir` stays an explicit required parameter, and the function still **returns** its result. | Ray writes per-trial dirs and Trainer writes checkpoints — a 40-trial GPU-day sweep can't live in memory. Precedent: `Trainer` requires `output_dir` and `train()` returns `TrainOutput`. What we kill is "the only way to see a result is to read a file afterward". |
| D15 | Manifests (`sha256`, `run_manifest.json`, dataset-name derivation) are behaviour of the persistence layer. | The pure path must not touch them. |
| D16 | Augmentation is cut from the first release. | `00b_augment_data.py` was broken in NER-API anyway (expected `--input-jsonl`; nothing emitted jsonl since the parquet migration). |
| D17 | Python floor `>=3.10`. | Most clusters are at this version. |
| D18 | Drop deps `nltk`, `pydantic`, `psutil` (zero imports). Keep `accelerate` and `optuna` despite zero direct imports. | `accelerate` is required by `Trainer`; `optuna` by `ray.tune.search.optuna.OptunaSearch`. Both would otherwise look dead and get pruned by someone later. |
| D19 | `ray` must be `ray[tune]`. | Code imports `ray.tune`, `ray.air`, `ray.tune.search.optuna` — none ship in the base wheel. |
| D20 | No explanatory comments in library code. Rationale goes in this document. | Comments that narrate what the next line does add maintenance surface. Comments are for the non-obvious only (e.g. why a dependency with no imports is required). |
| D21 | No `quickstart()` function. A runnable script in `examples/` instead. | A function whose only value is calling three others is dead weight as an API; as an example it is something to read and copy. |
| D22 | No shared multi-stage pipeline config, and no blocking `run_pipeline()`. | Stages have incompatible runtime shapes: seconds/CPU vs GPU-days/Ray/multi-GPU vs GPU-hours/single-GPU. Chaining them in-process either camps on a login node for days or reimplements SLURM dependencies badly. Point two runs at the same YAML if you want to; nothing forces it. |
| D23 | **Entity overlap resolution stays at encoding time, not corpus build time.** The canonical parquet holds raw source annotations, overlaps intact. | Keeps the corpus faithful to the source; overlap handling is a modelling choice, so several policies can be compared without regenerating the corpus. Consequence: `n_entities` and the split balance report count *unresolved* entities. |
| D24 | Superseded by D83. | |
| D25 | `n_entities` is pinned to `int32`, dropping NER-API's `downcast="integer"`. | A data-dependent dtype is a footgun: a corpus fitting `int8` silently changes schema when a densely-annotated document is added. |
| D26 | Implicit behaviour is exposed as public functions that the orchestrator defaults to, never buried inside it. | `derive_dataset_name`, `split_descriptor`, `read_dataset_metadata` are callable and overridable. The `<output_dir>/<dataset_name>/<split_descriptor>/` layout stays in the library because the training stage reads it — a contract, not a convenience. |
| D27 | One name per concept across the library: `random_state`, not `seed`. CLI accepts `--random-state`/`--seed` as aliases for one destination. | Two names for one thing is worse than either. Default unified on `DEFAULT_RANDOM_STATE`; the seed is recorded in the manifest. |
| D28 | `target_label` is validated against the labels present in the corpus, not against `CANONICAL_LABELS`. | The data layer is entity-agnostic (verified on a `GENE/protein/cell-line/Chemical` corpus). `CANONICAL_LABELS` is an alias table for normalization, not a whitelist. |
| D29 | **Script-shaped stages (HPO, inference) are migrated top-down, not bottom-up.** Agree the user-facing signature and return type first, then fill in behind it. | Migrating bottom-up would mean lifting a 691-line `main()` and inheriting Ray-script structure as the public API. |
| D30 | Superseded by D38. | |
| D31 | **Encoding never normalizes labels.** The corpus is the source of truth by then; `Encoder` uses the labels it is given. | NER-API re-normalized unconditionally at load, silently overriding a corpus built with `normalize_labels=False` — the same knob in two layers with the inner one winning. |
| D32 | **`normalize_labels` defaults to `False` everywhere.** A corpus keeps the labels its annotations declare; normalization is opt-in. | Renaming a corpus's own vocabulary by default is not ours to do. → notes |
| D32b | The alias table stays reachable without being front-of-house: `LABEL_ALIASES` and the `normalize_*` functions are exported from `lab.core` and take an `aliases` mapping. `build_corpus` and `prepare_dataset` keep the plain boolean. | A module-level dict users are expected to mutate is a trap. A parameter defaulting to `LABEL_ALIASES` gives a corpus with its own vocabulary a supported path — normalize the annotations, then hand them to `build_corpus`. |
| D33 | **`language` is a required argument**, on `Encoder`, `split_into_sentences` and `compute_text_stats`. Not read from `metadata.json`, not defaulted to `"es"`. | A fallback here is a silent guess: pysbd segments a Spanish clinical note wrongly under `en` without erroring, which shifts every window boundary. Mandatory is one rule with no hidden state. |
| D34 | **`ner/models/` builds architectures and nothing else.** No training, no prediction, no saving, no metrics. | NER-API's base model class was 661 lines doing all of that, so swapping an architecture meant subclassing the trainer. Split, the architecture is a function you can replace with your own. |
| D35 | Architectures are `"linear"` / `"crf"` / any callable `(base_model, label2id, id2label, **kwargs) -> model`, resolved by `build_model`, same shape as `Encoder`'s window strategies. | One extension mechanism for the library, not two. NER-API's "simple" is renamed `"linear"` — it describes the head. |
| D36 | The two `*Config` dataclasses (46 and 49 fields) are **not** ported. | Both were ~90% training and run plumbing with 3 genuinely model-level fields, now keyword arguments on the CRF constructor. |
| D37 | BIO constraint reasoning lives in a torch-free `ner/models/bio.py`. | Pure combinatorics over label strings, and the part most likely to be silently wrong; keeping it out of the tensor code makes it verifiable without torch. |
| D38 | **`train()` accepts a `transformers.TrainingArguments`.** `training_arguments(output_dir, **overrides)` builds one from a module-level `DEFAULTS` dict. | Battery = the helper's defaults; removable = build your own and pass it. Every `TrainingArguments` keyword works without us enumerating it. |
| D39 | The settings HF has no slot for stay explicit parameters on `train()`: `early_stopping_patience`, `pad_to_multiple_of`, `ignore_index`, `metrics_scope`, `save_model`, `track_resources`. | Six named arguments beat smuggling them into a config object HF would ignore. |
| D40 | `compute_metrics` is injected, not built. `trainer_class` defaults to `CRFTrainer` for a model exposing `decode_from_emissions`, `Trainer` otherwise. | Decouples training from evaluation. Duck-typing the CRF trainer means a custom architecture that can Viterbi-decode gets the right Trainer for free. |
| D41 | Discharged by D47. | |
| D42 | No `target_label` parameter on `train()`. Provenance goes through `run_metadata: dict`. | Nothing in training needs it once `compute_metrics` is injected and the model carries its own `label2id`. |
| D43 | **No multi-label.** One `target_label`, three classes, indefinitely. | User decision. Removes the pressure to design `2n+1` seams into evaluation that nothing would use. |
| D44 | **nervaluate/SemEval is the canonical scorer.** ~~The MultiClinNER official scorer stays available as an option.~~ The option is removed by D96. | User decision. `span_strict_f1` is what model selection tracks. |
| D45 | The legacy word-level scoring path is dropped (~250 lines). | `Encoder` only ever emits token-level rows, so the branch was unreachable. |
| D46 | `spans_from_corpus` takes labels **verbatim**. Now `lab.core.spans_from_corpus` (D96). | D32 applies: gold is scored as distributed. |
| D47 | `training_arguments()` defaults to `metric_for_best_model="span_strict_f1"` / `greater_is_better=True`. `train()` raises if a span metric is requested with no `compute_metrics`, naming `build_compute_metrics`. | Matches NER-API and the group's expectation. The guard turns a confusing mid-training KeyError into an immediate, actionable one. |
| D48 | **The training orchestrator is `ner/training/assessment.py::train_model`, task `ner.train_model`.** No `pipeline/` package. | In `transformers`, `pipelines` means inference; a tier named after itself is a layer that exists only to be a layer. |
| D49 | **`train_model(training_arguments=...)` accepts a `TrainingArguments`, a mapping of overrides applied to `DEFAULTS`, or `None`.** No `best_config.json` hand-off. | The mapping form is what makes the YAML path work, and it is not an opaque bag: its schema *is* `TrainingArguments`, and an unknown key raises on construction. Reading a JSON file back into a config object would make a file format part of the API. |
| D50 | The orchestrator returns metric tables and summaries — `fold_metrics`, `aggregate`, `summaries` — never `TrainingResult` or a live `Trainer`. | Holding one `Trainer` per fold keeps every fold's model resident on the GPU. `save_model=True` writes it, or call `lab.ner.train` directly. |
| D51 | `fold_metrics.parquet` and `assessment_summary.json` are written in **both** split modes; `fold_metrics` rows are reconstructed from `epoch_metrics` at the best epoch. | A single run and a k-fold run differ in how many rows the table has, not in whether it exists — downstream analysis does not branch on mode. |
| D52 | **`run_manifest.json` records no git commit.** | NER-API ran `git rev-parse HEAD` against its own repo, which a library cannot do: the interesting commit is the *user's* project. Recorded as Q11. |
| D53 | `train_model(random_state=...)` sets `TrainingArguments.seed`. | D27, and the CLI's `--random-state` reaches whatever task is named. |
| D54 | **HPO is not a loop over `train_model`.** A trial calls `train()` directly with pre-encoded rows; the corpus is windowed once per (base_model, strategy, context_tokens, max_length) variant per sweep. | `train_model` encodes internally, so a trial calling it would re-window the corpus every trial. |
| D55 | **The search space is Ray domains or a declarative mapping, interchangeably.** `build_search_space` merges `search_space` over `DEFAULT_SEARCH_SPACE`; a value may be a `tune` domain, the YAML mapping form, a scalar (pinned) or `None` (removed). | The declarative shape is what the sweep manifest records; merge-over-defaults matches D49. → notes |
| D56 | **HPO takes `split_dir`, like `train_model`.** A k-fold split is searched against one rotation, `validation_index`; the fixed holdout is never read. | One YAML value points both stages at the same data. Searching across all folds would multiply a GPU-day sweep by the fold count; the k-fold mean is the assessment's job (D51). |
| D57 | **`hpo_summary.json` records the winner as a ready-to-run `train_model` YAML config**, beside the raw trial record. Output only — nothing reads it back. | The hand-off survives as a document for the user, not as a file format in the API. Verified by running the emitted block through `train_model(**block)` unchanged. |
| D58 | **`train()` allows early stopping without `load_best_model_at_end` unless `save_model=True`.** | The old unconditional raise blocked the sweep shape: no checkpointing, early stopping on. Verified against transformers 5.14.1 that the stop logic works without saving. The raise stays where the footgun is real: saved weights would be the last epoch's. |
| D59 | No `max_epochs` parameter on `search_hyperparameters`. The cap is `num_train_epochs`, defaulting to 40 via `HPO_ARGUMENT_DEFAULTS`. | D27: it already has a home in `training_arguments`. |
| D60 | **The winner block is also written to `winner.yaml` in the sweep's run directory**, with `output_dir: <sweep>/final_train` (D77). `write_winner_config` is public. | D57 left the hand-off correct but manual — retyping ~18 keys where a typo silently changes the run. |
| D61 | **Inference is `ner/inference.py::predict_entities`, task `ner.predict_entities`.** It takes a `model_dir`, not a run directory. `--random-state` raises a `TypeError` here. | Pointing at the model keeps inference independent of the orchestrator's artifact layout: a `fold_XX/best_model` and a directory copied to scratch load identically. Inference is deterministic, and D42 forbids an unused parameter kept to satisfy a flag. |
| D62 | **A saved model carries an `encoding.json` beside its weights**, written by `train(model_encoding=...)`. Output only — `train_model` never reads one. | The artifact describes itself, as HF's `config.json` does; it is also the only thing that makes a CRF checkpoint loadable. → notes |
| D63 | `Encoder(require_target_label=False)` skips the check that `target_label` occurs in the corpus. Default stays `True`. | Rule 3. The guard is right for training, where the typo it catches produces an all-`O` corpus; wrong for inference, where unannotated documents legitimately contain no entity. |
| D64 | **Per-span `score` and `text` are decoded in `ner/evaluation/spans.py`**, not again in `inference.py`. Both off by default. | Two implementations of B/I transition handling that must agree forever is exactly the duplication the migration exists to remove. |
| D65 | **The stage entry points are re-exported lazily from their subpackage, via a module-level `__getattr__`.** Nine on `lab.ner`, `prepare_dataset` on `lab.core`. Everything else keeps its submodule path. | The folder structure carries meaning; lazy so `import lab.ner` stays instant and torch-free. → notes |
| D66 | **The pretrained starting point is `base_model`, never `checkpoint`.** `checkpoint` keeps its HF meaning — the `checkpoint-*` snapshots the `Trainer` writes. | D27 in the mirror direction: one name was doing two concepts, and they collided in the run manifest. → notes |
| D67 | **`source_manifest.json` records `n_entities_by_label`**, built by the public `lab.core.count_labels`. | A manifest saying 15064 entities without saying of what cannot help pick a `target_label`. Consequence: `parse_document_label_counts` raises when a document's `n_entities` disagrees with its `entities_json`, now on every `prepare_dataset` path. |
| D68 | **`gpus_per_trial` is removed from `search_hyperparameters`.** A trial reserves one GPU when the cluster reports any, none otherwise. | GPU sharing should not be a top-level knob; an unconditional GPU request makes Ray wait forever on CPU-only machines. → notes |
| D69 | **`DEFAULT_SEARCH_SPACE` searches `warmup_steps`, not `warmup_ratio`**, over the same `uniform(0.0, 0.1)` range. | transformers deprecates `warmup_ratio`; `warmup_steps < 1` is read as a proportion, so the sweep samples exactly the values it did before. |
| D70 | **The dependency is `nvidia-ml-py>=12.0.0`, not `pynvml`.** `import pynvml` is unchanged. | `pynvml` is a deprecated shim that warns on import; the distribution was renamed, the module was not. The floor is the one `pynvml` itself declared. |
| D71 | **Mixed precision is resolved per device, not fixed in `DEFAULTS`.** bf16 where supported, fp16 on pre-Ampere, neither without CUDA; naming either in the overrides takes them verbatim. | `fp16: True` was never a decision and is wrong on Ampere+. → notes |
| D72 | **`lab` ships `py.typed`, and each lazily-exporting subpackage an `__init__.pyi` stub declaring its exports.** | D65's `__getattr__` made every top-level attribute `Any` to type checkers. → notes |
| D73 | **Public signatures are annotated where the type is unambiguous; the model factories are deliberately left bare.** | The factories' only true common supertype is `nn.Module`, which would promise more than the pluggable architecture requires. → notes |
| D74 | **A document/annotation filename mismatch is reconciled by policy**: `on_mismatch: "error" \| "documents" \| "annotations"`, default `"error"`; `resolve_mismatch` is public and returns a `SourceMismatch` report. | Rule 3: a policy that drops the user's data carries an off switch, and the switch defaults to strict. → notes |
| D75 | Amended by D78: the `training/devices.py` module stands; `train_model` pins rather than refuses. | The regression that motivated it is in D78's note. |
| D76 | **The run manifest records `devices.policy`, `devices.trained_on` and `devices.effective_train_batch_size`.** | The policy is not derivable from anything else in the manifest: `gpu_count: 4` is otherwise ambiguous between a deliberate throughput run and a bypass, and a stderr warning is lost when a SLURM log rotates. |
| D77 | **`output_dir` is the run directory.** `run_name` is deleted; `claim_run_dir` refuses a directory holding a `run_manifest.json` unless `overwrite=True`. | Following HuggingFace: `run_name` was exactly `output_dir/"foo"`, and deleting it closes three defects at once. → notes |
| D78 | **`train_model` takes `devices: Literal[1, "all"] = 1` and pins to one GPU instead of refusing.** `require_single_device` stays the raise for HPO trials. | `n_gpu` is a plain property HF itself forces down to avoid `DataParallel`; pinning uses that seam and is order-independent, unlike `CUDA_VISIBLE_DEVICES`. → notes |
| D79 | **An annotation whose text disagrees with its document is resolved by `on_conflict: "raise" \| "rewrite"`**, default `"raise"`; the document always wins. | Labels come from offsets alone, so the `.ann` string can only win by relocating offsets, which was scoped out. → notes |
| D80 | **`on_conflict` gains `"drop"`**, removing every conflicting document with all its annotations; `resolve_conflicts` returns the resolved pair plus the report. | Rewrite is only right when offsets are trustworthy; dropping the whole file is the third honest answer, and dropping only the row would train `O` over a real entity. → notes |
| D81 | **`core/stats.py::compute_text_stats`/`compute_annotation_stats` take `base_model: str`, not a tokenizer, and `language` is required.** Amended by D87, which puts the `stats` switch on `prepare_dataset`. | These are the simplest-script entry point; `base_model` matches `train_model` (D66) and is honest about what is passed. `language` per D33. |
| D82 | **`ner_lab` becomes one subpackage of a group-wide library, `lab`.** Supersedes D1. | One repo, one namespace, one docs site and CLI for the unit; per-tool environments come from extras (D85), not from separate libraries. |
| D83 | **Layout is `lab.core` / `lab.ner` / `lab.nel` / `lab.cli`, and imports point one way: `core ← {ner, nel}`.** Task subpackages never import each other. Supersedes D2, D24. | A subpackage is defined by the artifacts it consumes and produces; cross-tool composition happens through `core`'s contracts, not through imports. |
| D84 | **`core` holds the data contracts and nothing torch: corpus, spans, split, stats, segmentation, generic span scoring, provenance, task registry. HPO and training stay under `ner`.** | Nothing moves to a shared layer until a second subpackage needs it; segmentation has two consumers, HPO and the trainer have one. |
| D85 | **Dependencies are extras: `lab` alone is torch-free; `lab[torch]` carries the shared `torch`/`transformers` pin; `lab[ner]`, `lab[nel]` build on it.** Supersedes D5. | D5's measurement (extras save nothing for HPO users) stopped holding once a tool existed that does not need ray, CRF or pysbd. The shared pin keeps the unit on one torch (D7). |
| D86 | **Task names are namespaced — `ner.train_model`, `core.prepare_dataset` — each subpackage owns a string-only `tasks.py`, and `core.tasks.resolve_task` turns a missing extra into a message naming it.** The CLI stays `lab run`, plus `lab tasks`. | The YAML is the argument surface; per-subpackage subcommands would duplicate the `task:` key and reopen the flag surface D11 closed. |
| D87 | **`cli.run` is `task(**parameters)` and nothing else.** `prepare_dataset` owns `stats`/`base_model`/`language`. Amends D81. | The CLI cannot know one task's parameters once it dispatches for several subpackages. → notes |
| D88 | **The restructure landed and was verified before any NEL code entered the repo.** | A rename is verified by existing checks; an integration has none yet. One commit boundary each keeps a failure attributable. |
| D89 | **No `ner_lab` compatibility shim.** | 0.1.0, unpublished, internal users only. |
| D90 | **The one-way import rule is checked by `verification/verify_layering.py`**, a static scan, not by import-linter. It also checks that `core` and the CLI import no torch. | No CI exists to give a linter contract teeth; the verification suite is what runs before a stage is called done. |
| D91 | **`lab.nel` takes `nlp4bia-linking`'s library surface, not its research code.** In: records, readers, the lexical matchers, the retrievers, the cross-encoder reranker, RRF, retrieval and hierarchy metrics, the pipeline. Out: `scripts/`, triplet generation, FAISS profiling, the profiler, corpus-specific helpers. | The package's own docs say the scripts are never imported; the rest is experiment tooling with corpus paths inside. Their fate is Q13. |
| D92 | **`nel.link_entities` appends `code`, `code_term`, `code_score`, `candidates_json` to the span table it is given; an incoming `code` column is gold, kept as `gold_code`, and scored.** | The prefixed names keep NER's `score` and NEL's score apart in one row. → notes |
| D93 | Superseded by D98. | |
| D94 | **NEL's records and class names are kept as written; the span table is converted to and from them at the task boundary, in `nel/linking.py`.** | Equivalence before improvement: the side-by-side checks compare the ported classes against the original package call for call. |
| D95 | **`lab[nel]` is `lab[torch]` plus scikit-learn, networkx, tqdm, sentence-transformers and faiss, with a platform marker choosing `faiss-gpu` on Linux x86_64.** NEL's `transformers<5` cap is dropped. | sentence-transformers 6 runs against transformers 5.14; the verify venv proves it on the lexical, sparse and FAISS paths. |
| D96 | **The MultiClinNER scorer is removed; `lab.core.score_characters` adds character-level P/R/F1 to the canonical metrics.** `ner/evaluation/multiclinner.py`, `InferenceResult.official` and `multiclinner_eval.json` are gone; `spans_from_corpus` moves to `lab.core.spans`. Amends D44, D46. | Its strict F1 was verified identical to nervaluate's except where it was wrong (predictions in a gold-less document never counted as false positives); its character F1 is replaced by the standard character-as-unit definition, which needs no matching step. → notes |
| D97 | **`predict_entities` scores span and character metrics against the gold as annotated, not against the gold reconstructed from windows.** Token diagnostics stay window-based and are added only when `documents` itself carries the gold; `reference` takes precedence over the corpus's entities. | Reconstructed gold has already lost what overlap resolution merged and windows truncated, so scoring against it cannot count those as missed. Training keeps the window-based number because rows are all it has, so the two `span_strict_f1` values may legitimately differ. |
| D98 | **Gold codes enter the corpus through `prepare_dataset`, from a linking TSV whose path the caller passes; each row is joined to its BRAT entity on `(filename, start, end)`.** `.ann` files are not read for codes. Supersedes D93. | The corpora in use (SympTEMIST) ship codes only in TSVs beside the BRAT, several variants per split; which one is gold is the experiment's choice, not the library's. |
| D99 | **A code is an optional `code` string on each `entities_json` entry, written exactly as the TSV has it — composites stay `+`-joined.** `sem_rel`, `is_abbrev`, `is_composite` and `need_context` are not carried. | It is what NEL's own TSV reader yields, so `link_entities` sees the same gold as before (D94); the other columns have no consumer yet and can be added later as further optional fields. |
| D100 | **`NO_CODE` is kept as a code.** | It records an annotator's judgement that no concept applies, which is not the same as a mention nobody coded. |
| D101 | **A TSV row whose code is unusable is skipped: the entity stays, uncoded.** Identical duplicate rows count once. | The source carries spreadsheet damage (`1.66753E+16`, a literal `N+O`) whose original ids cannot be recovered. |
| D102 | **A corpus carries codes when its entries do: a coded entity has a `code` key, an uncoded one has none.** No corpus-level flag, no per-document column. Closes Q14. | A corpus prepared without codes stays byte-identical, so the document fingerprints that guard split reuse do not move; counts belong in the manifest. |
| D103 | **"Unusable" (D101) is only the obvious damage: a code in spreadsheet scientific notation, or one with a `+`-part that is empty or has no digit and is not `NO_CODE`.** Everything else passes as written. | Checking codes is the corpus's job, not the library's; the filter is a courtesy, and a digit rule holds for ICD-10 and HPO ids as well as SNOMED. |
| D104 | **A TSV row naming no BRAT entity, or two rows giving one span different codes, raises by default; `on_code_mismatch="drop"` skips them with a warning and lists them in the source manifest.** Rows for documents another policy already dropped go with them. | The same contract as `on_mismatch` and `on_conflict`: a disagreeing source is usually a broken export, and dropping is a choice the caller makes and the manifest records. |
| D105 | **`source_manifest.json` reports codes from the corpus itself: `has_codes`, entities with a code (overall and by label), `NO_CODE`, composite and unique counts; with a TSV, also its path, sha256, row count and skips.** | Computed from the corpus, the numbers hold for a coded `source_parquet` too, not only for one built from a TSV. |
| D106 | **`link_entities` evaluates by default: every span is linked and incoming codes are gold. `keep_gold=True` completes instead: coded spans keep their code, only uncoded spans are linked, a `code_source` column says which is which, and nothing is scored.** Amends D92. | A partly linked corpus is only useful filled in, but filling it in leaves nothing to score, so the two uses cannot share one behaviour. |

## Notes

### D32 — `normalize_labels` defaults to `False`

CARMEN-I's `.ann` holds `FARMACO` and its `metadata.json` declares `FARMACO`; the two agree,
and the only thing overruling them was `LABEL_ALIASES`, a table inside this library. With the
default off nothing is rewritten unless asked, so no metadata-precedence machinery and no
rename warning are needed. Normalization was never blanket reassignment: unknown labels pass
through canonicalized (accents stripped, uppercased, `-`/space → `_`); only the alias keys
are reassigned.

### D55 — search space forms

The declarative `{"type": "loguniform", "low": ..., "high": ...}` form is what the sweep
manifest records, and `describe_search_space` round-trips it back. Pin-and-remove (a scalar
pins, `None` removes) is what makes a partial override usable. `variant` is reserved, built
from `base_models`/`strategies`/`context_tokens`/`max_lengths`. `effective_train_batch_size`
is searched with the micro batch derived under `max_micro_batch_size`, because searching
both produced duplicate experiments.

### D62 — `encoding.json`

Restating eight encoder values where a wrong `max_length` silently shifts every window
boundary is the transcription failure `winner.yaml` had just been added to remove (D60).
It is also the only thing that makes a CRF checkpoint loadable: `CRFForTokenClassification`
is an `nn.Module`, not a `PreTrainedModel`, so `Trainer.save_model` writes a bare state dict
with no `config.json` to rebuild the wrapper from. `train()` without it still saves weights;
only a linear model can be reloaded from that.

### D65 — lazy re-exports

**Which names:** one is top-level if it starts a stage or must be constructed to start one.
Result dataclasses are excluded (you never construct one), as are the mutable defaults
`DEFAULTS`/`DEFAULT_SEARCH_SPACE`/`HPO_ARGUMENT_DEFAULTS` (D32b's trap). ~180 public names across the subpackages is the pandas shape, where the folder
structure carries meaning, not the transformers one; re-exporting everything would move the
haystack. **Why lazy:** a package `__init__` executes before any submodule, so a plain import
of the HPO entry point would make every import cost ray + torch — measured 0.61s → 6.6s — and
raise in an environment without torch. With `__getattr__`, importing the package loads
nothing and only the torch functions pull torch. Precedent is `transformers`' own `__init__`.

### D66 — `base_model`

`run_manifest.json` wrote `model.checkpoint` meaning the pretrained encoder while
`training_summary.json` wrote `best.checkpoint` meaning a Trainer snapshot, in one run
directory. `base_model` over `base_model_path` because the values are usually Hub identifiers
(HF's own argument is `pretrained_model_name_or_path` for that reason); plural `base_models`
in HPO because there it is a search dimension.

### D68 — no `gpus_per_trial`

Hardcoding `{"gpu": 1}` was rejected once the consequence surfaced: the end-to-end sweep
verification runs on a CPU-only machine, and an unconditional GPU request does not fail
there, it makes Ray **wait forever**. Resolving from `ray.cluster_resources()` keeps the knob
gone and the sweep testable. The residual risk — a silent CPU downgrade on a misconfigured
allocation — is blunted by the smoke test running first and by the manifest recording what
was resolved. `trial.require_single_device` is the real single-GPU guard.

### D71 — mixed precision per device

fp16's smallest normal is 6.1e-5 while transformer gradients reach 1e-9, so it needs dynamic
loss scaling and can diverge if the scale collapses. bf16 carries fp32's exponent range, so
none of that machinery exists. There is no speed trade: A100/H100 tensor cores run both at
identical throughput. A fixed default cannot be right for both, since V100/T4 lack bf16. The
cost accepted is that numerics depend on the node, which is why the run manifest records the
resolved dtype beside the GPU that chose it. Side effect: CPU runs work unaided, where
`fp16: True` used to be invalid.

### D72 / D73 — typing

A module that defines `__getattr__` makes every attribute a type checker cannot bind
statically resolve to `Any`, so a typo type-checked clean and nothing had a signature or a
completion. A `.pyi` takes priority over the `.py` for pyright and mypy, so the stub is the
declared surface while the runtime keeps the lazy import; the stub's `__all__` is asserted
equal to the runtime's. `py.typed` is the other half — without it an editable install is a
third-party library whose annotations a checker is not permitted to trust. The model
factories stay bare because `build_model` returns a `PreTrainedModel`, or a
`CRFForTokenClassification` (an `nn.Module`), or whatever a user's callable hands back;
declaring `nn.Module` would promise what the pluggable architecture does not require.
`encoding/` and `evaluation/` import `PreTrainedTokenizerBase` under `TYPE_CHECKING` only,
keeping the torch-free half torch-free.

### D74 — `on_mismatch`

The two directions were asymmetric before: an `.ann` with no `.txt` raised, while a `.txt`
with no `.ann` was silently kept with zero entities. `"documents"` keeps that asymmetry and
only drops orphan annotations; `"annotations"` makes the annotation set define the corpus,
additionally dropping unannotated documents — a real behaviour change for that side, hence
opt-in. Neither policy can keep an orphan annotation: there is no text for its offsets. The
report is a return value rather than `DataFrame.attrs`, so `prepare_dataset` resolves its own
sources and hands `build_corpus` the aligned pair. A policy that would drop *every* document
raises instead of returning an empty corpus.

### D77 — `output_dir` is the run directory

`run_name="foo"` was exactly `output_dir=output_dir/"foo"`, so the simplification is
deletion. It closes three defects: the collision with `TrainingArguments.run_name`, which
both reached through `train_model` meaning unrelated things; the unreachable `timestamp`
parameter on `run_directory_name`; and `run_name="/abs"` silently discarding `output_dir`
(`Path("/a/b") / "/etc"` is `/etc`). The timestamp had been doing two jobs, organising and
guaranteeing uniqueness; flat keeps the first, so the overwrite guard supplies the second.
It is checked once, before any `mkdir`, so a wrong path costs a second rather than
GPU-hours. `run_directory_name()` stays public, so the old layout is one call away. The
winner config emits `<sweep>/final_train` because the sweep's parent `output_dir` would
scatter final-train artifacts across the user's top-level runs directory.

### D78 — pinning to one GPU

The regression that started this (D75) was confirmed, not hypothetical: the sweep searched
at batch 16 and the retrain on `--gres=gpu:4` trained at 64 via `nn.DataParallel`, which
the Trainer applies whenever `n_gpu > 1`. Steps per epoch fall by the device count and every
step-denominated setting follows — a fractional `warmup_steps`, the cosine decay — so the
sweep's validation score never reproduced on retrain. D75 refused such allocations and told
the caller to set `CUDA_VISIBLE_DEVICES`, but that only works before CUDA initialises, which
`arguments.py` does early via `is_bf16_supported()`; the fix could only be applied outside
the process, and the raise came *after* a sweep had spent its GPU-hours. Pinning instead:
`n_gpu` is a property over `_n_gpu` that HF itself forces down to avoid `DataParallel`, and
`train_batch_size` derives from it, so `devices=1` reads `arguments.device` to freeze the
cached `_setup_devices`, then sets `_n_gpu = min(visible, 1)`. **Why `Literal[1, "all"]`:**
`Trainer` wraps with a bare `nn.DataParallel(model)` whose `device_ids` defaults to every
visible device regardless of `_n_gpu`, so `devices=2` on a 4-GPU node would make the batch
accounting say 2 while replication spanned 4 — the exact silent mismatch this exists to
prevent. **Why trials keep the raise:** a trial holds a one-GPU Ray reservation, so a second
visible device means the reservation was bypassed; pinning would hide a broken cluster
setup behind a run that looks fine.

### D79 / D80 — `on_conflict`

Labels are assigned from character offsets alone (`tagging.py` tags a token by overlap with
`entity["start"]`/`entity["end"]`), and the entity's `text` never reaches the model. Storing
the `.ann` surface form against unchanged offsets would record one thing and train on
another, so the `.ann` can only "win" by relocating offsets, which needs a search with
ambiguous and absent cases — scoped out. `"rewrite"` replaces the surface form with
`text[start:end]`, the one resolution that leaves the corpus internally consistent; the
downstream raise sites stay strict, and an out-of-range offset still fails validation.
`"drop"` exists because rewrite is only right when the offsets are the trustworthy half;
when they drifted, rewriting mislabels a span. Dropping only the conflicting row was
rejected: the document would read as fully annotated while a real entity is absent, training
`O` over it. `resolve_conflicts` returns the resolved pair so a caller cannot forget to act on
the report and write a corpus with the conflicting documents kept but their annotations
stripped. Both policies warn when they fire; the default raises.

### D87 — `stats` on `prepare_dataset`

`prepare_dataset` validates `stats`/`base_model`/`language` up front with its other inputs,
so a missing `base_model` fails before any conversion starts, where the old CLI branch only
failed after the corpus was built. One behaviour change: the CLI branch popped
`normalize_labels` off the YAML and passed it only to the annotation statistics, so from YAML
the corpus itself was never normalized although the documentation said it was. Inside the
task the one parameter reaches both. The cost D81 avoided — `dataset.py` importing
`stats.py` — is paid.

### D96 — character metrics replace the MultiClinNER scorer

Checked side by side on perturbed synthetic predictions before removal: strict P/R/F1 was
identical in every ordinary case, since both are greedy exclusive matching with P = TP /
n_pred and R = TP / n_gold. The two differences were MultiClinNER's own quirk — only gold
documents are visited, so predictions in a document with no gold are never false positives
— and nervaluate's order dependence (an overlapping prediction listed before the exact one
consumes the gold span), which is inherited from NER-API and logged as an open idea. The
ported `char_f1` was a mean best-overlap score per annotation with non-exclusive matching;
the replacement counts characters, so it is order- and overlap-independent and reads as a
plain P/R/F1 with counts (`char_correct`, `char_missed`, `char_spurious`). It lives in
`lab.core.scoring` so it reaches every consumer at once: `span_metrics`, training's
`compute_metrics` and `epoch_metrics.parquet`, `predict_entities`, and any two span tables.

### D92 — the linked span table

NEL's own TSV format already used `code` for the *gold* code, and its output wrote the
linking score as `score`, the column NER's predictions carry for their confidence. NEL's
output verbatim (`gold_code`, `predicted_code`, `predicted_term`, `score`, `method`) was
rejected for the prefixed set because a row that came out of `predict_entities` and went
through `link_entities` then reads unambiguously — `score` is still what NER meant,
`code_score` what NEL meant — and `code` on the output is what a downstream consumer wants to
read. `candidates_json` holds the whole `top_k` list because the reranker, RRF provenance and
any error analysis need more than the top hit, and one JSON cell keeps the table one row per
mention. The per-run `method` is in the manifest, not on every row.
