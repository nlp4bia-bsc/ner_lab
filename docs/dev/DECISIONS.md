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
| D17 | Superseded by D129. | |
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
| D33 | **`language` is a required argument**, on `Encoder`, `split_into_sentences` and `compute_text_stats`. Not read from `metadata.json`, not defaulted to `"es"`. Amended by D128: it is an ISO 639-1 code, and any other value raises. | A fallback here is a silent guess: pysbd segments a Spanish clinical note wrongly under `en` without erroring, which shifts every window boundary. Mandatory is one rule with no hidden state. |
| D34 | **`ner/models/` builds architectures and nothing else.** No training, no prediction, no saving, no metrics. | NER-API's base model class was 661 lines doing all of that, so swapping an architecture meant subclassing the trainer. Split, the architecture is a function you can replace with your own. |
| D35 | Architectures are `"linear"` / `"crf"` / any callable `(base_model, label2id, id2label, **kwargs) -> model`, resolved by `build_model`, same shape as `Encoder`'s window strategies. | One extension mechanism for the library, not two. NER-API's "simple" is renamed `"linear"` — it describes the head. |
| D36 | The two `*Config` dataclasses (46 and 49 fields) are **not** ported. | Both were ~90% training and run plumbing with 3 genuinely model-level fields, now keyword arguments on the CRF constructor. |
| D37 | BIO constraint reasoning lives in a torch-free `ner/models/bio.py`. | Pure combinatorics over label strings, and the part most likely to be silently wrong; keeping it out of the tensor code makes it verifiable without torch. |
| D38 | **`train()` accepts a `transformers.TrainingArguments`.** `training_arguments(output_dir, **overrides)` builds one from a module-level `DEFAULTS` dict. | Battery = the helper's defaults; removable = build your own and pass it. Every `TrainingArguments` keyword works without us enumerating it. |
| D39 | The settings HF has no slot for stay explicit parameters on `train()`: `early_stopping_patience`, `pad_to_multiple_of`, `ignore_index`, `metrics_scope`, `save_model`, `track_resources`. | Six named arguments beat smuggling them into a config object HF would ignore. |
| D40 | `compute_metrics` is injected, not built. `trainer_class` defaults to `CRFTrainer` for a model exposing `decode_from_emissions`, `Trainer` otherwise. | Decouples training from evaluation. Duck-typing the CRF trainer means a custom architecture that can Viterbi-decode gets the right Trainer for free. |
| D41 | Discharged by D47. | |
| D42 | No `target_label` parameter on `train()`. Provenance goes through `run_metadata: dict`. | Nothing in training needs it once `compute_metrics` is injected and the model carries its own `label2id`. |
| D43 | Narrowed by D145. A single-label model keeps one `target_label` and three classes; a model with a head per type is the separate multihead path. D127 keeps the NER error analysis to one `target_label`. | User decision. Removes the pressure to design `2n+1` seams into evaluation that nothing would use. |
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
| D61 | **Inference is `ner/inference/predict.py::predict_entities`, task `ner.predict_entities`.** It takes a `model_dir`, not a run directory. `--random-state` raises a `TypeError` here. | Pointing at the model keeps inference independent of the orchestrator's artifact layout: a `fold_XX/best_model` and a directory copied to scratch load identically. Inference is deterministic, and D42 forbids an unused parameter kept to satisfy a flag. |
| D62 | **A saved model carries an `encoding.json` beside its weights**, written by `train(model_encoding=...)`. Output only — `train_model` never reads one. | The artifact describes itself, as HF's `config.json` does; it is also the only thing that makes a CRF checkpoint loadable. → notes |
| D63 | `Encoder(require_target_label=False)` skips the check that `target_label` occurs in the corpus. Default stays `True`. | Rule 3. The guard is right for training, where the typo it catches produces an all-`O` corpus; wrong for inference, where unannotated documents legitimately contain no entity. |
| D64 | **Per-span `score` and `text` are decoded in `ner/evaluation/spans.py`**, not again in `ner/inference/`. Both off by default. | Two implementations of B/I transition handling that must agree forever is exactly the duplication the migration exists to remove. |
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
| D81 | **`core/stats/`'s `compute_text_stats`/`compute_annotation_stats` take `base_model: str`, not a tokenizer, and `language` is required.** Amended by D87, which puts the `stats` switch on `prepare_dataset`. | These are the simplest-script entry point; `base_model` matches `train_model` (D66) and is honest about what is passed. `language` per D33. |
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
| D94 | Superseded by D112. | |
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
| D107 | **Only the encoder methods (`transformer_faiss`, `dense`) keep the gazetteer's representations on disk.** The lexical and sparse methods rebuild in memory as before. | Encoding the gazetteer is the cost, minutes on a GPU; refitting TF-IDF or hashing takes seconds. |
| D108 | **Persistence is on by default: `index_dir`, `<output_dir>/gazetteer_index` when `None`, is loaded when it holds an index and built and saved into when it does not; `save_index=False` turns the writing off.** `index_dir` is refused with the other methods; the manifest records `index: loaded \| built \| in_memory`. | One argument covers building, keeping and reusing, and the default keeps the run directory self-contained, as `search_hyperparameters` does. Reuse across runs is an explicit `index_dir`. |
| D109 | **What is kept is the raw embeddings (`embeddings.npy`), their vocabulary (`vocabulary.parquet`) and `index_manifest.json`, written last; the FAISS index is rebuilt on load.** `TransformerFaissRetriever.fit_faiss_from_embeddings` and `DenseRetriever`'s existing `vector_db` take them. | SQ and PQ indexes keep only lossy vectors, a GPU index cannot be written as is, and `dense` has no FAISS index at all. Raw vectors serve every `f_type` and both methods; rebuilding a flat index costs nothing next to encoding. |
| D110 | **An index records what its vectors depend on — method, `base_model`, a fingerprint of the gazetteer's `term`/`code` content, `n_terms`, and the encoding settings — and one that differs from the run raises, naming the fields.** A directory with files but no manifest is refused. FAISS settings are not recorded. | The strict default of `on_mismatch` and `on_conflict`: a silently reused index links against the wrong vectors. The fingerprint is over content because the gazetteer may be a frame. `batch_size` is recorded under `mean` pooling, which averages over padding (Q20). |
| D111 | **FAISS stays the search backend; usearch was not adopted.** | Retrieval accuracy must not drop, so search is exact, and exact search is fastest as a flat index on a GPU. usearch's strength is approximate HNSW search, and once the embeddings are kept, search is no longer the bottleneck. |
| D112 | **`lab.nel` is lab code, refactored to `docs/dev/CODE_STYLE.md`: classes and records renamed to what they are, compatibility aliases and duplicates removed.** Behaviour is unchanged except where a removal says otherwise; `link_entities` outputs, metric keys and the YAML method names are untouched, except that `whoosh` is gone. | The equivalence D94 protected is kept by `verify_nel`'s side-by-side checks, which now compare the renamed classes to the originals. → notes |
| D113 | **`seeds_per_trial` defaults to 3, down from 5.** | A trial's score must be a mean: on a 40-trial ELCardioCC sweep, seeds spread ±0.006–0.015 span F1, the size of the gaps between trials. Five seeds cut the noise on the mean to about 0.0045 against 0.0058 for three, at 5/3 the cost, and at 5 seeds a 40-trial sweep ran 55–80 GPU-hours. |
| D114 | **A word is a Unicode UAX #29 word, untailored, for every tokenizer: `core.split_into_words` defines the `word_id` the encoder labels by, and the words the text statistics count.** A token belongs to the word its start falls in; a token that is whitespace throughout belongs to none and is never labelled. Implemented with `regex`'s `WORD` flag, now a direct dependency. | The tokenizer's own words decided which entities could be learned: mdeberta's SentencePiece pre-tokenizer splits on whitespace alone and emits a bare `▁` ahead of a word, so on ELCardioCC 2026 31% of entities started on an unlabelled subword and 41% ended mid-word, in training and in the gold scored against. One standard gives every model the same words. → notes |
| D115 | **Before training, a built model runs one forward pass on CPU at `max_length`, and the run prints a line when it passes.** `lab.ner.models.check_max_length`; `train_model` runs it on each fold's model, `search_hyperparameters` on each base model at each of its `max_lengths` in the driver before Ray starts, under `report`. Inference does not: it encodes at the length the model was trained at. | A window longer than the model's positions otherwise surfaced as a CUDA device-side assert pointing at the wrong line (roberta-small-greek: `max_position_embeddings` 130, run at 256). The limit is not readable from the config without per-architecture knowledge — RoBERTa offsets by the padding index, DeBERTa is relative, rotary models declare 8k — and running the model needs none. The printed line is so the log shows the check was made. |
| D116 | **NER ensembling combines saved models given by path — independently trained, different backbones — in a new `lab.ner.ensemble` subpackage with two tasks: `ner.fit_ensemble` (members, and validation when tuning, to an ensemble directory) and `ner.predict_ensemble` (an ensemble directory over documents).** `predict_entities` is not overloaded to accept an ensemble. | Members are individual models that already predict on their own; nothing groups them into configurations, folds or seeds. A separate task keeps each signature honest: `strategy` and the like belong to one model and would be ambiguous across members. → notes |
| D117 | **Tuning data is the caller's: a gold-annotated `validation` input, in any form `read_documents` accepts.** The library does not check whether any member trained on it. | Which data to tune on, and whether the members saw it, is the researcher's call; the method takes models and data and records what it was given. |
| D118 | **A fit writes `ensemble.json` — the only file `predict_ensemble` reads: `format_version`, `method`, `votes`, `target_label`, `threshold`, and per member `model_dir`, `weight` and `weights_sha256` — plus `ensemble_manifest.json` (validation path, sha256 and counts, each member's `model`/`encoding` blocks, search settings, timings), `ensemble_trials.parquet` when it searched, and `ensemble_metrics.json` when it had validation.** `predict_ensemble` raises, naming the member, when a member's weights no longer match their hash. | The read-back artifact / written-only provenance split of `encoding.json` against `run_manifest.json`. A path alone would silently apply an ensemble to a model retrained in place. `ensemble_metrics.json` scores each member alone beside the ensemble, so whether it beats its best member is on paper. |
| D119 | **`write_predictions` writes the span table as both `predictions.tsv` and `predictions.parquet`**, for `predict_entities` and `predict_ensemble` alike. | Parquet keeps dtypes and text exactly; a TSV read back with pandas' default NA handling turns a surface form `NA`, `nan` or `null` into `NaN`. |
| D120 | **Members are combined on word tables: one row per labelled word, keyed `(doc_id, word_id)`, carrying the member's own `tag` and `p_O`/`p_B`/`p_I` by label name.** Members must share `target_label`, or the ensemble raises. A combined span's offsets are its words' `split_into_words` extents. Tables are built in memory, one member loaded at a time. | D114's `word_id` indexes the document's UAX #29 words, so it is the same for every tokenizer, window size and strategy; `label2id` order is not shared, so columns go by name. `split_into_words` extents do not depend on which member's tokenizer is read. → notes |
| D121 | **A CRF member's probabilities are its forward-backward marginals over the whole window, from `CRFForTokenClassification.marginals_from_emissions`, returned as log-marginals in float32; its `tag` is the Viterbi path.** The ensemble finds the method by duck-typing, as D40 does. | The CRF path returns Viterbi as one-hot pseudo-logits, so its softmax is 1.0 on the chosen tag and a CRF member would outvote every linear one. Log-marginals pass through softmax unchanged, so both architectures reach voting in one shape. |
| D122 | **Decoding is two-stage, per document, over the sorted `word_id`s present in any member's table. A word is in an entity when `Σ wₘ·eₘ > τ`, strictly, weights normalised over the members that labelled it; an entity word is `B` after a non-entity word, or when the combined B weight beats the I weight, ties continuing the entity.** `eₘ` is 1/0 from the member's `tag` for hard votes, `p_B + p_I` for soft. A span's `score` is the mean combined probability of its words' chosen tags, for hard votes too. | Valid BIO by construction, one code path for hard and soft, and τ means one thing — the share of weight that must say entity — which is what weighted voting tunes. A missing member is not evidence either way. → notes |
| D123 | **Methods are `majority` (hard votes, equal weights, `τ = 0.5`, so strictly more than half; ties to `O`), `soft` (probabilities, equal weights, `τ = 0.5`, or tuned with `tune_threshold=True`) and `weighted` (tuned weights and τ, `votes="soft"` by default or `"hard"`).** `votes` passed with another method raises. Validation is required when anything is tuned; otherwise it is optional and only scored. | `votes` is meaningless where the method fixes the vote (D42). An untuned ensemble given validation still gets the member-by-member comparison of D118. A strict `> 0.5` rather than `τ = (⌊n/2⌋+1)/n`: the share is renormalised over the members present (D122), and a fixed `3/4` for four members would refuse two of the three present at a word. |
| D124 | **Tuning is Optuna TPE: one raw weight per member in [0, 1], normalised, and τ in [0.05, 0.95] (τ alone for `soft`), `n_trials=200`, maximising `metric="span_strict_f1"` — any `span_metrics` key — with `random_state` seeding the sampler.** Ties go to the first best trial. | One evaluation is a weighted sum, a decode and a score on CPU, so hundreds of trials are cheap; a grid explodes with members (about 19,000 points at five, step 0.1). `span_strict_f1` is what model selection already tracks (D44). |
| D125 | **No per-member calibration in the first version.** | Tuned weights and τ absorb most of it; it would make plain `soft` need validation, and a CRF needs its emissions and transitions scaled, not its marginals. It can be added later as a per-member field of `ensemble.json`. |
| D126 | **A member trained with a callable window strategy is refused by `fit_ensemble`, naming it.** | `encoding.json` cannot name a callable, and one `strategy` argument cannot say which member it is for. |
| D127 | **NER error analysis is of one `target_label`: gold and training are restricted to it, and predictions carrying another label raise.** Its error taxonomy is `CORRECT`, `MISSED`, `SPURIOUS` and `BOUNDARY_ERROR`; the label errors, confusion matrices and label-shift tables of the analysis branch are not ported. Pairing an error stays label-blind, so label errors return with multi-label models in the one function that names a pair. Extended by D141 with `UNCREDITED_EXACT`. | Inferring multi-label from the labels present switched it on whenever a single-label model was scored against a multi-label corpus, counting every entity of the other labels as missed. Nothing produces multi-label predictions yet to build or verify the machinery against. |
| D128 | **`split_into_sentences` uses pysbd for the 23 languages pysbd has, Punkt for the nine European ones it lacks (cs, et, fi, ml, no, pt, sl, sv, tr), and ICU's sentence break rules for every other ISO 639-1 code. Any other `language` raises.** The table is fixed in `core.sentences`, never read from what is installed. Words stay UAX #29 (D114). | Each language gets the most specific segmenter available, and ICU leaves no ISO 639-1 language without one: Catalan, Basque and Galician are in neither pysbd nor Punkt. Without the ISO 639-1 list a typo such as `"sp"` would reach ICU's root rules without error, the silent guess D33 rules out. → notes |
| D129 | **Python floor `>=3.11`.** Supersedes D17. | `icu4py`, the ICU binding D128 uses, requires 3.11. |
| D130 | **The Punkt models of D128's nine languages ship inside `lab`, under `lab/core/data/punkt_tab`, read with nltk's own decoder. nltk is imported only when a Punkt language is segmented.** | nltk downloads its models at runtime, which offline nodes cannot do, and nltk 3.10 refuses to open model files outside its own data roots. Importing nltk at module load added about a second to every `import lab.core`. |
| D131 | **Every segmenter's spans take pysbd's shape: contiguous from the first non-whitespace character to the end, trailing whitespace on the sentence before it, none for a document that is whitespace throughout. The segmenter and its version are recorded as `sentence_segmenter` in `encoding.json` and as two `text_stats` columns.** `encoder_from_description` ignores the recorded value and picks again from `language`. | Windowing and the statistics must not depend on which segmenter ran. Which one ran moves every window boundary, so it is recorded; ensemble members need not agree on it, since `word_id` does not depend on sentences (D120). |
| D132 | **NER error analysis does not audit its inputs.** The analysis branch's per-span annotation audit, its tables and event columns, and the document-text parameters it needed are not ported; the manifest records the input counts, duplicate gold and prediction rows included. A paired boundary error keeps its whitespace and punctuation flags. | Span text from a corpus or a prediction is the document's own slice, and a disagreeing source is settled at corpus build (D79, D80), so the surface checks cannot fire on the library's own inputs. What can — duplicated or whitespace-edged gold — describes the corpus, the same for every model analysed against it. |
| D133 | **Lexical similarity in the NER error analysis is normalized Levenshtein, `1 − edits / max(len)`, on NFKC-normalized, casefolded, whitespace-collapsed text; `similarity_threshold=0.80` separates `LEXICALLY_SIMILAR` from `NOVEL`.** The branch's `hybrid` mode, `min(Levenshtein, Jaro–Winkler)`, is not ported. | On 3,440 clinical-style pairs the minimum changed the class at 0.80 for 11 (0.3%): Jaro–Winkler rewards a shared prefix and is almost always the higher of the two. One measure states in one clause, and the threshold is named for what it thresholds. |
| D134 | **The NER error analysis measures variability two ways: a document bootstrap (`bootstrap_samples=2000`, `confidence=0.95`, percentile intervals) for uncertainty, and per-document scores for heterogeneity.** The branch's non-overlapping document partitions are not ported. Amended by D142. | A partition's spread is a subsample's, set mostly by the chosen `partition_size` and drawn from one shuffle, so it restates the bootstrap with a knob that moves the answer; whether a few documents dominate shows directly, and by name, in the per-document table. |
| D135 | **The NER error analysis reads predictions with a new `ner.inference.io.read_predictions`, the inverse of `write_predictions`, or takes the span frame itself; gold is `reference` and training is read the same way, a corpus or an annotation TSV, through `read_reference`/`resolve_gold` restricted to `target_label`.** The branch's `core.read_spans`, `validate_spans` and `SPAN_ALIASES` are not ported. | Predictions were the one thing the library wrote and never read back. An alias table for no particular source format gives the boundary two names for each column (D27); `CODE_SPAN_ALIASES` exists for one external format. |
| D136 | **NER error analysis is `ner/analysis`'s `analyze_predictions`, task `ner.analyze_predictions`, run through `lab run` like every task; it returns an `ErrorAnalysis` holding the metrics, the event table and every table in memory, and writes `events.parquet`, `tables/`, `figures/` (`figures=True`) and `analysis_manifest.json` into `output_dir`.** `target_label` is required; `output_dir` is created and rewritten without a guard, as inference's is. `write_figures(output_dir)` redraws the figures from the written tables. The manifest holds inputs, parameters, metrics and timing only. | The branch's per-feature Typer commands reopen the flag surface D11 and D86 closed. A rerun costs minutes, not GPU-hours, so `claim_run_dir` is not needed. Definitions and citations are documentation, and plot styling belongs to the plotting code; neither describes a run. → notes |
| D137 | **The NER error analysis is part of `lab[ner]`: `rapidfuzz`, `scipy` and `matplotlib` join the `ner` extra and `ner/tasks.py`'s `REQUIRES`.** No `lab[analysis]` extra; the span-table readers stay in `ner/inference/io.py`, whose package imports torch. | Predictions to analyse come from `predict_entities` or `predict_ensemble`, so the NER environment is already there. A torch-free analysis would move the readers for a use nobody has yet (D84). |
| D138 | **A mention's length in the NER error analysis is its count of UAX #29 words, `len(split_into_words(text))`, punctuation included.** | D114's word is what the encoder labels, so the count is the number of tagging decisions a strict match needs; the branch's whitespace count called `anti-TNF` one word where the model tags three. |
| D139 | **NER error analysis reports a subgroup on the side that defines it: gold mentions by their own class, with support, strict and character recall and the outcome mix; predictions by their own surface's class, with strict and character precision. No per-class F1.** Overlap and nesting are gold properties only. Per-class recall is counted from the event table, not by re-scoring filtered frames; overall P/R/F1 is `predict_entities`' own, and the bootstrap covers it and per-class recall. | Filtering gold and predictions each by its own class and scoring the pair mixed two populations: a prediction one word too long on a seen mention counted against `SEEN` recall and `NOVEL` precision at once, and prediction-side overlap meant one prediction overlapping another. Whether a model generalises is a question about gold mentions. → notes |
| D140 | **A mention is `SEEN` when its normalized form (D133) occurs among the training mentions of `target_label`; training frequency counts normalized forms too, so `SEEN` holds exactly when similarity is 1.0.** A `seen_verbatim` flag keeps the case-sensitive view. | The branch took `SEEN` from the raw string and similarity from the normalized one, so `hta` against a training `HTA` was `LEXICALLY_SIMILAR` in the `EXACT` bin with frequency 0. Raw similarity is no fix: case-sensitive Levenshtein puts `HTA` and `hta` at 0. |
| D141 | **After strict matching, the NER error analysis pairs each document's leftover gold spans and predictions in three steps: identical spans become `UNCREDITED_EXACT`; two or more predictions each inside one gold span (fragmentation), or two or more gold spans inside one prediction (merging), whose union reaches `overlap_threshold` in character Dice, become one `BOUNDARY_ERROR` unit; the rest are assigned one to one by `scipy.optimize.linear_sum_assignment` maximising Dice, pairs below the threshold dropped.** `overlap_threshold=0.60`, with sensitivity over 0.40–0.80. Strict counts are never changed. Extended by D144. | nervaluate credits a gold span to the first overlapping prediction in a document, so an exact prediction listed after one is spurious; the branch filed that as a miss and a spurious prediction. Its weight's other terms were dead under D127 or a function of Dice, and its fragmentation needed every fragment below the threshold, so `anti` + `TNF` against `anti-TNF` was a boundary error plus a stray. → notes |
| D142 | **The document bootstrap gives an interval for every gold-subgroup recall and every prediction-subgroup precision, not only overall P/R/F1 and per-class recall.** Amends D134. | One documents × counts array per subgroup value costs almost nothing, and an interval shows a thin bin for what it is, where the branch's hollow marker below n = 10 was an arbitrary cut. |
| D143 | **The NER error analysis draws 14 figures, each answering one question, from `ner/analysis/figures/`, with matplotlib's object-oriented API: `error_taxonomy`, `exposure`, `predicted_exposure`, `mention_properties`, `mention_traits`, `boundary_kinds`, `boundary_offsets`, `overlap_sensitivity`, `oracle_error_budget`, `error_surfaces`, `documents`, `confidence_by_outcome`, `calibration`, `confidence_sweep`.** A figure with nothing to show is skipped; the manifest lists those written. `figure_format` is `"svg"` (default), `"pdf"` or `"png"`. Okabe–Ito colours are fixed per outcome. No plot manifest, caption file or prose inside a figure. Amended by D144. | The branch's figures mostly drew tables D127, D134 and D139 removed, so the set was redesigned from the questions a reader asks rather than ported. Reading guidance belongs in the documentation and a paper's caption; `matplotlib.use` at import changed global state for every importer. → notes |
| D144 | **A one-to-one boundary error also records its `matched_edge`, the edge it got right: `START`, `END` or `NEITHER` (shifted errors are always `NEITHER`; fragmentations and merges have none). `boundary_summary` has a row per kind and matched edge; `boundary_kinds` draws one bar per pair, grouped by kind; `boundary_offsets` is a 2D grid of start against end offset instead of two histograms.** Extends D141, amends D143. | `TOO_LONG` and `TOO_SHORT` each held three different errors (one edge wrong at either side, or both), and the edge a model gets wrong points at different causes: trailing modifiers and prepositional phrases at the end, determiners and preceding adjectives at the start. Two separate offset histograms could not show that a correct start goes with a wrong end. → notes |
| D145 | **Narrows D43: a multihead model — one shared Transformer, one independent O/B/I head per entity type — lives in its own `lab.ner.multihead` package with its own tasks, `ner.train_multihead`, `ner.search_multihead` and `ner.predict_multihead`.** Single-label tasks and signatures are unchanged, and shared code carries no multihead branch: it is reached through what it already injects (`compute_metrics`, and the trial pieces of D149), and batched by the collator of D151, which pads multihead labels like any others. → notes | Cross-type overlapping annotations need independent decisions a flat softmax cannot make. A mode signalled inside shared orchestrators turns every entry point into a branch list. |
| D146 | **The head catalogue is `entity_labels`, in head order; left out, it is every type annotated in the split's train and validation partitions, sorted, the same for every fold. A listed type annotated nowhere there raises; a corpus type with no head is left unlabelled and warns with counts. `overlap_policy` resolves overlaps within each type with the single-label policies and default, never across types.** | Folds that train different heads cannot be averaged. Per type, a head keeps exactly the entities a single-label model for that type keeps, so the two are comparable. |
| D147 | **Each head is scored with nervaluate on its own type's spans, exactly as a single-label model is; micro sums the heads' counts (`sum_span_results`, how nervaluate builds its own overall), macro averages the heads, and keys are `head_<type>_*`, plain (micro) and `span_<scenario>_macro_*`. The loss averages each head's cross-entropy, then the heads with labelled positions; selection defaults to `span_strict_macro_f1`.** | Keeps all four scenarios and `min_overlap_percentage`, and makes per-head numbers directly comparable with specialists. Every type weighs the same in both the loss and selection. |
| D148 | **A saved multihead model's `encoding.json` records the backbone's full configuration as `backbone_config`; loading builds the backbone from it with no pretrained weights and loads the fine-tuned weights strictly. A digest of the head catalogue travels in the weights and must match.** | The directory rebuilds without the base model it was trained from. A reordered catalogue would otherwise load silently with heads swapped. |
| D149 | **An `EncodedVariant` carries what its trials need — `build_model` and `compute_metrics` — decided by whoever encodes it; the model-independent sweep (setup, length check, Ray, smoke test, Optuna, summary, winner, report) lives in `lab.ner.hpo.sweep`. `search_multihead`'s winner is a `ner.train_multihead` config, recorded in `hpo_summary.json` under `train_multihead`.** | One sweep machine for every kind of model, with no architecture branch in a trial. Ray starts only after the length check, as before. |
| D150 | **`train` sets `model_accepts_loss_kwargs = False` on every Trainer it builds, so the loss is always divided by `gradient_accumulation_steps`.** → notes | Under transformers 5, a `forward` taking `**kwargs` — the stock token-classification heads and `CRFForTokenClassification` alike — makes the Trainer skip that division, assuming the model normalises by `num_items_in_batch`; none here does. |
| D151 | **Every batch is padded on the right by `TokenClassificationCollator`, whatever the tokenizer's `padding_side`, for `train`, `predict_logits` and so every task; it pads labels of one id per position and of one per head alike. It replaces `DataCollatorForTokenClassification`.** → notes | Span reconstruction and token diagnostics read each row's predictions from position 0; a left-padding tokenizer shifted every shorter row's logits past them, silently. |

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
`core/stats/` — is paid.

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

### D112 — `lab.nel` refactored

Renamed, old names removed: `MentionAnnotation` → `Mention`; `MatchCandidate` → `Candidate`, its
`candidate_term` → `term` (so `EntityLinkingPipeline.write_outputs` writes `term` in
`candidates_json`); `BaseEntityMatcher` → `LexicalMatcher`; `StringMatchMatcher` →
`ExactMatcher`; `MatrixBiEncoder` → `SparseRetriever`, with `BaseBiEncoder` folded into it;
`FaissBiEncoder` → `SparseFaissRetriever`; `HerbertFaissBiEncoder` →
`TransformerFaissRetriever`; `BaseDataFrameLexicalRetriever` → `DataFrameLexicalRetriever`;
`EntityLinkingCrossEncoder` merged into `CrossEncoderReranker`; the `load_*` readers →
`read_mentions_tsv`, `read_gazetteer_entries`, `read_concepts_tsv`, `read_hierarchy_tsv`,
`read_text_dir`, `read_brat_dir`, `read_brat_ann`; `build_snomed_graph`, `load_snomed_graph`,
`load_snomed_graph_pickle` → `build_ontology_graph`, `read_ontology_graph`,
`read_ontology_graph_pickle`; `snomed_graph_distance_and_direction` →
`graph_distance_and_direction`; `load_embeddings`/`save_embeddings` → `read_embeddings`/
`write_embeddings`; `EntityLinkingPipeline.save_outputs` → `write_outputs`. Outside `nel`, the
CLI's `load_config` → `read_config`: files are read and written with `read_*`/`write_*`, models
loaded and saved with `load_*`/`save_*`.

Removed: `WhooshContextMatcher` and the `whoosh` method, an alias of `levenshtein`;
`SimpleCrossEncoder`; the aliases `load_ontology_graph_pickle` (the old one),
`evaluate_predictions` and `fit_with_progress_no_loss_logging`; `ontology_distance_and_direction`,
whose `non_rel` label `ontology_summary_from_codes` now produces itself; `calculate_recall_at_k`,
`first_code` and `_parse_codes`, duplicates of `retrieval_metrics_from_codes` and
`parse_code_list`; the `train(df_hard_triplets, ...)` overload, so `CrossEncoderReranker.train`
is PyTorch's `train(mode)` again and triplets go through `train_hard_triplets`; the dead
`triplets.generator` import in `rerank_candidates`; the guards for scikit-learn, torch,
sentence-transformers and pyarrow, which `lab[nel]` and the core install require.

What breaks: code importing any old name, or the removed modules `nel/cross_encoder.py`,
`nel/retrieval/{base,matrix,faiss,sentence_transformer}.py`; a config with `method: whoosh`. Two
behaviour changes follow from the removals. `graph_metrics_for_dataframe` reads candidate lists
with `parse_code_list`, so a Python-literal list string is parsed rather than raising, and empty
or null codes are skipped. `SparseRetriever` and `SparseFaissRetriever` now carry a `method`, so
RRF names them `matrix_biencoder` and `faiss_biencoder` instead of by class name.

New modules: `matching/bm25.py` (the index and score both BM25s shared), `matching/dataframe.py`
(the DataFrame retrievers, out of `lexical.py`), `retrieval/faiss_index.py` (index construction,
out of the transformer retriever), `evaluation/graphs.py`, and the `cross_encoder/` package
(`examples`, `training`, `reranker`).

### D114 — UAX #29 words

`encoded.word_ids()` is the tokenizer's pre-tokenizer, and `build_row` labels only a word's
first subword. Measured over the 11,979 ELCardioCC 2026 train entities, the share starting /
ending mid-word with the tokenizer's words was mdeberta-v3-base 31.2% / 40.9%, EuroBERT-210m
6.0% / 0.6%, roberta-base-biomedical-clinical-es 1.3% / 0.8%, ModernBERT-base 1.2% / 0.7%,
bert-base-greek-uncased 0.4% / 0.4%. With UAX #29 words every one falls to 1.5–2.3% / 0.7–1.1%.
Evaluation rebuilds gold from the rows, so the damage did not show in the score.

`regex`'s `WORD` flag was compared with ICU 78.3's word break iterator (root locale) over
the 4,000 ELCardioCC documents and gave identical boundaries. ICU itself was not adopted:
PyICU builds against the system ICU, which offline cluster nodes do not reliably have, and
`regex` is already a transformers dependency.

The standard is used untailored. Its `MidLetter` rule joins letters across a colon, so
`Ιστορικό:ΧΚΜ` is one word and an entity written straight after a colon cannot be learned
(115 ELCardioCC entities, about 1%); ICU's root locale does the same. A colon break was
rejected as a lab-specific deviation. A further ~1.4% of that corpus's entities start inside
a word because their annotated offsets are shifted, which no word definition recovers.

What changes: rows of every tokenizer, wherever the tokenizer's words and UAX #29's differ
(`12.5` and `can't` become one word, punctuation always its own); whitespace-only tokens,
such as SentencePiece's bare `▁` or a byte-level token for a repeated space, now get
`IGNORE_INDEX` instead of a label; the statistics' `words`, `vocabulary_words` and
`mattr_words`, which were `\w+` runs. NER-API equivalence in `verify_encoder` no longer
compares `labels` and `word_ids`: the windows, ids and offsets must still be identical, and
the labels must agree wherever both put one.

### D116 — ensembling

The scope is models that are each good on their own, on different backbones: ensembling
fold models or seed replicas of one configuration was considered and is not a use case, so
nothing groups members or averages within one. The flow is `fit_ensemble` on a validation
set, then `predict_ensemble` on the documents to score or label; `predict_ensemble` writes
what `predict_entities` writes, plus the ensemble and each member's hash in its manifest.

Layout: `ensemble/tables.py` (a member's word table, the CRF marginals path included),
`ensemble/voting.py` (weighted combination and decoding), `ensemble/search.py` (the Optuna
fit), `ensemble/io.py` (`ensemble.json`, hash checks), `ensemble/fit.py` and
`ensemble/predict.py` (the tasks). `marginals_from_emissions` lives in `models/crf.py`.

### D120 / D122 — word tables and decoding

A member labels only the first subword of a word, inside a window's core, so it gives at most
one row per word. It gives none when no token starts in the word — a token straddling two
UAX #29 words, such as a SentencePiece `-b` starting on a `-` that belongs to no word — so
which words go unlabelled depends on the tokenizer. Such a word is decided by the members
that did label it. A word no member labelled has no tag; decoding skips it rather than
breaking on it, and a span around it still covers it through its words' extents, as
`bio_to_spans` covers unlabelled positions.

`split_into_words` extents differ from `expand_to_word_extent`'s only where a token runs past
its word (`mg.` covering the `.`); there the ensemble's span is the word's and does not
depend on which member's tokenizer is read.

Constrained Viterbi over averaged probabilities was the alternative. It can repair an orphan
`I` to `O` where the two-stage rule makes it `B`, but has no natural threshold: τ would be a
bias added to `O`. A tuned τ should keep weakly supported words out of stage one anyway.

The equivalence a one-member ensemble owes `predict_entities` holds for hard votes, which
reproduce `bio_to_spans` orphan-`I` handling included, with two exceptions: a CRF member
under soft votes decodes from its marginals, not its Viterbi path; and decoding per document
lets a span continue across a window boundary when the next window's first word votes `I`,
where `bio_to_spans` decodes per window. Windows end at sentence boundaries, so the second
arises only inside a sentence too long for one window.

### D128 — sentence segmenters

pysbd and nltk were also considered for word splitting, and neither has a language-specific
one: pysbd only splits sentences, and nltk's `word_tokenize` uses `language` for Punkt alone,
its word step being the English Treebank tokenizer, which splits `can't` into `ca` + `n't` and
rewrites quotes, losing offsets. Words therefore stay as D114 defines them.

ICU comes from `icu4py`, not PyICU: its wheels embed ICU 78.3, the version D114 compared
`regex` against, so the system-ICU problem that ruled PyICU out does not arise. Its
`segments()` offsets are UTF-16 code units, which are converted to string indices; a
character outside the Basic Multilingual Plane would otherwise shift every later offset.

### D139 — subgroups on the side that defines them

The branch scored a class by filtering gold to gold mentions of that class, predictions to
predictions whose own surface fell in it, and running nervaluate on the pair. Recall then
counted gold of the class and precision counted predictions of the class, so the F1 joined
two populations, and an error whose two sides fell in different classes was charged to both:
gold `insuficiencia cardiaca` (seen) predicted as `insuficiencia cardiaca aguda` (novel) is a
miss in `SEEN` and a false positive in `NOVEL`. Structural features changed meaning between
the sides — computed on the predictions, "overlapping" meant one prediction overlapping
another.

Precision by predicted surface is kept because it answers a question of its own: when the
model predicts a surface it saw in training, how often is it right. It is reported beside
recall, never combined with it.

### D141 — pairing leftover spans

Checked against nervaluate 1.2.1: with gold `[0, 8)` and predictions `[0, 3)` then `[0, 8)` in
that order, strict is one incorrect and one spurious, no correct; reversed, one correct.
`span_dataframe` sorts by `(start, end)`, so a shorter overlapping prediction always comes
first. Strict counts must equal `predict_entities`' (D97), so the analysis names the artifact
`UNCREDITED_EXACT` rather than correcting it.

The branch's assignment weight was `1e6·same_boundary + 1e5·Dice + 1e4·IoU + 1e3·same_label +
100/(1+distance) + confidence`. Under one label a same-boundary pair is a strict match (or the
artifact above), the label term is constant, and IoU is `D/(2−D)`; summed over a matching,
the magnitudes did not keep the terms ranked anyway.

Containment is what makes fragmentation and merging unambiguous: the model found the entity
and split it, or found several and joined them. A prediction reaching outside its gold span
is an ordinary boundary error. Fixing a fragmentation in the oracle budget removes its k
false positives and adds one true positive; fixing a merge removes one false positive and
adds k true positives.

### D136 — the analysis package

Layout: `analysis/events.py` (strict events from nervaluate's indices, checked against
`span_metrics`), `pairing.py` (D141), `exposure.py` (D133, D140), `features.py` (a mention's
surface features and D138's word count; overlap and nesting among gold), `subgroups.py`
(D139), `errors.py` (taxonomy, boundary details and summary, threshold sensitivity, oracle
budget, recurrent error surfaces), `variability.py` (D134), `confidence.py`, `figures/`, and
`analyze.py` (the task, `ErrorAnalysis`, `write_figures`).

Tables: `gold_mentions`, `predicted_mentions` and `units`, the row-level tables the others
summarise; `gold_subgroups` and `predicted_subgroups`, one row per family and value, replacing
the branch's seven `metrics_by_*` tables and its two gold-conditioned exposure tables; then
`error_taxonomy`, `boundary_errors`, `boundary_summary`, `overlap_sensitivity`,
`oracle_error_budget`, `error_surfaces`, `documents`, `bootstrap_intervals`, and with a varying
score `confidence_by_outcome`, `calibration` and `confidence_sweep`. Overall metrics are in the manifest and the
result, as for `predict_entities`. `confidence_summary` and `error_intersections` are not
ported: the first only told plotting whether scores vary, and two of the second's five flags
are gone (D132, D139).

### D143 — the figures

Each figure, the question it answers and its form: `error_taxonomy` (what errors: sorted
bars of diagnostic units), `exposure` (does it find unseen mentions: recall with interval
beside the 100% outcome mix per class), `predicted_exposure` (precision by predicted-surface
class), `mention_properties` (small multiples of recall over training frequency, similarity
bin, word count and casing), `mention_traits` (dumbbells of recall without and with each
trait: acronym-like, digit, punctuation, parentheses, hyphen, slash, Greek letter,
overlapping, nested), `boundary_kinds` (too long, too short, shifted, fragmented, merged,
split by matched edge since D144), `boundary_offsets` (start against end offset, tails folded
at ±10, a 2D grid since D144), `overlap_sensitivity`
(unit counts across Dice thresholds, the chosen one marked), `oracle_error_budget` (observed
to oracle F1 per scenario), `error_surfaces` (top 15 surfaces stacked by error type),
`documents` (empirical CDF of per-document strict and character F1), `confidence_by_outcome`
(score CDF per outcome), `calibration` (reliability diagram), `confidence_sweep` (P/R/F1 and
retained share against threshold, two panels). The confidence figures need a varying `score`;
`documents` needs two documents; the boundary figures need a boundary error.

Okabe–Ito passed the categorical checks (CVD and normal-vision separation of adjacent
colours); its lighter colours fall under 3:1 against white, which direct labels and the
parquet tables relieve. Text stays text in SVG (`svg.fonttype: none`) so it can be edited;
widths are 3.5 in and 7.2 in, for one and two paper columns. Every trait is kept even where
a corpus gives it little support: the interval says so, and the user drops what they do not
need.

### D144 — the matched edge

`matched_edge` is orthogonal to `boundary_kind`: the kind says the direction (the prediction
contains the gold, lies inside it, or is displaced), the edge says which side went wrong.
Kept as two columns rather than eight kinds so a table can group by either. Both edges
cannot match, since that pair is exact.

`boundary_kinds` gives each kind and edge its own bar in the one boundary-error orange, the
meaning in the labels ("too long, past the end", "too short, starts late"). Stacking the
edges within a kind's bar needed three colours: tints of the orange failed the categorical
palette checks (the lightest read as grey and sat too close to the mid tint), and other hues
would collide with the outcome colours every figure shares.

`boundary_offsets` bins start offset (x) against end offset (y), both folded at ±10, coloured
on a log scale in one orange ramp. The vertical line through zero holds the errors that got
the start right, the horizontal one those that got the end right, and the four quadrants
are too long, too short, shifted left and shifted right.

### D145 — multihead as its own path

The first implementation, on Fernando's `work` branch, added multihead inside the
single-label path: `target_label=None` meant multihead in the encoder and scoring,
`architecture == "multihead"` in the orchestrators, `hasattr(model, "entity_labels")` in
the collator and trainer, and `DataFrame.attrs` carried the gold and the catalogue between
them — about thirty branch points, a hundred by the end of that branch. Every shared entry
point stopped reading as a list of steps, and `train_model` loosened `target_label` and
`language` to optional parameters that raised when left out.

Here multihead has its own encoder, model, collator, scoring, orchestrators and loader, and
reuses the shared pieces unchanged: windowing and tagging (extracted from `Encoder` as
`window_document`, `label_window`, `build_label_ids`, `resolve_window_strategy`), span
reconstruction and token diagnostics on a per-head view of the rows, `train`, the run and
fold helpers, and the sweep. What a multihead run needs that a single-label one does not
enters as an argument, never as a branch.

### D150 — the loss under gradient accumulation

With accumulation over N micro-batches the Trainer sums their gradients and, to average
them, divides each loss by N (`Trainer.training_step`); Accelerate is configured with
`num_steps=1`, so nothing else divides. It skips the division when it believes the model
normalises its own loss by `num_items_in_batch`, which it infers from a `forward` taking
`**kwargs` unless the model sets `accepts_loss_kwargs`. Under transformers 5.14 the BERT,
RoBERTa and DeBERTa-v2 token-classification heads take `**kwargs` and compute a plain mean
cross-entropy, as does `CRFForTokenClassification`, and none sets the attribute.

The accumulated gradient was therefore N times the average. AdamW's update is nearly
invariant to a constant gradient scale, but `max_grad_norm=1.0` clips the scaled gradient,
so clipping fired more often and rescaled each step by a varying factor, and the logged
`training_loss` was N times too large. Default sweeps (effective batches 16/32/64 under a
64 micro-batch cap) never accumulate; a lowered `max_micro_batch_size` or an explicit
`gradient_accumulation_steps` did, and configurations that needed accumulation were then not
trained like those that did not. The Trainer's own `compute_loss` docstring prescribes this
override for a loss that ignores `num_items_in_batch`.

### D151 — padding on the right

`DataCollatorForTokenClassification` pads each batch on the tokenizer's `padding_side`, and
the model's logits come back in that padded layout. `predicted_spans` takes a row's first
`len(labels)` predictions and reads them against the unpadded offsets and word ids the row
stores; `token_diagnostics` pads the gold on the right. With `padding_side="left"` every row
shorter than its batch's longest was compared position by position against padding or
earlier tokens: wrong spans, wrong metrics, wrong best-epoch selection, wrong ensemble word
tables, and nothing raised. BERT, RoBERTa, DeBERTa and XLM-R tokenizers pad right by
default, so no run had hit it; many decoder-family tokenizers pad left.

Padding right in the collator fixes the cause on every path, including a tokenizer handed to
`train` or `predict_logits` directly, where forcing `padding_side` at load time would not
reach. Refusing a left-padding tokenizer would leave a YAML user no way to run one. For a
right-padding tokenizer the batches are what the stock collator built: the same `-100` label
padding and `pad_to_multiple_of`.
