# Verification

Behavioural checks for `lab`, kept outside `src/` so they are never packaged or installed.
They are plain scripts, not a test suite — run them directly, read the output.

```bash
uv venv .venv-verify --python 3.12
uv pip install --python .venv-verify pandas pyarrow pyyaml pysbd nltk icu4py transformers datasets accelerate pytorch-crf nervaluate "ray[tune]" optuna
uv pip install --python .venv-verify scikit-learn networkx faiss-cpu sentence-transformers rapidfuzz
uv pip install --python .venv-verify torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-verify --no-deps -e .

.venv-verify/bin/python verification/run_all.py
```

`torch` is the CPU build, and `faiss-cpu` stands in for the `faiss-gpu` the `nel` extra
prefers on Linux. `lab.core` never imports torch — `verify_layering.py` checks that — but
`lab.ner.models`, `lab.ner.training`, `lab.ner.hpo` and `lab.nel`'s encoder retrievers do,
and a CPU wheel is enough:
the end-to-end training checks run against a randomly initialized miniature BERT written to
a temporary directory, so nothing is downloaded. Everything but `verify_hpo.py` runs in
seconds; that one starts a local Ray instance for its mini-sweep and adds a minute or two.

| Script | Covers |
|---|---|
| `verify_layering.py` | the one-way import rule: `core` imports no task subpackage, task subpackages never import each other, `core` and the CLI import no torch |
| `verify_data.py` | labels, BRAT reading, the canonical schema, stratification, splitting, `prepare_dataset` |
| `verify_encoding.py` | overlap policies, segmentation, sentence segmenter routing (pysbd, Punkt, ICU), UAX #29 words, windowing, IOB2 tagging, row assembly |
| `verify_encoder.py` | `Encoder` construction and encoding, custom strategies, NER-API equivalence |
| `verify_models.py` | BIO constraint masks, the architecture registry, custom factories, the `max_length` forward-pass check |
| `verify_training.py` | row conversion, `training_arguments`, and linear + CRF training end to end |
| `verify_evaluation.py` | span reconstruction, nervaluate scoring, character metrics, token metrics |
| `verify_assessment.py` | `train_model`: fold rotation, both split modes, manifests, aggregation |
| `verify_hpo.py` | search-space forms, variants, trial scoring, OOM handling, a real two-trial Ray sweep, the winner block re-run through `train_model` |
| `verify_inference.py` | span decoding, `predict_entities` end to end, scoring against gold and against a reference |
| `verify_ensemble.py` | voting and two-stage decoding, one-member equivalence with `predict_entities`, CRF marginals against brute-force path enumeration, `ensemble.json` and weight digests, `fit_ensemble` and `predict_ensemble` end to end |
| `verify_nel.py` | nlp4bia-linking's own tests, side-by-side equivalence with it, `link_entities` end to end on every method, with a hierarchy, with the reranker |

Everything runs on synthetic fixtures built by `fixtures.py`, so the scripts pass on a
machine with no corpora at all. Two groups of checks additionally use real data and skip
cleanly when it is missing:

- **Real corpora** — `NER_LAB_SAMPLES`, defaulting to `~/bsc/NER-API/data_samples`.
- **NER-API equivalence** — `NER_API_ROOT`, defaulting to `~/bsc/NER-API`. Runs the old
  `DataLoader` and the new `Encoder` over the same 60 documents for both window strategies,
  and asserts the resulting frames are identical. Byte-level tokenizers are skipped by
  design: `tokenize_document` trims the leading space out of each token's offset and NER-API
  does not, so sentence assignment and windowing legitimately differ.
- **nlp4bia-linking equivalence** — `NLP4BIA_LINKING_ROOT`, defaulting to
  `~/bsc/nlp4bia-linking`. Runs every matcher, the sparse retrievers, RRF, the metrics, the
  graph distances and the readers from both packages over the same synthetic gazetteer and
  mentions, and asserts the candidates are identical.

A skip is printed, never silently passed. Check the summary line for a skip count before
trusting a green run.

## Multi-Head, fully local fixtures

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_multihead.py
.venv-verify/bin/python verification/benchmark_multihead.py --repeats 1000
```

`verify_multihead.py` is registered in `run_all.py`. It builds a real local WordPiece
tokenizer and random tiny BERT/DistilBERT, exercises joint learning, overlap, strict metrics,
Trainer/CLI, both documented YAMLs, checkpoint round trips, malformed checkpoints,
linear/CRF regression paths, fold discovery and a real two-trial/two-seed Ray sweep.
CUDA skips explicitly when absent. It needs no Hugging Face model download; some older
scripts still fetch BERT/GPT-2 **tokenizers**, even though their model weights are local.
The benchmark runs separate processes and reports forward counts, model-only elapsed time,
throughput, parameters and peak process RSS; it does not measure end-to-end application time.
See [MULTIHEAD_RESULTS.md](MULTIHEAD_RESULTS.md) for executed results and remaining blockers.

## Unified ensembles

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_ensemble.py
.venv-verify/bin/python verification/benchmark_ensemble.py --repeats 50
```

The original ensemble suite now uses a shared local WordPiece tokenizer and still trains
its tiny legacy models through the existing training orchestrator. It also calls
`verify_unified_ensemble.py`, separated for heterogeneous catalogue/coverage combinatorics,
all native member families, hybrid A–F, scoped optimization, exact span methods, old/new
artifacts, calibration provenance, all six documented YAMLs and composition with NEL.
The benchmark separates member pipeline, batched inference, alignment, combination,
decoding and Optuna, with one process per scenario and equal padded length.
[UNIFIED_ENSEMBLE_RESULTS.md](UNIFIED_ENSEMBLE_RESULTS.md) records actual results and limits;
these tests do not require network downloads or imply GPU/clinical-model validation.


Constrained CRF verification (local tiny Transformer; no model download):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python verification/verify_chain.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python verification/verify_word_crf.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python verification/benchmark_crf.py --output /tmp/crf.json
```

`verify_chain` enumerates all legal BIO2 paths of lengths 1–5. `verify_word_crf` covers
word compaction, classic/Multi-Head training and loading, independent annotation-free
inference, marginal ensembles and real Ray/Optuna winner replay. Ray requires local sockets.
`benchmark_crf` runs nine CPU ablations with three seeds in isolated processes; its synthetic
strict scores do not establish clinical model quality. See
[the recorded evidence](CONSTRAINED_CRF_STATUS.md).


## Independent sigmoid Multi-Label

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/verify_multilabel.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  .venv-verify/bin/python verification/benchmark_crf.py \
  --variants linear_argmax multihead_argmax multilabel_bce multilabel_weighted_bce \
  multilabel_focal multilabel_asymmetric multihead_crf --output /tmp/multilabel.json
```

The new verification is registered in `run_all.py`. Numeric loss and Viterbi oracles,
independent overlapping targets, CPU autocast, model learning, standalone checkpoints,
nondefault decoders, cached decoder fitting, Ray conditional loss profiles, folds, CLI YAML
and hybrid ensembles use local tiny BERT/DistilBERT fixtures. The shared benchmark retains
its original nine default CRF ablations; `--variants` explicitly selects added binary
families. Cached decoder comparisons reuse each binary worker's same emissions.
[MULTILABEL_RESULTS.md](MULTILABEL_RESULTS.md) separates executed evidence, skips and blocks.
