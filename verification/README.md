# Verification

Behavioural checks for `lab`, kept outside `src/` so they are never packaged or installed.
They are plain scripts, not a test suite — run them directly, read the output.

```bash
uv venv .venv-verify --python 3.12
uv pip install --python .venv-verify pandas pyarrow pyyaml pysbd transformers datasets accelerate pytorch-crf nervaluate "ray[tune]" optuna
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
| `verify_encoding.py` | overlap policies, segmentation, windowing, IOB2 tagging, row assembly |
| `verify_encoder.py` | `Encoder` construction and encoding, custom strategies, NER-API equivalence |
| `verify_models.py` | BIO constraint masks, the architecture registry, custom factories |
| `verify_training.py` | row conversion, `training_arguments`, and linear + CRF training end to end |
| `verify_evaluation.py` | span reconstruction, nervaluate scoring, character metrics, token metrics |
| `verify_assessment.py` | `train_model`: fold rotation, both split modes, manifests, aggregation |
| `verify_hpo.py` | search-space forms, variants, trial scoring, OOM handling, a real two-trial Ray sweep, the winner block re-run through `train_model` |
| `verify_inference.py` | span decoding, `predict_entities` end to end, scoring against gold and against a reference |
| `verify_nel.py` | nlp4bia-linking's own tests, side-by-side equivalence with it, `link_entities` end to end on every method, with a hierarchy, with the reranker |

Everything runs on synthetic fixtures built by `fixtures.py`, so the scripts pass on a
machine with no corpora at all. Two groups of checks additionally use real data and skip
cleanly when it is missing:

- **Real corpora** — `NER_LAB_SAMPLES`, defaulting to `~/bsc/NER-API/data_samples`.
- **NER-API equivalence** — `NER_API_ROOT`, defaulting to `~/bsc/NER-API`. Runs the old
  `DataLoader` and the new `Encoder` over the same 60 documents for both tokenizers and both
  window strategies, and asserts the resulting frames are identical.
- **nlp4bia-linking equivalence** — `NLP4BIA_LINKING_ROOT`, defaulting to
  `~/bsc/nlp4bia-linking`. Runs every matcher, the sparse retrievers, RRF, the metrics, the
  graph distances and the readers from both packages over the same synthetic gazetteer and
  mentions, and asserts the candidates are identical.

A skip is printed, never silently passed. Check the summary line for a skip count before
trusting a green run.
