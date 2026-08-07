# Verification

Behavioural checks for `ner_lab`, kept outside `src/` so they are never packaged or
installed. They are plain scripts, not a test suite — run them directly, read the output.

```bash
uv venv .venv-verify --python 3.12
uv pip install --python .venv-verify pandas pyarrow pyyaml pysbd transformers datasets accelerate pytorch-crf nervaluate
uv pip install --python .venv-verify torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-verify --no-deps -e .

.venv-verify/bin/python verification/run_all.py
```

`torch` is the CPU build. `data/` and `encoding/` never touch it, but `models/` and
`training/` do, and a CPU wheel is enough: the end-to-end training checks run against a
randomly initialized miniature BERT written to a temporary directory, so nothing is
downloaded and a full run stays in the seconds.

| Script | Covers |
|---|---|
| `verify_data.py` | labels, BRAT reading, the canonical schema, stratification, splitting, `prepare_dataset` |
| `verify_encoding.py` | overlap policies, segmentation, windowing, IOB2 tagging, row assembly |
| `verify_encoder.py` | `Encoder` construction and encoding, custom strategies, NER-API equivalence |
| `verify_models.py` | BIO constraint masks, the architecture registry, custom factories |
| `verify_training.py` | row conversion, `training_arguments`, and linear + CRF training end to end |
| `verify_evaluation.py` | span reconstruction, nervaluate scoring, token metrics, official scorer |

Everything runs on synthetic fixtures built by `fixtures.py`, so the scripts pass on a
machine with no corpora at all. Two groups of checks additionally use real data and skip
cleanly when it is missing:

- **Real corpora** — `NER_LAB_SAMPLES`, defaulting to `~/bsc/NER-API/data_samples`.
- **NER-API equivalence** — `NER_API_ROOT`, defaulting to `~/bsc/NER-API`. Runs the old
  `DataLoader` and the new `Encoder` over the same 60 documents for both tokenizers and both
  window strategies, and asserts the resulting frames are identical.

A skip is printed, never silently passed. Check the summary line for a skip count before
trusting a green run.
