# Cluster Testing Notes

Running log of issues/annoyances found while testing library on cluster. Not decisions — raw findings to work through later.

## Open

## Resolved

### Top-level imports missing for main entry-point functions — D65

Ten stage entry points are now importable as `from ner_lab import train_model`, resolved lazily
so `import ner_lab` still imports nothing and the torch-free half of the library (`data`,
`encoding`, `evaluation`) stays usable without torch. Submodule paths are unchanged.

### `checkpoints` param name in `search_hyperparameters` — D66

Renamed to `base_models`, and `checkpoint` to `base_model` everywhere it meant the pretrained
starting point. The name was doing two jobs: `run_manifest.json`'s `model.checkpoint` (the
backbone) and `training_summary.json`'s `best.checkpoint` (a Trainer snapshot) were the same
word for opposite things in one run directory. `checkpoint` now only means what HF means by it.

### `source_manifest` doesn't list entity labels present in dataset — D67

`source_manifest.json` now carries `n_entities_by_label`, a `{label: count}` mapping whose values
sum to `n_entities`. `ner_lab.data.count_labels(corpus)` returns the same mapping for any corpus
DataFrame.

### `gpus_per_trial` param should be removed from `search_hyperparameters` — D68

Removed. A trial takes one GPU when Ray reports any and runs on CPU when it reports none —
not hardcoded to 1, because an unconditional GPU request makes Ray wait forever on a machine
without one, which is how the end-to-end sweep verification runs here.

### `warmup_ratio` deprecated in transformers, use `warmup_steps` — D69

Renamed in `DEFAULT_SEARCH_SPACE`, same `uniform(0.0, 0.1)` range. No range change was needed:
`warmup_steps` accepts floats and reads anything below 1 as a fraction of total training steps,
so the sweep samples exactly what it sampled before.

### `pynvml` deprecated, use `nvidia-ml-py` — D70

`pyproject.toml` now requires `nvidia-ml-py>=12.0.0` — the floor `pynvml` itself declared. The
`import pynvml` in `tracking.py` is unchanged: only the distribution was renamed.
