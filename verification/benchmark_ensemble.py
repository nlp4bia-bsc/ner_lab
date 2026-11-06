"""Measure ensemble inference, alignment, combination, decoding and tuning separately."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
from verify_unified_ensemble import corpus_fixture, make_checkpoints

from lab.core.segmentation import split_into_words
from lab.core.spans import spans_from_corpus
from lab.ner.ensemble.entities import align_entities
from lab.ner.ensemble.search import search_entity_parameters
from lab.ner.ensemble.tables import predict_members
from lab.ner.ensemble.voting import combine_votes, decode_combined

SCENARIOS = {
    "classic": [0, 2],
    "multihead": [3, 9],
    "hybrid": [0, 1, 6],
    "one_head": [8],
    "three_heads": [6],
    "one_multihead": [3],
}


def measure(scenario, repeats):
    import lab.ner.ensemble.tables as adapters
    from lab.ner.inference.loading import load_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as temporary:
        paths = make_checkpoints(Path(temporary))
        corpus = corpus_fixture()
        texts = dict(zip(corpus.doc_id, corpus.text))
        paths = [paths[index] for index in SCENARIOS[scenario]]
        forwards = []

        def tracked(path, **kwargs):
            model, tokenizer, metadata = load_model(path, **kwargs)
            backbone = model.backbone if hasattr(model, "backbone") else model.base_model
            backbone.register_forward_hook(lambda *args: forwards.append(1))
            return model, tokenizer, metadata

        started = time.perf_counter()
        with patch.object(adapters, "load_model", side_effect=tracked):
            predictions = predict_members(
                paths, corpus, device="cpu", batch_size=2, pad_to_multiple_of=32
            )
        inference_total = time.perf_counter() - started
        started = time.perf_counter()
        labels = sorted({label for member in predictions for label in member.entity_labels})
        aligned = align_entities(predictions, labels)
        alignment = time.perf_counter() - started
        boundaries = {doc_id: split_into_words(text) for doc_id, text in texts.items()}
        started = time.perf_counter()
        for _ in range(repeats):
            combined = {
                label: combine_votes(entity.votes, [1.0] * len(entity.member_indices), "soft")
                for label, entity in aligned.items()
            }
        combination = (time.perf_counter() - started) / repeats
        started = time.perf_counter()
        for _ in range(repeats):
            for label, words in combined.items():
                decode_combined(words, 0.5, texts, label, "break", boundaries)
        decoding = (time.perf_counter() - started) / repeats
        started = time.perf_counter()
        search_entity_parameters(
            aligned,
            spans_from_corpus(corpus),
            texts,
            len(paths),
            "soft",
            True,
            "entity",
            "entity",
            n_trials=3,
            random_state=42,
            min_gold_entities=1,
        )
        tuning = time.perf_counter() - started
        member_times = {
            key: sum(member.timings[key] for member in predictions)
            for key in predictions[0].timings
        }
        payload = sum(
            member.words.memory_usage(index=True, deep=True).sum()
            + member.spans.memory_usage(index=True, deep=True).sum()
            for member in predictions
        )
        arrays = sum(
            entity.votes.present.nbytes
            + entity.votes.tags.nbytes
            + entity.votes.probabilities.nbytes
            for entity in aligned.values()
        )
        if len(forwards) != len(paths) * math.ceil(len(corpus) / 2):
            raise AssertionError("Backbones ran again during optimization or decoding")
        return dict(
            scenario=scenario,
            members=len(paths),
            labels=labels,
            batch_size=2,
            documents=len(corpus),
            max_length=32,
            threads=2,
            device="cpu",
            inference_total_sec=inference_total,
            **member_times,
            alignment_sec=alignment,
            combination_sec=combination,
            decoding_sec=decoding,
            optuna_three_trials_sec=tuning,
            backbone_forwards=len(forwards),
            peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            prediction_payload_bytes=int(payload),
            aligned_array_bytes=arrays,
            machine=platform.machine(),
            torch=torch.__version__,
            repeats=repeats,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.scenario:
        print("BENCHMARK=" + json.dumps(measure(args.scenario, args.repeats)))
        return
    results = []
    for scenario in SCENARIOS:
        completed = subprocess.run(
            [sys.executable, __file__, "--scenario", scenario, "--repeats", str(args.repeats)],
            check=True,
            capture_output=True,
            text=True,
        )
        line = next(line for line in completed.stdout.splitlines() if line.startswith("BENCHMARK="))
        results.append(json.loads(line.split("=", 1)[1]))
    report = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
