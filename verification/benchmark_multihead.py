"""Measure CPU forward cost under identical inputs; run from the repository root.

Each scenario runs in its own process so ru_maxrss is a process peak, not a shared
high-water mark. Timings cover model inference only, excluding encoding and decoding.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from fixtures import tiny_base_model
from verify_multihead import local_tokenizer

from lab.ner import build_model


def measure(mode: str, heads: int, batch: int, length: int, repeats: int) -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as temporary:
        tokenizer = local_tokenizer()
        base = tiny_base_model(Path(temporary), tokenizer)
        vocabulary = {"O": 0, "B": 1, "I": 2}
        multihead = mode == "multihead"
        models = [
            build_model(
                str(base),
                vocabulary,
                dict(enumerate(vocabulary)),
                architecture="multihead" if multihead else "linear",
                **({"entity_labels": [f"ENTITY_{i}" for i in range(heads)]} if multihead else {}),
            )
            for _ in range(heads if mode == "independent" else 1)
        ]
        for model in models:
            model.eval()
        inputs = dict(
            input_ids=torch.randint(5, tokenizer.vocab_size, (batch, length)),
            attention_mask=torch.ones(batch, length, dtype=torch.long),
        )
        calls = []
        handles = []
        for model in models:
            backbone = model.backbone if multihead else model.base_model
            handles.append(backbone.register_forward_hook(lambda *args: calls.append(1)))
        with torch.inference_mode():
            for _ in range(5):
                for model in models:
                    model(**inputs)
            calls.clear()
            started = time.perf_counter()
            for _ in range(repeats):
                for model in models:
                    model(**inputs)
            elapsed = time.perf_counter() - started
        for handle in handles:
            handle.remove()
        parameters = sum(p.numel() for model in models for p in model.parameters())
        classifier = sum(p.numel() for model in models for p in model.classifier.parameters())

    return dict(
        mode=mode,
        heads=heads if mode != "linear" else 1,
        batch=batch,
        length=length,
        repeats=repeats,
        forwards=len(calls),
        seconds=elapsed,
        windows_per_second=batch * repeats / elapsed,
        tokens_per_second=batch * length * repeats / elapsed,
        parameters=parameters,
        classifier_parameters=classifier,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        threads=2,
        device="cpu",
        machine=platform.machine(),
        torch=torch.__version__,
        scope="model forward only, five warmups",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["linear", "multihead", "independent"])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode:
        print(
            "BENCHMARK="
            + json.dumps(measure(args.mode, args.heads, args.batch, args.length, args.repeats))
        )
        return
    results = []
    for mode in ["linear", "multihead", "independent"]:
        command = [
            sys.executable,
            __file__,
            "--mode",
            mode,
            "--heads",
            str(args.heads),
            "--batch",
            str(args.batch),
            "--length",
            str(args.length),
            "--repeats",
            str(args.repeats),
        ]
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        result = next(line for line in process.stdout.splitlines() if line.startswith("BENCHMARK="))
        results.append(json.loads(result.split("=", 1)[1]))
    report = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
