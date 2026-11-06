"""Reproducible CRF kernel audit, optionally compared with a local Git revision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

from lab.ner.models.chain import ConstrainedChain
from lab.ner.models.packing import pack_words


def baseline_modules(revision):
    modules = []
    for name in ("chain", "packing"):
        source = subprocess.run(
            ["git", "show", f"{revision}:src/lab/ner/models/{name}.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        module = types.ModuleType("audit_baseline_" + name)
        sys.modules[module.__name__] = module
        exec(compile(source, f"{revision}/{name}.py", "exec"), module.__dict__)
        modules.append(module)
    return modules[0].ConstrainedChain, modules[1].pack_words


def measure(function, device, repeats):
    for _ in range(5):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) / repeats
    peak = torch.cuda.max_memory_allocated(device) - allocated if device.type == "cuda" else None
    return {"wall_seconds_per_call": elapsed, "incremental_peak_cuda_bytes": peak}


def audit(chain_type, pack, values, mask, repeats):
    chain = chain_type({0: "O", 1: "B", 2: "I"}, values.shape[2]).to(values.device)
    packed = pack(values, mask)
    gold = torch.zeros_like(packed.mask, dtype=torch.long)

    def packing():
        compact = pack(values, mask)
        return compact.restore(compact.emissions)

    operations = {
        "pack_restore": packing,
        "gold_score": lambda: chain.gold_score(packed.emissions, gold, packed.mask),
        "nll": lambda: chain(packed.emissions, gold, packed.mask),
        "viterbi": lambda: chain.decode(packed.emissions, packed.mask),
        "marginals": lambda: chain.log_marginals(packed.emissions, packed.mask),
    }
    with torch.no_grad():
        metrics = {
            name: measure(function, values.device, repeats) for name, function in operations.items()
        }
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            packing()
        metrics["pack_restore"]["nonzero_operations"] = sum(
            event.count for event in profile.key_averages() if event.key == "aten::nonzero"
        )
        metrics["pack_restore"]["scalar_extractions"] = sum(
            event.count
            for event in profile.key_averages()
            if event.key == "aten::_local_scalar_dense"
        )
        outputs = {name: function().detach().cpu() for name, function in operations.items()}
    return metrics, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="An existing local Git revision, for example ea8c4f1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("repeats must be positive")
    device = torch.device(
        "cuda:0"
        if arguments.device == "cuda" or arguments.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(51)
    values = torch.randn(8, 128, 6, 3, device=device)
    mask = torch.rand(8, 128, 6, device=device) > 0.25
    mask[0] = False
    current, outputs = audit(ConstrainedChain, pack_words, values, mask, arguments.repeats)
    report = {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "threads": 2,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "protocol": {
            "shape": list(values.shape),
            "repeats": arguments.repeats,
            "warmups": 5,
            "gradient": False,
            "includes_backbone": False,
            "cuda_timing_synchronized": device.type == "cuda",
        },
        "current": current,
    }
    if arguments.baseline:
        chain_type, pack = baseline_modules(arguments.baseline)
        baseline, reference = audit(chain_type, pack, values, mask, arguments.repeats)
        report["baseline_revision"] = arguments.baseline
        report["baseline"] = baseline
        report["numerical_equivalence"] = {
            name: bool(torch.allclose(output, reference[name], atol=2e-5, rtol=2e-5))
            for name, output in outputs.items()
        }
        if not all(report["numerical_equivalence"].values()):
            raise AssertionError("Kernel outputs differ from baseline")
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
