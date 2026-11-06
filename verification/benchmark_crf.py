"""Reproducible tiny CPU ablations; isolated processes measure native peak RSS.

Synthetic repeated clinical templates exercise mechanics, not clinical generalization.
All variants use identical backbone weights, document splits, seeds and step budgets.
Classic rows evaluate SYMPTOM alone; Multi-Head rows evaluate the four-type task.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from fixtures import local_tokenizer, tiny_base_model
from verify_multihead import overlap_corpus

from lab.core.spans import spans_from_corpus
from lab.ner import Encoder, build_model
from lab.ner.evaluation.multihead import strict_multihead_metrics
from lab.ner.evaluation.spans import predicted_spans
from lab.ner.training.collation import classification_collator
from lab.ner.training.dataset import to_dataset

VARIANTS = {
    "linear_argmax": ("linear", {}),
    "linear_constrained": ("crf", {"loss_type": "cross_entropy"}),
    "crf_unconstrained": ("crf", {"constrain_transitions": False}),
    "crf_masked": ("crf", {}),
    "crf_auxiliary": ("crf", {"aux_ce_weight": 0.3}),
    "multihead_argmax": ("multihead", {}),
    "multihead_constrained": ("multihead_crf", {"loss_type": "cross_entropy"}),
    "multihead_crf": ("multihead_crf", {}),
    "multihead_crf_auxiliary": ("multihead_crf", {"aux_ce_weight": 0.3}),
}


def timed(function, repeats):
    for _ in range(3):
        function()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats


def measure(variant, seed, steps, repeats):
    torch.set_num_threads(2)
    architecture, kwargs = VARIANTS[variant]
    multihead = variant.startswith("multihead")
    structured = architecture in ("crf", "multihead_crf")
    tokenizer = local_tokenizer()
    corpus = overlap_corpus()
    train_corpus, validation = corpus.iloc[:6], corpus.iloc[6:]
    encoder = Encoder(
        tokenizer, None if multihead else "SYMPTOM", "es", max_length=32, include_word_mask=True
    )
    train_rows = encoder.encode(train_corpus)
    validation_rows = encoder.encode(validation)
    with tempfile.TemporaryDirectory() as temporary:
        torch.manual_seed(97)
        base = tiny_base_model(Path(temporary), tokenizer)
        torch.manual_seed(seed)
        model = build_model(
            str(base),
            encoder.label2id,
            encoder.id2label,
            architecture,
            **({"entity_labels": encoder.entity_labels} if multihead else {}),
            **kwargs,
        )
        # All variants share the exact dropout policy and emission initialization.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        torch.manual_seed(seed)
        with torch.no_grad():
            model.classifier.weight.normal_(0, 0.02)
            model.classifier.bias.zero_()
        collator = classification_collator(model, tokenizer, pad_to_multiple_of=8)

        def batch(rows):
            dataset = to_dataset(rows)
            features = [dataset[index] for index in range(len(dataset))]
            if not structured:
                features = [
                    {k: v for k, v in feature.items() if k != "crf_word_mask"}
                    for feature in features
                ]
            return collator(features)

        train_batch, validation_batch = batch(train_rows), batch(validation_rows)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
        start = time.perf_counter()
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**train_batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        training_seconds = time.perf_counter() - start
        model.eval()
        features = {k: v for k, v in validation_batch.items() if k != "labels"}
        calls = []
        backbone = model.backbone if hasattr(model, "backbone") else model.base_model
        hook = backbone.register_forward_hook(lambda *_: calls.append(1))
        with torch.inference_mode():
            forward_seconds = timed(lambda: model(**features), repeats)
            outputs = model(**features)
            emissions = outputs.logits
            decode_seconds, marginal_seconds = 0.0, 0.0
            if structured:
                decode = lambda: model.structured_from_emissions(
                    emissions, features["crf_word_mask"], features["attention_mask"]
                )
                decode_seconds = timed(decode, repeats)
                tags = decode().tags
                marginal_seconds = timed(
                    lambda: model.structured_from_emissions(
                        emissions,
                        features["crf_word_mask"],
                        features["attention_mask"],
                        probabilities=True,
                    ),
                    repeats,
                )
                from lab.ner.models.word_crf import decision_logits

                predictions = decision_logits(tags).numpy()
            else:
                decode_seconds = timed(lambda: emissions.argmax(-1), repeats)
                tags = emissions.argmax(-1)
                predictions = emissions.numpy()
        hook.remove()
        spans = predicted_spans(validation_rows, predictions, tokenizer, encoder.id2label)
        labels = encoder.entity_labels or ["SYMPTOM"]
        gold = spans_from_corpus(validation)
        gold = gold[gold.label.isin(labels)]
        metrics = strict_multihead_metrics(gold, spans, labels)
        illegal = 0
        for index, row in enumerate(validation_rows.itertuples()):
            selected = tags[index, : len(row.input_ids)].numpy()[row.crf_word_mask]
            selected = selected[:, None] if selected.ndim == 1 else selected
            illegal += int((selected[:1] == 2).sum())
            illegal += int(((selected[1:] == 2) & (selected[:-1] == 0)).sum())
        parameters = sum(p.numel() for p in model.parameters())
        chain_parameters = sum(p.numel() for p in model.chain.parameters()) if structured else 0
        return {
            "variant": variant,
            "seed": seed,
            "steps": steps,
            "heads": len(labels),
            "training_seconds": training_seconds,
            "forward_ms": 1000 * forward_seconds,
            "decode_ms": 1000 * decode_seconds,
            "decode_with_marginals_ms": 1000 * marginal_seconds,
            "windows_per_second": len(validation_rows) / (forward_seconds + decode_seconds),
            "backbone_calls": len(calls),
            "expected_backbone_calls": repeats + 4,
            "parameters": parameters,
            "chain_parameters": chain_parameters,
            "classifier_parameters": sum(p.numel() for p in model.classifier.parameters()),
            "trainable_chain_parameters": sum(
                p.numel() for p in model.chain.parameters() if p.requires_grad
            )
            if structured
            else 0,
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_vram_mib": None,
            "illegal_transitions": illegal,
            **{key: value for key, value in metrics.items() if key.startswith("span_strict")},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("/tmp/crf-benchmark.json"))
    args = parser.parse_args()
    if args.variant:
        args.output.write_text(
            json.dumps(measure(args.variant, args.seed, args.steps, args.repeats), indent=2)
        )
        return
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        for variant in VARIANTS:
            for seed in (21, 22, 23):
                target = Path(temporary) / "result.json"
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--variant",
                        variant,
                        "--seed",
                        str(seed),
                        "--steps",
                        str(args.steps),
                        "--repeats",
                        str(args.repeats),
                        "--output",
                        str(target),
                    ],
                    check=True,
                )
                results.append(json.loads(target.read_text()))
    report = {
        "environment": {
            "device": "cpu",
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": 2,
            "cpus_available": len(os.sched_getaffinity(0)),
        },
        "protocol": {
            "batch_train": 6,
            "batch_validation": 3,
            "padded_length": 16,
            "max_length": 32,
            "backbone": "random BERT H32 L2 heads2",
            "steps": args.steps,
            "seeds": [21, 22, 23],
            "timing_repeats": args.repeats,
            "warmups": 3,
            "limitations": "Repeated synthetic template across documents; no clinical generalization evidence. RSS includes imports, model and training optimizer. Classic SYMPTOM and four-type tasks differ; compare within each family.",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    frame = pd.DataFrame(results)
    print(
        frame.groupby("variant")[
            [
                "span_strict_macro_f1",
                "forward_ms",
                "decode_ms",
                "training_seconds",
                "peak_process_rss_mib",
            ]
        ]
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
