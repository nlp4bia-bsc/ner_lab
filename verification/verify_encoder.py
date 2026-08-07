"""Verify the Encoder, and diff it against NER-API's DataLoader when that repo is present."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run
from fixtures import samples_root, synthetic_corpus

from ner_lab.data import build_corpus, write_corpus
from ner_lab.encoding import Encoder, build_iob2_labels, build_window

NER_API_ENV = "NER_API_ROOT"
DEFAULT_NER_API = Path.home() / "bsc" / "NER-API"


def verify_construction(checks: Checks, tokenizer) -> None:
    encoder = Encoder(tokenizer, "DISEASE", "es")

    checks.equal("label2id is the three-class vocabulary", len(encoder.label2id), 3)
    checks.equal(
        "content budget accounts for special tokens",
        encoder.max_content_length,
        256 - len(encoder.prefix_ids) - len(encoder.suffix_ids),
    )

    checks.raises(
        "language is required",
        TypeError,
        Encoder,
        tokenizer,
        "DISEASE",
    )
    checks.raises(
        "context_tokens is required for the context strategy",
        ValueError,
        Encoder,
        tokenizer,
        "DISEASE",
        "es",
        strategy="context",
        match="context_tokens is required",
    )
    checks.raises(
        "context_tokens is rejected for the greedy strategy",
        ValueError,
        Encoder,
        tokenizer,
        "DISEASE",
        "es",
        context_tokens=16,
        match="only applies",
    )
    checks.raises(
        "an unknown strategy raises",
        ValueError,
        Encoder,
        tokenizer,
        "DISEASE",
        "es",
        strategy="sliding",
        match="Unknown strategy",
    )
    checks.raises(
        "a max_length below the template raises",
        ValueError,
        Encoder,
        tokenizer,
        "DISEASE",
        "es",
        max_length=1,
        match="leaves no room",
    )
    checks.raises(
        "an empty target_label raises",
        ValueError,
        Encoder,
        tokenizer,
        "  ",
        "es",
    )


def verify_encode(checks: Checks, tokenizer, workspace: Path) -> None:
    corpus = synthetic_corpus(20)
    encoder = Encoder(tokenizer, "DISEASE", "es", max_length=64)

    rows = encoder.encode(corpus)

    checks.check("encoding produces rows", len(rows) > 0)
    checks.equal(
        "row columns",
        sorted(rows.columns),
        [
            "attention_mask",
            "doc_id",
            "input_ids",
            "labels",
            "token_offsets",
            "window_end",
            "window_start",
            "word_ids",
        ],
    )
    checks.equal(
        "every document is represented",
        set(rows["doc_id"]),
        set(corpus["doc_id"]),
    )
    checks.check(
        "no row exceeds max_length",
        rows["input_ids"].map(len).max() <= 64,
    )
    checks.check(
        "labels align with input_ids on every row",
        all(len(a) == len(b) for a, b in zip(rows["input_ids"], rows["labels"])),
    )
    checks.check(
        "some rows carry a positive label",
        any(any(label > 0 for label in labels) for labels in rows["labels"]),
    )
    checks.check(
        "window spans are well formed",
        all(start < end for start, end in zip(rows["window_start"], rows["window_end"])),
    )

    corpus_path = workspace / "corpus.parquet"
    write_corpus(corpus, corpus_path)

    from_parquet = encoder.encode_parquet(corpus_path)
    checks.frames_equal("encode_parquet matches encode", from_parquet, rows)

    small_batch = Encoder(
        tokenizer, "DISEASE", "es", max_length=64, documents_per_batch=3
    ).encode_parquet(corpus_path)
    checks.frames_equal("batch size does not change the output", small_batch, rows)

    document = corpus.iloc[0]
    single = encoder.encode_document(
        document["doc_id"], document["text"], json.loads(document["entities_json"])
    )
    checks.equal(
        "encode_document matches that document's rows",
        len(single),
        int((rows["doc_id"] == document["doc_id"]).sum()),
    )

    context_rows = Encoder(
        tokenizer, "DISEASE", "es", max_length=64, strategy="context", context_tokens=8
    ).encode(corpus)

    checks.check("the context strategy produces rows", len(context_rows) > 0)
    checks.check(
        "context windows overlap, so there are more of them",
        len(context_rows) >= len(rows),
    )
    checks.check(
        "context rows still fit max_length",
        context_rows["input_ids"].map(len).max() <= 64,
    )

    absent = Encoder(tokenizer, "MEDICATION", "es", max_length=64)
    checks.raises(
        "a target_label absent from the corpus raises",
        ValueError,
        absent.encode,
        corpus,
        match="Labels present: ['DISEASE']",
    )

    verbatim = synthetic_corpus(6, label="FARMACO")
    checks.equal(
        "encoding never normalizes labels",
        len(Encoder(tokenizer, "FARMACO", "es", max_length=64).encode(verbatim)) > 0,
        True,
    )
    checks.raises(
        "the alias of a corpus label is still refused",
        ValueError,
        Encoder(tokenizer, "MEDICATION", "es", max_length=64).encode,
        verbatim,
        match="Labels present: ['FARMACO']",
    )

    gene = synthetic_corpus(6, label="GENE")
    checks.check(
        "an arbitrary entity type encodes",
        len(Encoder(tokenizer, "GENE", "es", max_length=64).encode(gene)) > 0,
    )


def verify_custom_strategy(checks: Checks, tokenizer) -> None:
    corpus = synthetic_corpus(8)
    calls = []

    def whole_document(sentence_ranges, tokens, entities, text, max_content_length):
        calls.append(len(tokens))

        return [build_window(0, min(len(tokens), max_content_length), tokens, text)]

    rows = Encoder(tokenizer, "DISEASE", "es", max_length=256, strategy=whole_document).encode(corpus)

    checks.equal("a callable strategy is used", len(calls), len(corpus))
    checks.equal("one window per document", len(rows), len(corpus))

    def cored(sentence_ranges, tokens, entities, text, max_content_length):
        end = min(len(tokens), max_content_length)
        window = build_window(0, end, tokens, text)
        window["core_token_start"] = end // 2
        window["core_token_end"] = end

        return [window]

    cored_rows = Encoder(tokenizer, "DISEASE", "es", max_length=256, strategy=cored).encode(corpus)
    first = cored_rows.iloc[0]
    prefix_length = len(Encoder(tokenizer, "DISEASE", "es").prefix_ids)
    flank_length = len(first["token_offsets"]) // 2

    checks.equal(
        "a custom core gets its flank masked",
        set(first["labels"][prefix_length:prefix_length + flank_length]),
        {-100},
    )
    checks.check(
        "the core itself is labelled",
        any(label >= 0 for label in first["labels"][prefix_length + flank_length:]),
    )


def verify_against_ner_api(checks: Checks, tokenizers: dict, workspace: Path) -> None:
    ner_api = Path(os.environ.get(NER_API_ENV, DEFAULT_NER_API))
    samples = samples_root()

    if not (ner_api / "src").is_dir():
        checks.skip("NER-API equivalence", f"{ner_api} not found")
        return

    if samples is None:
        checks.skip("NER-API equivalence", "sample corpora not found")
        return

    source = samples / "MultiClinNER-es-train-disease"

    if not (source / "txt").is_dir():
        checks.skip("NER-API equivalence", f"{source} has no txt/ directory")
        return

    sys.path.insert(0, str(ner_api / "src"))

    try:
        from preprocessing.dataset_loader import DataLoader
    except Exception as error:
        checks.skip("NER-API equivalence", f"cannot import DataLoader: {error}")
        return

    corpus = build_corpus(source / "txt", source / "ann", normalize_labels=True)
    sample = corpus.head(60).reset_index(drop=True)

    sample_path = workspace / "equivalence.parquet"
    write_corpus(sample, sample_path)

    policy = "merge_same_label_then_keep_longest"

    for tokenizer_name, tokenizer in tokenizers.items():
        for strategy, extra in (("greedy", {}), ("context", {"context_tokens": 64})):
            expected = DataLoader(
                tokenizer=tokenizer,
                target_label="DISEASE",
                max_length=256,
                overlap_policy=policy,
                strategy=strategy,
                **extra,
            ).load(sample_path)

            encoder = Encoder(
                tokenizer,
                "DISEASE",
                "es",
                max_length=256,
                overlap_policy=policy,
                strategy=strategy,
                **extra,
            )
            actual = encoder.encode(sample)

            label = f"{tokenizer_name}/{strategy}"

            if not checks.equal(f"{label}: row count ({len(expected)})", len(actual), len(expected)):
                continue

            checks.equal(f"{label}: columns", sorted(actual.columns), sorted(expected.columns))
            checks.frames_equal(
                f"{label}: rows identical",
                actual[sorted(actual.columns)].reset_index(drop=True),
                expected[sorted(expected.columns)].reset_index(drop=True),
            )
            checks.frames_equal(
                f"{label}: encode_parquet agrees too",
                encoder.encode_parquet(sample_path),
                actual,
            )


def main() -> int:
    from transformers import AutoTokenizer

    checks = Checks("encoder")

    tokenizers = {
        "bert": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "gpt2": AutoTokenizer.from_pretrained("gpt2"),
    }

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        verify_construction(checks, tokenizers["bert"])
        verify_encode(checks, tokenizers["bert"], workspace)
        verify_custom_strategy(checks, tokenizers["bert"])
        verify_against_ner_api(checks, tokenizers, workspace)

    return checks.report()


if __name__ == "__main__":
    run(main)
