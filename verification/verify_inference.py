"""Verify inference: model loading, span decoding with scores, artifacts, gold scoring."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


def verify_bio_scores(checks: Checks) -> None:
    from lab.core.spans import span_dataframe
    from lab.ner.evaluation.spans import bio_to_spans, softmax

    offsets = [(0, 4), (5, 9), (10, 14), (15, 20)]
    tags = ["B-DISEASE", "I-DISEASE", "O", "B-DISEASE"]
    text = "aaaa bbbb cccc ddddd"

    plain = bio_to_spans("doc", offsets, tags)

    checks.equal("two spans are merged out of the tag run", len(plain), 2)
    checks.equal("text stays empty without the document", plain[0]["text"], "")
    checks.check("and no score is invented", "score" not in plain[0])

    scored = bio_to_spans("doc", offsets, tags, scores=[0.9, 0.7, 0.1, 0.5], text=text)

    checks.equal("a span's score is the mean of its tokens", scored[0]["score"], 0.8)
    checks.equal("a single-token span keeps its own score", scored[1]["score"], 0.5)
    checks.equal("the surface form comes from the document", scored[0]["text"], "aaaa bbbb")
    checks.equal("and the second one too", scored[1]["text"], "ddddd")
    checks.raises(
        "scores of the wrong length raise",
        ValueError,
        bio_to_spans,
        "doc",
        offsets,
        tags,
        scores=[0.9],
        match="same length",
    )

    duplicated = span_dataframe(
        [
            {"filename": "d", "label": "L", "start_span": 5, "end_span": 9, "text": "x", "score": 0.3},
            {"filename": "d", "label": "L", "start_span": 5, "end_span": 9, "text": "x", "score": 0.8},
            {"filename": "d", "label": "L", "start_span": 0, "end_span": 2, "text": "y", "score": 0.4},
        ]
    )

    checks.equal("a span two windows both predicted is kept once", len(duplicated), 2)
    checks.equal(
        "and the higher-scoring copy is the one kept",
        float(duplicated.loc[duplicated["start_span"] == 5, "score"].iloc[0]),
        0.8,
    )
    checks.equal(
        "scored spans come back in document order",
        duplicated["start_span"].tolist(),
        [0, 5],
    )
    checks.check("the score column is carried", "score" in duplicated.columns)
    checks.check(
        "an unscored frame has no score column", "score" not in span_dataframe([]).columns
    )
    checks.check(
        "but predicting nothing still keeps it when scores were asked for",
        "score" in span_dataframe([], scored=True).columns,
    )

    probabilities = softmax(np.array([[0.0, 0.0], [1000.0, 0.0]]))

    checks.check("softmax rows sum to one", np.allclose(probabilities.sum(axis=-1), 1.0))
    checks.check("a large logit does not overflow", np.isfinite(probabilities).all())


def verify_encoder_description(checks: Checks) -> None:
    from transformers import AutoTokenizer

    from lab.ner.encoding import Encoder, describe_encoder, encoder_from_description

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(
        tokenizer=tokenizer,
        target_label="DISEASE",
        language="es",
        max_length=64,
        strategy="context",
        context_tokens=8,
        min_sentence_tokens=3,
    )
    description = describe_encoder(encoder)

    checks.equal("the windowing budget is recorded", description["max_length"], 64)
    checks.equal("the strategy is recorded by name", description["strategy"], "context")
    checks.equal("its context width comes with it", description["context_tokens"], 8)
    checks.equal("the label vocabulary is recorded", description["label2id"], encoder.label2id)
    checks.equal(
        "the description survives a JSON round trip",
        json.loads(json.dumps(description)),
        description,
    )

    rebuilt = encoder_from_description(description, tokenizer)

    checks.equal("the rebuilt encoder windows the same", rebuilt.max_length, 64)
    checks.equal("with the same strategy", rebuilt.strategy, "context")
    checks.equal("and the same vocabulary", rebuilt.label2id, encoder.label2id)
    checks.check("the guard is still on by default", rebuilt.require_target_label)
    checks.check(
        "an override reaches the rebuilt encoder",
        not encoder_from_description(
            description, tokenizer, require_target_label=False
        ).require_target_label,
    )

    def my_strategy(ranges, tokens, entities, text, budget):
        return []

    custom = describe_encoder(
        Encoder(tokenizer=tokenizer, target_label="DISEASE", language="es", strategy=my_strategy)
    )

    checks.equal("a callable strategy is recorded by name", custom["strategy"], "my_strategy")
    checks.raises(
        "and cannot be rebuilt without being handed back",
        ValueError,
        encoder_from_description,
        custom,
        tokenizer,
        match="Pass strategy=",
    )
    checks.equal(
        "handing it back rebuilds the encoder",
        encoder_from_description(custom, tokenizer, strategy=my_strategy).strategy,
        my_strategy,
    )


def verify_unannotated_encoding(checks: Checks) -> None:
    from transformers import AutoTokenizer

    from fixtures import synthetic_corpus
    from lab.ner.encoding import Encoder
    from lab.ner.encoding.rows import IGNORE_INDEX

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    corpus = synthetic_corpus(4)
    unannotated = corpus.assign(entities_json="[]", n_entities=0)

    settings = {
        "tokenizer": tokenizer,
        "target_label": "DISEASE",
        "language": "es",
        "max_length": 64,
    }

    checks.raises(
        "an unannotated corpus is rejected by default",
        ValueError,
        Encoder(**settings).encode,
        unannotated,
        match="does not occur",
    )

    rows = Encoder(**settings, require_target_label=False).encode(unannotated)

    checks.check("turning the guard off encodes it", len(rows) > 0)

    labels = [label for row in rows["labels"] for label in row]
    outside = Encoder(**settings).label2id["O"]

    checks.check("content tokens are labelled O, not masked", outside in labels)
    checks.check("special tokens stay masked", IGNORE_INDEX in labels)
    checks.check(
        "and nothing is labelled as an entity",
        all(label in (outside, IGNORE_INDEX) for label in labels),
    )


def verify_empty_predictions(checks: Checks) -> None:
    from transformers import AutoTokenizer

    from fixtures import synthetic_corpus
    from lab.ner.encoding import Encoder
    from lab.core.spans import SCORED_SPAN_COLUMNS
    from lab.ner.inference import decode_spans, write_predictions

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(
        tokenizer=tokenizer, target_label="DISEASE", language="es", max_length=64
    )
    rows = encoder.encode(synthetic_corpus(2))

    width = max(len(row) for row in rows["input_ids"])
    outside = encoder.label2id["O"]
    predictions = np.zeros((len(rows), width, len(encoder.label2id)))
    predictions[:, :, outside] = 1.0

    spans = decode_spans(
        rows=rows,
        predictions=predictions,
        tokenizer=tokenizer,
        id2label=encoder.id2label,
        texts={},
        min_score=0.5,
    )

    checks.equal("a model predicting only O decodes no spans", len(spans), 0)
    checks.equal(
        "and still carries the score column min_score reads",
        list(spans.columns),
        SCORED_SPAN_COLUMNS,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = write_predictions(spans, Path(tmp) / "predictions.tsv")

        checks.check("an empty prediction file is still written", path.exists())
        checks.equal(
            "with the header alone",
            path.read_text(encoding="utf-8").strip(),
            "\t".join(SCORED_SPAN_COLUMNS),
        )


def verify_inference_arguments(checks: Checks) -> None:
    import torch

    from lab.ner.inference import resolve_device
    from lab.ner.training.arguments import training_arguments

    checks.equal("an explicit cpu device is honoured", resolve_device("cpu").type, "cpu")
    checks.equal(
        "auto resolves to what this machine has",
        resolve_device("auto").type,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    if not torch.cuda.is_available():
        checks.raises(
            "cuda raises when there is none",
            ValueError,
            resolve_device,
            "cuda",
            match="not available",
        )

    with tempfile.TemporaryDirectory() as tmp:
        arguments = training_arguments(tmp, use_cpu=True, fp16=False)

        checks.equal(
            "use_cpu keeps the Trainer off the GPU it would otherwise take",
            arguments.device.type,
            "cpu",
        )


def verify_document_reading(checks: Checks) -> None:
    from fixtures import synthetic_corpus
    from lab.ner.inference import has_gold, read_documents

    corpus = synthetic_corpus(4)

    checks.equal("a corpus frame is read as-is", len(read_documents(corpus)), 4)
    checks.check("and it carries gold to score against", has_gold(read_documents(corpus)))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        parquet_path = root / "corpus.parquet"
        corpus.to_parquet(parquet_path, index=False)

        checks.equal("a corpus parquet is read", len(read_documents(parquet_path)), 4)

        texts_dir = root / "texts"
        texts_dir.mkdir()

        for name in ("a", "b"):
            (texts_dir / f"{name}.txt").write_text("El paciente tiene fiebre alta.", encoding="utf-8")

        (texts_dir / "ignored.md").write_text("not a document", encoding="utf-8")

        raw = read_documents(texts_dir)

        checks.equal("a directory of text files is read", len(raw), 2)
        checks.equal("the pattern decides what counts", sorted(raw["doc_id"]), ["a", "b"])
        checks.check("raw text carries no gold", not has_gold(raw))
        checks.equal("every document gets an empty entity list", raw["entities_json"].iloc[0], "[]")

        mapping = read_documents({"doc": "El paciente tiene fiebre alta."})

        checks.equal("a mapping is read too", len(mapping), 1)
        checks.raises(
            "a directory with no matching file raises",
            FileNotFoundError,
            read_documents,
            texts_dir,
            pattern="*.conll",
        )


def verify_reference_reading(checks: Checks) -> None:
    from fixtures import synthetic_corpus
    from lab.ner.inference import read_reference, write_gold

    corpus = synthetic_corpus(4)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        parquet_path = root / "gold.parquet"
        corpus.to_parquet(parquet_path, index=False)

        from_parquet = read_reference(parquet_path, "DISEASE")

        checks.check("a corpus parquet becomes gold spans", len(from_parquet) > 0)
        checks.check(
            "restricted to the target label",
            set(from_parquet["label"].unique()) == {"DISEASE"},
        )

        tsv_path = write_gold(from_parquet, root / "gold.tsv")
        from_tsv = read_reference(tsv_path, "DISEASE")

        checks.equal("an annotation TSV reads back the same count", len(from_tsv), len(from_parquet))
        checks.raises(
            "a missing reference raises",
            FileNotFoundError,
            read_reference,
            root / "absent.tsv",
        )


def verify_task_registration(checks: Checks) -> None:
    from lab.ner.inference import predict_entities
    from lab.core.tasks import resolve_task

    checks.check(
        "the task name resolves to inference",
        resolve_task("ner.predict_entities") is predict_entities,
    )


def verify_missing_encoding(checks: Checks) -> None:
    from lab.ner.inference import load_model

    with tempfile.TemporaryDirectory() as tmp:
        checks.raises(
            "a model directory with no encoding.json raises",
            FileNotFoundError,
            load_model,
            tmp,
            match="encoding.json",
        )


def train_saved_model(root: Path, architecture: str) -> Path:
    """Train the miniature BERT for one epoch with weights kept, returning best_model/."""
    from transformers import AutoTokenizer

    from fixtures import synthetic_corpus, tiny_base_model
    from lab.core import prepare_dataset, write_corpus
    from lab.ner.training import train_model

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    base_model = tiny_base_model(root, tokenizer)
    source = write_corpus(synthetic_corpus(16), root / "source" / "documents.parquet")
    prepared = prepare_dataset(
        output_dir=root / "datasets", source_parquet=source, dataset_name="synthetic"
    )

    result = train_model(
        split_dir=prepared.split_dir,
        output_dir=root / "runs" / f"{architecture}_run",
        base_model=str(base_model),
        target_label="DISEASE",
        language="es",
        architecture=architecture,
        training_arguments={
            "num_train_epochs": 1,
            "per_device_train_batch_size": 4,
            "fp16": False,
            "load_best_model_at_end": False,
            "save_strategy": "no",
        },
        max_length=64,
        early_stopping_patience=None,
        save_model=True,
        track_resources=False,
    )

    return result.run_dir / "best_model"


def verify_end_to_end(checks: Checks) -> None:
    from fixtures import synthetic_corpus
    from lab.core.spans import SCORED_SPAN_COLUMNS, SPAN_COLUMNS
    from lab.ner.inference import load_model, predict_entities

    shared = tempfile.TemporaryDirectory()
    root = Path(shared.name)

    model_dir = train_saved_model(root / "linear", "linear")

    checks.check("training writes encoding.json beside the weights", (model_dir / "encoding.json").exists())

    model, tokenizer, description = load_model(model_dir)

    checks.equal("the architecture is recorded", description["model"]["architecture"], "linear")
    checks.equal("the target label is recorded", description["encoding"]["target_label"], "DISEASE")
    checks.equal("the windowing budget is recorded", description["encoding"]["max_length"], 64)
    checks.check("the model comes back in eval mode", not model.training)

    gold_corpus = synthetic_corpus(6)
    scored = predict_entities(
        model_dir=model_dir,
        documents=gold_corpus,
        output_dir=root / "gold_inference",
        device="cpu",
        batch_size=4,
    )

    checks.check("predictions are written", scored.paths["predictions"].exists())
    checks.check("the manifest is written", scored.paths["inference_manifest"].exists())
    checks.check("gold input is scored", scored.metrics is not None)
    checks.check("the metrics file is written", scored.paths["metrics"].exists())
    checks.check("span metrics are among them", "span_strict_f1" in scored.metrics)
    checks.check("character metrics too", "char_f1" in scored.metrics)
    checks.check("and token diagnostics, since the corpus carried the gold", "token_micro_f1" in scored.metrics)
    checks.check("its gold set is written back out", scored.paths["gold"].exists())

    gold_written = pd.read_csv(scored.paths["gold"], sep="\t", dtype=str, keep_default_na=False)
    checks.equal("the gold file is a span table", list(gold_written.columns), SPAN_COLUMNS)
    checks.equal("restricted to the target label", set(gold_written["label"]), {"DISEASE"})
    checks.equal(
        "and it is the gold as annotated, not as windowed",
        len(gold_written),
        int(gold_corpus["n_entities"].sum()),
    )
    checks.check("the summary carries the headline scores", "char F1" in scored.summary())
    checks.equal(
        "the manifest counts the documents",
        scored.manifest["n_documents"],
        len(gold_corpus),
    )
    checks.check("and records the encoding used", scored.manifest["encoding"]["max_length"] == 64)

    predictions = pd.read_csv(scored.paths["predictions"], sep="\t", dtype=str, keep_default_na=False)

    checks.equal(
        "the prediction file is the span table plus a score",
        list(predictions.columns),
        SCORED_SPAN_COLUMNS,
    )
    checks.equal("its rows are the returned spans", len(predictions), len(scored.spans))

    if not scored.spans.empty:
        checks.check(
            "every score is a probability",
            bool(((scored.spans["score"] > 0) & (scored.spans["score"] <= 1)).all()),
        )
        checks.check(
            "each span's text is the document's own",
            all(
                text == document_text(gold_corpus, filename)[start:end]
                for filename, start, end, text in zip(
                    scored.spans["filename"],
                    scored.spans["start_span"],
                    scored.spans["end_span"],
                    scored.spans["text"],
                )
            ),
        )

    raw_dir = root / "raw"
    raw_dir.mkdir()

    for row in gold_corpus.head(3).itertuples(index=False):
        (raw_dir / f"{row.doc_id}.txt").write_text(row.text, encoding="utf-8")

    unscored = predict_entities(
        model_dir=model_dir,
        documents=raw_dir,
        output_dir=root / "raw_inference",
        device="cpu",
        batch_size=4,
    )

    checks.check("raw text predicts without gold", unscored.paths["predictions"].exists())
    checks.check("and is not scored", unscored.metrics is None)
    checks.check("so no gold file is written", "gold" not in unscored.paths)
    checks.equal("the manifest says so", unscored.manifest["scored_against_gold"], False)
    checks.equal("its documents are the ones on disk", unscored.manifest["n_documents"], 3)

    referenced = predict_entities(
        model_dir=model_dir,
        documents=raw_dir,
        output_dir=root / "referenced_inference",
        reference=root / "linear" / "source" / "documents.parquet",
        device="cpu",
        batch_size=4,
    )

    checks.check("a reference gets raw text scored", referenced.metrics is not None)
    checks.check("with span metrics", "span_strict_f1" in referenced.metrics)
    checks.check("and character metrics", "char_f1" in referenced.metrics)
    checks.check("but no token diagnostics, which need gold rows", "token_micro_f1" not in referenced.metrics)
    checks.check("the reference is recorded", referenced.manifest["reference"] is not None)

    filtered = predict_entities(
        model_dir=model_dir,
        documents=gold_corpus,
        output_dir=root / "filtered_inference",
        device="cpu",
        batch_size=4,
        min_score=1.01,
    )

    checks.equal("min_score above every score drops everything", len(filtered.spans), 0)
    checks.equal("and the manifest records the threshold", filtered.manifest["min_score"], 1.01)

    crf_model_dir = train_saved_model(root / "crf", "crf")

    checks.check("a CRF run saves weights too", (crf_model_dir / "encoding.json").exists())

    crf_model, _, crf_description = load_model(crf_model_dir, device="cpu")

    checks.equal("its architecture is recorded", crf_description["model"]["architecture"], "crf")
    checks.check("and the backbone it was built over", "base_model" in crf_description["model"])
    checks.check("a CRF model reloads and can decode", hasattr(crf_model, "decode_from_emissions"))

    crf = predict_entities(
        model_dir=crf_model_dir,
        documents=gold_corpus,
        output_dir=root / "crf_inference",
        device="cpu",
        batch_size=4,
    )

    checks.check("a CRF model predicts through Viterbi decoding", crf.paths["predictions"].exists())
    checks.check("and is scored like a linear one", crf.metrics is not None)

    shared.cleanup()


def document_text(corpus: pd.DataFrame, doc_id: str) -> str:
    """One document's text, for checking a decoded span's surface form."""
    return str(corpus.loc[corpus["doc_id"] == doc_id, "text"].iloc[0])


def main() -> int:
    checks = Checks("inference")

    verify_bio_scores(checks)

    if not torch_available():
        checks.skip("inference", "torch is not installed in this environment")

        return checks.report()

    verify_encoder_description(checks)
    verify_unannotated_encoding(checks)
    verify_empty_predictions(checks)
    verify_inference_arguments(checks)
    verify_document_reading(checks)
    verify_reference_reading(checks)
    verify_task_registration(checks)
    verify_missing_encoding(checks)
    verify_end_to_end(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
