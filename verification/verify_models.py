"""Verify the BIO constraint logic, and the model factory when torch is installed."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run

from lab.ner.models.bio import (
    bio_constraint_masks,
    bio_entity,
    is_inside_label,
    normalize_id2label,
    normalize_label2id,
)

VOCABULARY = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE", 3: "B-GENE", 4: "I-GENE"}


def verify_bio(checks: Checks) -> None:
    checks.equal("O has no entity", bio_entity("O"), None)
    checks.equal("B- prefix is stripped", bio_entity("B-DISEASE"), "DISEASE")
    checks.equal("I- prefix is stripped", bio_entity("I-DISEASE"), "DISEASE")
    checks.equal("a bare label is its own entity", bio_entity("DISEASE"), "DISEASE")
    checks.equal("a hyphenated entity survives", bio_entity("B-CELL-LINE"), "CELL-LINE")

    checks.check("I- is inside", is_inside_label("I-DISEASE"))
    checks.check("B- is not inside", not is_inside_label("B-DISEASE"))
    checks.check("O is not inside", not is_inside_label("O"))

    checks.equal("label2id coerces types", normalize_label2id({"O": "0"}), {"O": 0})
    checks.equal("id2label coerces types", normalize_id2label({"0": "O"}), {0: "O"})

    invalid_start, invalid = bio_constraint_masks(VOCABULARY)

    checks.equal("one start flag per label", len(invalid_start), len(VOCABULARY))
    checks.equal(
        "only I- labels are invalid starts",
        [index for index, flag in enumerate(invalid_start) if flag],
        [2, 4],
    )

    checks.equal("transition matrix is square", (len(invalid), len(invalid[0])), (5, 5))
    checks.check("O to I-DISEASE is invalid", invalid[0][2])
    checks.check("B-DISEASE to I-DISEASE is valid", not invalid[1][2])
    checks.check("I-DISEASE to I-DISEASE is valid", not invalid[2][2])
    checks.check("B-GENE to I-DISEASE is invalid", invalid[3][2])
    checks.check("I-GENE to I-DISEASE is invalid", invalid[4][2])
    checks.check("B-DISEASE to I-GENE is invalid", invalid[1][4])
    checks.check("anything to O is valid", not any(row[0] for row in invalid))
    checks.check("anything to B-DISEASE is valid", not any(row[1] for row in invalid))

    single_start, single = bio_constraint_masks({0: "O", 1: "B-DISEASE", 2: "I-DISEASE"})

    checks.equal("single entity start flags", single_start, [False, False, True])
    checks.equal(
        "single entity invalid transitions",
        [(row, column) for row, values in enumerate(single) for column, flag in enumerate(values) if flag],
        [(0, 2)],
    )

    empty_start, empty = bio_constraint_masks({0: "O"})

    checks.equal("an O-only vocabulary has no invalid start", empty_start, [False])
    checks.equal("an O-only vocabulary has no invalid transition", empty, [[False]])


def verify_registry(checks: Checks) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        checks.skip("model factory", "torch is not installed in this environment")
        return

    from lab.ner.models import BUILTIN_ARCHITECTURES, build_model

    checks.equal("builtin architectures", BUILTIN_ARCHITECTURES, ("linear", "crf"))

    calls = []

    def custom(base_model, label2id, id2label, **kwargs):
        calls.append((base_model, tuple(sorted(label2id)), kwargs))

        return "custom-model"

    label2id = {"O": 0, "B-DISEASE": 1, "I-DISEASE": 2}
    id2label = {index: label for label, index in label2id.items()}

    result = build_model("dummy", label2id, id2label, architecture=custom, dropout=0.3)

    checks.equal("a callable architecture is used", result, "custom-model")
    checks.equal("the factory receives the base_model", calls[0][0], "dummy")
    checks.equal("the factory receives the vocabulary", calls[0][1], tuple(sorted(label2id)))
    checks.equal("extra kwargs reach the factory", calls[0][2], {"dropout": 0.3})

    checks.raises(
        "an unknown architecture raises",
        ValueError,
        build_model,
        "dummy",
        label2id,
        id2label,
        architecture="lstm",
        match="Unknown architecture",
    )

    def typed(base_model, label2id, id2label, **kwargs):
        return (label2id, id2label)

    normalized = build_model("dummy", {"O": "0"}, {"0": "O"}, architecture=typed)

    checks.equal("label2id is normalized before the factory sees it", normalized[0], {"O": 0})
    checks.equal("id2label is normalized before the factory sees it", normalized[1], {0: "O"})


def verify_capacity(checks: Checks) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        checks.skip("max_length check", "torch is not installed in this environment")
        return

    from fixtures import tiny_base_model
    from transformers import (
        AutoTokenizer,
        BertConfig,
        BertForTokenClassification,
        RobertaConfig,
        RobertaForTokenClassification,
    )

    from lab.ner.models import build_model, check_max_length

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sizes = {"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
             "intermediate_size": 64, "vocab_size": tokenizer.vocab_size, "num_labels": 3}

    bert = BertForTokenClassification(BertConfig(max_position_embeddings=64, **sizes))
    bert.train()
    printed = io.StringIO()

    with contextlib.redirect_stdout(printed):
        check_max_length(bert, tokenizer, 64, "tiny-bert")

    checks.check("a length the model takes passes and says so", "[check] tiny-bert" in printed.getvalue())
    checks.check("the model's training mode is restored", bert.training)

    silent = io.StringIO()

    with contextlib.redirect_stdout(silent):
        check_max_length(bert, tokenizer, 64, "tiny-bert", report=False)

    checks.equal("report=False prints nothing", silent.getvalue(), "")
    checks.raises(
        "one token past the position table raises, naming the length",
        ValueError,
        check_max_length,
        bert,
        tokenizer,
        65,
        "tiny-bert",
        match="max_length=65",
    )

    roberta = RobertaForTokenClassification(
        RobertaConfig(max_position_embeddings=66, pad_token_id=1, **sizes)
    )

    printed = io.StringIO()

    with contextlib.redirect_stdout(printed):
        check_max_length(roberta, tokenizer, 64, "tiny-roberta")

    checks.check(
        "RoBERTa takes max_position_embeddings - 2, its padding offset",
        "[check] tiny-roberta" in printed.getvalue(),
    )
    checks.raises(
        "and not one more",
        ValueError,
        check_max_length,
        roberta,
        tokenizer,
        65,
        "tiny-roberta",
        match="max_position_embeddings=66",
    )

    with tempfile.TemporaryDirectory() as tmp:
        base_model = tiny_base_model(Path(tmp), tokenizer)
        crf = build_model(str(base_model), {"O": 0, "B-D": 1, "I-D": 2}, {0: "O", 1: "B-D", 2: "I-D"}, "crf")
        printed = io.StringIO()

        with contextlib.redirect_stdout(printed):
            check_max_length(crf, tokenizer, 64, "tiny-crf")

        checks.check("the CRF architecture is checked too", "[check] tiny-crf" in printed.getvalue())


def main() -> int:
    checks = Checks("models")

    verify_bio(checks)
    verify_registry(checks)
    verify_capacity(checks)

    return checks.report()


if __name__ == "__main__":
    run(main)
