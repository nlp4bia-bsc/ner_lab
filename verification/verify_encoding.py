"""Verify the encoding leaf modules: overlaps, segmentation, windowing, tagging, rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _harness import Checks, run

from ner_lab.encoding import (
    IGNORE_INDEX,
    build_iob2_labels,
    build_label_vocabulary,
    build_row,
    build_window,
    compute_max_content_length,
    merge_ranges,
    merge_sentences_crossing_entities,
    merge_short_sentences,
    resolve_entities,
    select_context_windows,
    select_greedy_windows,
    sentence_token_ranges,
    special_token_template,
    split_into_sentences,
    split_oversized_sentence,
    tokenize_document,
)

TEXT = (
    "El paciente refiere fiebre alta desde el lunes. "
    "Se observa neumonia bilateral en la radiografia. "
    "No refiere dolor toracico ni disnea de reposo. "
    "Se pauta tratamiento antibiotico durante diez dias."
)


def entity(start: int, end: int, label: str = "DISEASE", text: str | None = None) -> dict:
    return {"start": start, "end": end, "label": label, "text": text or TEXT[start:end]}


def verify_overlaps(checks: Checks) -> None:
    text = "neumonia bilateral grave"
    outer = {"start": 0, "end": 24, "label": "DISEASE", "text": text}
    inner = {"start": 0, "end": 8, "label": "DISEASE", "text": "neumonia"}
    other = {"start": 9, "end": 18, "label": "SYMPTOM", "text": "bilateral"}

    checks.equal(
        "none keeps every overlap",
        len(resolve_entities("d", [outer, inner], text, "none")),
        2,
    )
    checks.equal(
        "keep_longest collapses to the longest",
        resolve_entities("d", [outer, inner], text, "keep_longest")[0]["end"],
        24,
    )
    checks.equal(
        "keep_shortest collapses to the shortest",
        resolve_entities("d", [outer, inner], text, "keep_shortest")[0]["end"],
        8,
    )
    checks.raises(
        "error policy raises on overlap",
        ValueError,
        resolve_entities,
        "d",
        [outer, inner],
        text,
        "error",
    )
    checks.equal(
        "error policy accepts disjoint spans",
        len(resolve_entities("d", [inner, {**other, "start": 9}], text, "error")),
        2,
    )

    merged = resolve_entities(
        "d",
        [inner, {"start": 9, "end": 24, "label": "DISEASE", "text": "bilateral grave"}],
        text,
        "merge_same_label_then_keep_longest",
    )
    checks.equal("adjacent same-label spans are not merged", len(merged), 2)

    overlapping_same_label = [
        {"start": 0, "end": 18, "label": "DISEASE", "text": "neumonia bilateral"},
        {"start": 9, "end": 24, "label": "DISEASE", "text": "bilateral grave"},
    ]
    fused = resolve_entities("d", overlapping_same_label, text, "merge_same_label_then_keep_longest")

    checks.equal("overlapping same-label spans fuse", len(fused), 1)
    checks.equal("fused span covers both", (fused[0]["start"], fused[0]["end"]), (0, 24))
    checks.equal("fused text is re-read from the document", fused[0]["text"], text)

    extra = {**inner, "id": "T1", "note": "kept"}
    preserved = resolve_entities("d", [extra], text, "none")[0]

    checks.equal("extra entity keys survive", preserved["id"], "T1")
    checks.equal("unrelated keys survive too", preserved["note"], "kept")

    checks.raises(
        "span past the end of the text raises",
        ValueError,
        resolve_entities,
        "d",
        [{"start": 0, "end": 99, "label": "DISEASE", "text": text}],
        text,
        "none",
    )
    checks.raises(
        "mismatched entity text raises",
        ValueError,
        resolve_entities,
        "d",
        [{"start": 0, "end": 8, "label": "DISEASE", "text": "wrong"}],
        text,
        "none",
    )
    checks.raises(
        "inverted span raises",
        ValueError,
        resolve_entities,
        "d",
        [{"start": 8, "end": 2, "label": "DISEASE", "text": ""}],
        text,
        "none",
    )
    checks.raises(
        "unknown policy raises",
        ValueError,
        resolve_entities,
        "d",
        [inner],
        text,
        "keep_middle",
    )


def verify_segmentation(checks: Checks, tokenizer) -> None:
    sentences = split_into_sentences(TEXT, "es")

    checks.equal("four sentences", len(sentences), 4)
    checks.check(
        "sentence spans round-trip",
        all(TEXT[s["start"]:s["end"]] == s["text"] for s in sentences),
    )
    checks.equal("sentences start at zero", sentences[0]["start"], 0)
    checks.equal("sentences reach the end", sentences[-1]["end"], len(TEXT))

    tokens = tokenize_document(TEXT, tokenizer)

    checks.check("document tokenizes", len(tokens) > 0)
    checks.check(
        "token offsets round-trip",
        all(TEXT[t["start"]:t["end"]] for t in tokens),
    )
    checks.check("tokens carry word ids", all("word_id" in t for t in tokens))
    checks.check(
        "token offsets are non-decreasing",
        all(a["start"] <= b["start"] for a, b in zip(tokens, tokens[1:])),
    )

    ranges = sentence_token_ranges(sentences, tokens)

    checks.equal("one range per sentence", len(ranges), len(sentences))
    checks.equal("ranges tile the token stream", ranges[0]["token_start"], 0)
    checks.equal("last range ends at the last token", ranges[-1]["token_end"], len(tokens))
    checks.check(
        "ranges are contiguous",
        all(a["token_end"] == b["token_start"] for a, b in zip(ranges, ranges[1:])),
    )

    crossing = [entity(40, 60)]
    merged = merge_sentences_crossing_entities(sentences, crossing, TEXT)

    checks.check("an entity spanning a boundary merges its sentences", len(merged) < len(sentences))
    checks.equal("merging preserves coverage", merged[-1]["end"], len(TEXT))
    checks.equal(
        "no entity leaves the sentences alone",
        len(merge_sentences_crossing_entities(sentences, [], TEXT)),
        len(sentences),
    )

    short = merge_short_sentences(ranges, 1000, TEXT)
    checks.equal("a huge minimum merges everything", len(short), 1)
    checks.equal(
        "a zero minimum merges nothing", len(merge_short_sentences(ranges, 0, TEXT)), len(ranges)
    )

    fused = merge_ranges(ranges[0], ranges[1], TEXT)

    checks.equal("merge_ranges takes the left start", fused["token_start"], ranges[0]["token_start"])
    checks.equal("merge_ranges takes the right end", fused["token_end"], ranges[1]["token_end"])
    checks.equal("merge_ranges rebuilds the text", fused["text"], TEXT[fused["start"]:fused["end"]])


def verify_windowing(checks: Checks, tokenizer) -> None:
    sentences = split_into_sentences(TEXT, "es")
    tokens = tokenize_document(TEXT, tokenizer)
    ranges = sentence_token_ranges(sentences, tokens)
    entities = [entity(20, 31)]

    windows = select_greedy_windows(ranges, tokens, entities, TEXT, 1000)

    checks.equal("a generous budget gives one window", len(windows), 1)
    checks.equal("that window covers the document", windows[0]["token_end"], len(tokens))

    tight = select_greedy_windows(ranges, tokens, entities, TEXT, 12)

    checks.check("a tight budget gives several windows", len(tight) > 1)
    checks.check(
        "no window exceeds the budget",
        all(w["token_end"] - w["token_start"] <= 12 for w in tight),
    )
    checks.check(
        "greedy windows do not overlap",
        all(a["token_end"] <= b["token_start"] for a, b in zip(tight, tight[1:])),
    )
    checks.equal(
        "greedy windows cover every token",
        sum(w["token_end"] - w["token_start"] for w in tight),
        len(tokens),
    )
    checks.check("greedy windows carry no core", all("core_token_start" not in w for w in tight))

    contexts = select_context_windows(ranges, tokens, entities, TEXT, 64, context_tokens=8)

    checks.check("context windows carry a core", all("core_token_start" in w for w in contexts))
    checks.check(
        "cores tile the document",
        [w["core_token_start"] for w in contexts][0] == 0,
    )
    checks.check(
        "cores are contiguous",
        all(
            a["core_token_end"] == b["core_token_start"] for a, b in zip(contexts, contexts[1:])
        ),
    )
    checks.check(
        "context extends beyond the core",
        any(w["token_start"] < w["core_token_start"] for w in contexts),
    )
    checks.check(
        "no context window exceeds the budget",
        all(w["token_end"] - w["token_start"] <= 64 for w in contexts),
    )

    no_context = select_context_windows(ranges, tokens, entities, TEXT, 64, context_tokens=0)
    checks.check(
        "zero context makes the window its core",
        all(
            w["token_start"] == w["core_token_start"] and w["token_end"] == w["core_token_end"]
            for w in no_context
        ),
    )

    oversized = split_oversized_sentence(ranges[0], tokens, entities, TEXT, 4)

    checks.check("an oversized sentence splits", len(oversized) > 1)
    checks.check(
        "each piece fits the budget",
        all(p["token_end"] - p["token_start"] <= 4 for p in oversized),
    )
    checks.equal(
        "pieces cover the whole sentence",
        (oversized[0]["token_start"], oversized[-1]["token_end"]),
        (ranges[0]["token_start"], ranges[0]["token_end"]),
    )

    window = build_window(0, 5, tokens, TEXT)
    checks.equal("build_window spans the token range", window["start"], tokens[0]["start"])
    checks.equal("build_window ends on the last token", window["end"], tokens[4]["end"])
    checks.equal("build_window text matches", window["text"], TEXT[window["start"]:window["end"]])


def verify_tagging(checks: Checks, tokenizer) -> None:
    tokens = tokenize_document(TEXT, tokenizer)
    entities = [entity(20, 31)]

    labelled = build_iob2_labels(tokens, entities, "DISEASE")
    tags = [token.get("label") for token in labelled]

    checks.equal("every token is labelled", len(labelled), len(tokens))
    checks.equal("B- appears exactly once", tags.count("B-DISEASE"), 1)
    checks.check("I- follows B-", tags.index("B-DISEASE") < len(tags))
    checks.check("tokens outside the entity are O", tags[0] == "O")
    checks.check(
        "the entity's characters are covered",
        all(
            token.get("label", "O") != "O"
            for token in labelled
            if 20 <= token["start"] < 31
        ),
    )

    checks.equal(
        "a non-target label is ignored",
        set(token.get("label") for token in build_iob2_labels(tokens, entities, "SYMPTOM")),
        {"O"},
    )
    checks.equal(
        "no entities gives all O",
        set(token.get("label") for token in build_iob2_labels(tokens, [], "DISEASE")),
        {"O"},
    )

    two = build_iob2_labels(tokens, [entity(20, 31), entity(58, 76)], "DISEASE")
    checks.equal(
        "two entities give two B- tags",
        [token.get("label") for token in two].count("B-DISEASE"),
        2,
    )

    vocabulary = build_label_vocabulary("DISEASE")

    checks.equal(
        "vocabulary is O/B/I",
        sorted(vocabulary["label2id"]),
        ["B-DISEASE", "I-DISEASE", "O"],
    )
    checks.equal("O is id zero", vocabulary["label2id"]["O"], 0)
    checks.equal(
        "id2label inverts label2id",
        {v: k for k, v in vocabulary["label2id"].items()},
        vocabulary["id2label"],
    )
    checks.equal(
        "an arbitrary label builds a vocabulary",
        sorted(build_label_vocabulary("GENE")["label2id"]),
        ["B-GENE", "I-GENE", "O"],
    )
    checks.raises("an empty label raises", ValueError, build_label_vocabulary, "  ")


def verify_rows(checks: Checks, tokenizers: dict) -> None:
    for name, tokenizer in tokenizers.items():
        prefix, suffix = special_token_template(tokenizer)
        round_trip = tokenizer.encode("a", add_special_tokens=True)
        probe = tokenizer.encode("a", add_special_tokens=False)

        checks.equal(f"{name}: template reassembles the encoding", [*prefix, *probe, *suffix], round_trip)
        checks.equal(
            f"{name}: budget accounts for the template",
            compute_max_content_length(prefix, suffix, 256),
            256 - len(prefix) - len(suffix),
        )

    tokenizer = tokenizers["bert"]
    prefix, suffix = special_token_template(tokenizer)
    tokens = tokenize_document(TEXT, tokenizer)
    vocabulary = build_label_vocabulary("DISEASE")
    labelled = build_iob2_labels(tokens[:20], [entity(20, 31)], "DISEASE")

    row = build_row(labelled, vocabulary["label2id"], prefix, suffix)

    checks.equal(
        "input_ids length includes the template",
        len(row["input_ids"]),
        len(labelled) + len(prefix) + len(suffix),
    )
    checks.equal("labels align with input_ids", len(row["labels"]), len(row["input_ids"]))
    checks.equal("attention mask is all ones", set(row["attention_mask"]), {1})
    checks.equal(
        "special tokens are masked out",
        row["labels"][:len(prefix)] + row["labels"][len(row["labels"]) - len(suffix):],
        [IGNORE_INDEX] * (len(prefix) + len(suffix)),
    )
    checks.equal("token_offsets exclude the template", len(row["token_offsets"]), len(labelled))
    checks.equal("word_ids exclude the template", len(row["word_ids"]), len(labelled))

    unlabelled = [{k: v for k, v in token.items() if k != "label"} for token in labelled]
    context_row = build_row(unlabelled, vocabulary["label2id"], prefix, suffix)

    checks.equal(
        "tokens with no label are masked",
        set(context_row["labels"]),
        {IGNORE_INDEX},
    )

    word_ids = [token["word_id"] for token in labelled]
    content_labels = row["labels"][len(prefix):len(row["labels"]) - len(suffix)]
    continuations = [
        index
        for index in range(1, len(word_ids))
        if word_ids[index] is not None and word_ids[index] == word_ids[index - 1]
    ]

    checks.check(
        "continuation subwords are masked",
        all(content_labels[index] == IGNORE_INDEX for index in continuations),
    )


def main() -> int:
    from transformers import AutoTokenizer

    checks = Checks("encoding leaves")

    tokenizers = {
        "bert": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "gpt2": AutoTokenizer.from_pretrained("gpt2"),
    }

    verify_overlaps(checks)
    verify_segmentation(checks, tokenizers["bert"])
    verify_windowing(checks, tokenizers["bert"])
    verify_tagging(checks, tokenizers["bert"])
    verify_rows(checks, tokenizers)

    return checks.report()


if __name__ == "__main__":
    run(main)
