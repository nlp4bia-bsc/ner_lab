"""Sentence segmentation, routed by language to pysbd, Punkt or ICU."""

from __future__ import annotations

import bisect
from functools import cache
from importlib import metadata, resources
from typing import TYPE_CHECKING, Literal

import icu4py
import pysbd
from icu4py.breakers import SentenceBreaker

if TYPE_CHECKING:
    from nltk.tokenize.punkt import PunktSentenceTokenizer

SentenceSegmenter = Literal["pysbd", "punkt", "icu"]

PYSBD_LANGUAGES = (
    "am", "ar", "bg", "da", "de", "el", "en", "es", "fa", "fr", "hi", "hy",
    "it", "ja", "kk", "mr", "my", "nl", "pl", "ru", "sk", "ur", "zh",
)  # fmt: skip

PUNKT_LANGUAGES = {
    "cs": "czech",
    "et": "estonian",
    "fi": "finnish",
    "ml": "malayalam",
    "no": "norwegian",
    "pt": "portuguese",
    "sl": "slovene",
    "sv": "swedish",
    "tr": "turkish",
}

ISO_639_1_LANGUAGES = frozenset(
    """
    aa ab ae af ak am an ar as av ay az ba be bg bi bm bn bo br bs ca ce ch co cr cs cu cv cy
    da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht
    hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky
    la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny
    oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss
    st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo
    za zh zu
    """.split()
)

PUNKT_DATA_DIR = "data/punkt_tab"


def sentence_segmenter(language: str) -> SentenceSegmenter:
    """
    The segmenter `split_into_sentences` uses for `language`, an ISO 639-1 code.

    pysbd where it has the language, Punkt for the European languages it lacks, and ICU's
    sentence break rules for every other ISO 639-1 code. The choice is a fixed table, never
    what happens to be installed, so one language is always segmented the same way.
    """
    if language in PYSBD_LANGUAGES:
        return "pysbd"

    if language in PUNKT_LANGUAGES:
        return "punkt"

    if language in ISO_639_1_LANGUAGES:
        return "icu"

    raise ValueError(
        f"language {language!r} is not an ISO 639-1 code. Pass the two-letter code, "
        "such as 'es', 'pt' or 'ca'."
    )


def describe_sentence_segmenter(language: str) -> dict:
    """
    The segmenter used for `language` and its version, as plain data.

    Recorded beside anything windowed or counted by sentence, since a different segmenter
    moves every window boundary. The version is pysbd's, nltk's for Punkt (whose models
    ship inside this library), or the ICU version icu4py embeds.
    """
    segmenter = sentence_segmenter(language)
    versions = {
        "pysbd": metadata.version("pysbd"),
        "punkt": metadata.version("nltk"),
        "icu": icu4py.icu_version,
    }

    return {"segmenter": segmenter, "version": versions[segmenter]}


def split_into_sentences(text: str, language: str) -> list[dict]:
    """
    Split a document into sentence spans, in document order, with the segmenter
    `sentence_segmenter` picks for `language`.

    Spans are contiguous from the first non-whitespace character to the end of the
    document: trailing whitespace belongs to the sentence before it, and a document that is
    whitespace throughout has no sentences. Every segmenter is brought to this shape, the
    one pysbd returns.
    """
    segmenter = sentence_segmenter(language)

    if segmenter == "pysbd":
        return _pysbd_sentences(text, language)

    if segmenter == "punkt":
        starts = [start for start, _ in _punkt_tokenizer(language).span_tokenize(text)]
    else:
        starts = _icu_starts(text, language)

    return _spans_from_starts(text, starts)


def _pysbd_sentences(text: str, language: str) -> list[dict]:
    segmenter = pysbd.Segmenter(language=language, clean=False, char_span=True)

    return [
        {"start": span.start, "end": span.end, "text": span.sent}
        for span in segmenter.segment(text)
    ]


@cache
def _punkt_tokenizer(language: str) -> PunktSentenceTokenizer:
    from nltk.tabdata import PunktDecoder
    from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

    model_dir = resources.files("lab.core") / PUNKT_DATA_DIR / PUNKT_LANGUAGES[language]
    decoder = PunktDecoder()
    parameters = PunktParameters()

    with (model_dir / "collocations.tab").open(encoding="utf-8") as file:
        parameters.collocations = set(decoder.tab2tups(file))

    with (model_dir / "sent_starters.txt").open(encoding="utf-8") as file:
        parameters.sent_starters = decoder.txt2set(file)

    with (model_dir / "abbrev_types.txt").open(encoding="utf-8") as file:
        parameters.abbrev_types = decoder.txt2set(file)

    with (model_dir / "ortho_context.tab").open(encoding="utf-8") as file:
        parameters.ortho_context = decoder.tab2intdict(file)

    return PunktSentenceTokenizer(parameters)


def _icu_starts(text: str, language: str) -> list[int]:
    utf16_offsets = [0]

    for character in text:
        utf16_offsets.append(utf16_offsets[-1] + (2 if ord(character) > 0xFFFF else 1))

    return [
        bisect.bisect_left(utf16_offsets, start)
        for start, _ in SentenceBreaker(text, language).segments()
    ]


def _spans_from_starts(text: str, starts: list[int]) -> list[dict]:
    content_starts = []

    for start in starts:
        while start < len(text) and text[start].isspace():
            start += 1

        if start < len(text) and (not content_starts or start > content_starts[-1]):
            content_starts.append(start)

    ends = [*content_starts[1:], len(text)]

    return [
        {"start": start, "end": end, "text": text[start:end]}
        for start, end in zip(content_starts, ends)
    ]
