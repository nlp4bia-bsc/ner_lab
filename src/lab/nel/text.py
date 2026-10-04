"""Finding dictionary terms in a directory of plain-text documents."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from lab.nel.schemas import Mention


def mentions_from_text_dictionary(
    text_dictionary: list[str],
    texts_dir: str | Path,
    label: str | None = None,
    code: str | None = None,
) -> list[Mention]:
    """
    Every case-insensitive occurrence of a dictionary term in the `.txt` files of `texts_dir`.

    Occurrences may overlap, and each keeps the casing the document has.
    """
    terms = [str(term).strip() for term in text_dictionary if str(term).strip()]
    mentions: list[Mention] = []

    for path in Path(texts_dir).glob("*.txt"):
        content = path.read_text(encoding="utf-8")
        lowered_content = content.lower()

        for term in terms:
            for start in _occurrences(lowered_content, term.lower()):
                end = start + len(term)
                mentions.append(
                    Mention(
                        filename=path.stem,
                        label=label,
                        start_span=start,
                        end_span=end,
                        text=content[start:end],
                        code=code,
                    )
                )

    return mentions


def _occurrences(content: str, term: str) -> Iterator[int]:
    start = content.find(term)

    while start != -1:
        yield start
        start = content.find(term, start + 1)
