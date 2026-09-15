from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


_ARTICLES = re.compile(r"\b(?:a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(value: object) -> str:
    text = str(value).casefold()
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_matches(
    candidate: str | None,
    gold: str,
    aliases: Iterable[str] = (),
) -> bool:
    if candidate is None:
        return False

    if isinstance(aliases, str):
        aliases = (aliases,)

    targets = {normalize_answer(gold)}
    targets.update(normalize_answer(alias) for alias in aliases)
    targets.discard("")

    normalized_candidate = normalize_answer(candidate)

    return bool(normalized_candidate) and normalized_candidate in targets
