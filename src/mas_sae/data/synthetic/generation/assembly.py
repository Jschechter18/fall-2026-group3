"""Build candidates and sample filler paragraphs from original MuSiQue."""
import json
import random
from pathlib import Path
from uuid import uuid4

from ..contracts import DecompositionStep, SyntheticCandidate


def _validate_raw_answer_fields(raw: dict) -> None:
    """Reject malformed answer fields before sampling or alias normalization."""
    def require_text(value: object, field: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field}: expected a nonempty string; got {type(value).__name__}"
            )

    require_text(raw.get("answer"), "answer")
    aliases = raw.get("answer_aliases", [])
    if not isinstance(aliases, list):
        raise ValueError(
            f"answer_aliases: expected a list of strings; got {type(aliases).__name__}"
        )
    for position, alias in enumerate(aliases):
        require_text(alias, f"answer_aliases[{position}]")

    steps = raw.get("question_decomposition")
    if not isinstance(steps, list):
        raise ValueError("question_decomposition: expected a list")
    for position, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"question_decomposition[{position}]: expected an object")
        require_text(step.get("answer"), f"question_decomposition[{position}].answer")


class ParagraphPool:
    """Original MuSiQue paragraphs used as filler distractors, unique by title."""
    def __init__(self, paragraphs: list[dict]):
        self.paragraphs = paragraphs

    @classmethod
    def from_jsonl(cls, path: str | Path, *, max_records: int | None = 5000) -> "ParagraphPool":
        by_title: dict[str, dict] = {}
        with Path(path).open(encoding="utf-8") as stream:
            for count, line in enumerate(stream):
                if max_records is not None and count >= max_records:
                    break
                for paragraph in json.loads(line)["paragraphs"]:
                    by_title.setdefault(paragraph["title"], {
                        "title": paragraph["title"], "paragraph_text": paragraph["paragraph_text"]})
        return cls(list(by_title.values()))

    def sample(self, k: int, rng: random.Random, *, exclude_titles: set[str] = frozenset(),
               exclude_text: str = "") -> list[dict]:
        """Sample k paragraphs, skipping excluded titles and any text mentioning ``exclude_text``."""
        needle = exclude_text.casefold()
        chosen: list[dict] = []
        for paragraph in rng.sample(self.paragraphs, len(self.paragraphs)):
            if len(chosen) == k:
                break
            if paragraph["title"] in exclude_titles:
                continue
            if needle and needle in paragraph["paragraph_text"].casefold():
                continue
            chosen.append(dict(paragraph))
        if len(chosen) < k:
            raise ValueError(f"Paragraph pool too small: needed {k}, found {len(chosen)}")
        return chosen


def assemble_candidate(raw: dict, hop_type: str, pool: ParagraphPool, rng: random.Random, *,
                       total_paragraphs: int = 20) -> SyntheticCandidate:
    """Turn a raw model response into the ``SyntheticCandidate`` contract.

    Raises ``KeyError``/``TypeError``/``ValueError`` when the response is missing pieces;
    malformed answer fields are rejected before sampling or normalization.
    Remaining contract-level problems are reported by ``validate_candidate``.
    """
    _validate_raw_answer_fields(raw)
    supporting = [{"title": p["title"], "paragraph_text": p["paragraph_text"], "is_supporting": True}
                  for p in raw["supporting_paragraphs"]]
    distractors = [{"title": p["title"], "paragraph_text": p["paragraph_text"], "is_supporting": False}
                   for p in raw.get("distractors", [])]
    titles = {p["title"] for p in supporting + distractors}
    n_fill = total_paragraphs - len(supporting) - len(distractors)
    fillers = [{**p, "is_supporting": False}
               for p in pool.sample(max(n_fill, 0), rng, exclude_titles=titles, exclude_text=raw["answer"])]

    # Shuffle, then assign idx; remember where each supporting paragraph landed.
    tagged = [(("support", i), p) for i, p in enumerate(supporting)] + \
             [(("other", i), p) for i, p in enumerate(distractors + fillers)]
    rng.shuffle(tagged)
    paragraphs = []
    new_position = {}
    for idx, (tag, paragraph) in enumerate(tagged):
        paragraphs.append({"idx": idx, "title": paragraph["title"],
                           "paragraph_text": paragraph["paragraph_text"],
                           "is_supporting": paragraph["is_supporting"]})
        new_position[tag] = idx

    steps: list[DecompositionStep] = []
    for step_id, step in enumerate(raw["question_decomposition"], 1):
        support = step["support"]

        if type(support) is not int:
            raise ValueError(
                f"Step {step_id}: support must be an integer, "
                f"got {support!r}."
            )

        if not 0 <= support < len(supporting):
            raise ValueError(
                f"Step {step_id}: support must be between 0 and "
                f"{len(supporting) - 1}, got {support}."
            )

        steps.append({
            "id": step_id,
            "question": step["question"],
            "answer": step["answer"],
            "paragraph_support_idx": new_position[("support", support)],
        })

    return {
        "id": f"{hop_type}__synthetic_{uuid4().hex}",
        "question": raw["question"],
        "answer": raw["answer"],
        "answer_aliases": [alias for alias in raw.get("answer_aliases", [])
                           if not isinstance(alias, str)
                           or alias.strip().casefold() != raw["answer"].strip().casefold()],
        "answerable": True,
        "paragraphs": paragraphs,
        "question_decomposition": steps,
    }
