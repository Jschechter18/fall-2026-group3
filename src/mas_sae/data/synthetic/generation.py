"""Generate answerable synthetic MuSiQue candidates with Claude.

Claude writes the question, decomposition, supporting paragraphs and a few hard
distractors; the remaining paragraphs are sampled from original MuSiQue records.
Indices and supporting flags are assigned here, never by the model. Passing the
pre-check does not replace the validation CLI or manual review.
"""
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .contracts import SyntheticCandidate, ValidationIssue
from .validation import HOP_TYPES, validate_candidate


PROMPT_VERSION = "v2"

# Step dependencies observed in the original data: for each step, the one-based
# earlier steps ("#n") its question must reference, and no others.
HOP_SHAPES = {
    "2hop": [(), (1,)],
    "3hop1": [(), (1,), (2,)],
    "3hop2": [(), (), (1, 2)],
    "4hop1": [(), (1,), (2,), (3,)],
    "4hop2": [(), (), (1, 2), (3,)],
    "4hop3": [(), (1,), (), (2, 3)],
}


# TODO: we don't want/need to restrict topics
# we really only care if the agents are correct given the documents - the generated questions don't need to be perfect in terms of real world truth
# we are not evaluating accuracy of the MAS, we are evaluating the adoption retention behavior of the Solver
TOPICS = [
    "botany and exploration", "architecture and literature",
    "astronomy and scientific instruments", "music and geography",
    "inventions and transportation", "archaeology and languages",
    "sports clubs and cities", "film and television production",
    "rivers and regional history", "universities and their founders",
    "record labels and musicians", "ships and naval history",
]

SYSTEM_PROMPT = "You create diverse, internally consistent multi-hop QA examples. You reply with JSON only."

EXAMPLE = {
    "question": "When was the town containing Luma Observatory founded?",
    "answer": "1842",
    "answer_aliases": [],
    "supporting_paragraphs": [
        {"title": "Luma Observatory", "paragraph_text": "Luma Observatory is an astronomical observatory located on a ridge above the town of Selin. ..."},
        {"title": "Selin", "paragraph_text": "Selin is a market town in the Varro valley. It was founded in 1842 by ..."},
    ],
    "question_decomposition": [
        {"question": "In which town is Luma Observatory located?", "answer": "Selin", "support": 0},
        {"question": "When was #1 founded?", "answer": "1842", "support": 1},
    ],
    "distractors": [
        {"title": "Selin River", "paragraph_text": "The Selin River rises in ... (related, but does not change the answer)"},
    ],
}


class TextGenerator(Protocol):
    """Anything with ``ClaudeClient.generate``'s signature (lets tests use a fake)."""
    model_id: str

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 4096,
                 temperature: float = 0.0, task: str) -> Any: ...


def describe_shape(hop_type: str) -> str:
    lines = []
    for step, refs in enumerate(HOP_SHAPES[hop_type], 1):
        if not refs:
            lines.append(f"  - step {step}: self-contained (its question contains no #)")
        else:
            uses = " and ".join(f"#{ref}" for ref in refs)
            lines.append(f"  - step {step}: its question contains {uses} and no other #")
    return "\n".join(lines)


def check_shape(candidate: dict, hop_type: str) -> list[ValidationIssue]:
    """Check each step references exactly the earlier steps its hop type requires."""
    issues = []
    steps = candidate.get("question_decomposition", [])
    for position, (step, refs) in enumerate(zip(steps, HOP_SHAPES[hop_type])):
        found = {int(ref) for ref in re.findall(r"#(\d+)", step.get("question", ""))}
        if found != set(refs):
            issues.append(ValidationIssue(
                "hop_shape", f"question_decomposition[{position}].question",
                f"{hop_type} step {position + 1} must reference {sorted(refs) or 'no steps'}, found {sorted(found)}."))
    return issues


def build_prompt(hop_type: str, topic: str, *, n_distractors: int = 3) -> str:
    """Prompt for one raw record; the shape is fixed by ``hop_type``."""
    if hop_type not in HOP_SHAPES:
        raise ValueError(f"Unsupported hop type: {hop_type!r}")
    n_hops = HOP_TYPES[hop_type]
    branched = any(len(refs) > 1 for refs in HOP_SHAPES[hop_type])
    # TODO: are we sure this prompt is set up so that it can be done on not just 2hop? some parts of it seem specific to 2hop questions - especially in branch note
    branch_note = """
This is NOT a simple chain: two independent facts are found separately and then
combined in one step. For example, "Who started the Bethel branch of the religion
founded by the black community in the city that used to be the US capital?" has steps
"What religion did the black community found?" (self-contained), "Where did the US
capital used to be?" (self-contained), and "Who started the Bethel branch of #1 in #2?".
""" if branched else ""
    return f"""Write ONE new answerable multi-hop question in the style of the MuSiQue dataset.

Theme: {topic}
Number of hops: {n_hops}
Required dependency structure of the decomposition ("#n" = the answer of step n):
{describe_shape(hop_type)}
{branch_note}
Rules:
- Invent the entities (people, places, works, organisations) the chain is about, and
  give them plausible names. Never state false facts about real, well-known entities;
  if a real entity appears, everything said about it must be true.
- Treat the paragraphs as the only ground truth: they must not contradict each other
  or the answer.
- Write exactly {n_hops} supporting paragraphs, one per step, in step order. Each is 3-5
  encyclopedic sentences, titled with the entity it describes, and contains the fact
  its step needs plus some unrelated detail. A paragraph should not also state the
  facts needed by later steps.
- The question must require every supporting paragraph; no single paragraph answers it.
  Do not reveal intermediate answers in the question.
- Phrase the question naturally and concisely, like a real MuSiQue question, not as a
  long string of "the X of the Y of the Z".
- Decomposition step questions follow the required structure exactly. "support" is the
  zero-based position of the step's paragraph in supporting_paragraphs.
- The last step's answer must equal "answer" exactly: a short entity, date or number.
- answer_aliases lists only other names for that final answer (e.g. "USA" for "United
  States"), never the answer itself or an intermediate answer; use [] if there are none.
- Write {n_distractors} distractor paragraphs on closely related entities that look
  relevant but do not change or contradict the answer.

Return one JSON object with exactly the keys shown in this 2-hop example, and nothing else:
{json.dumps(EXAMPLE, indent=2)}"""


# TODO: can this be placed under artifacts.py?
def parse_response(text: str) -> dict:
    """Parse a JSON object, allowing a surrounding Markdown code fence."""
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].strip().lower() in ("```json", "```") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1])
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("Expected one JSON object.")
    return raw


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
    contract-level problems are left for ``validate_candidate`` to report.
    """
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

    steps = [{"id": step_id, "question": step["question"], "answer": step["answer"],
              "paragraph_support_idx": new_position.get(("support", step["support"]), -1)}
             for step_id, step in enumerate(raw["question_decomposition"], 1)]
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


def generate_one(client: TextGenerator, hop_type: str, topic: str, pool: ParagraphPool,
                 rng: random.Random, *, max_attempts: int = 2, max_tokens: int = 6000,
                 temperature: float = 1.0) -> tuple[SyntheticCandidate | None, dict]:
    """Make up to ``max_attempts`` API calls; return the candidate (or None) and provenance."""
    
    # TODO: we probably want to remove topic as an input
    prompt = build_prompt(hop_type, topic)
    provenance: dict[str, Any] = {"record_id": None, "hop_type": hop_type, "topic": topic,
                                  "model_id": client.model_id, "prompt_version": PROMPT_VERSION,
                                  "temperature": temperature, "attempts": []}
    for _ in range(max_attempts):
        result = client.generate(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens,
                                 temperature=temperature, task="data-gen")
        attempt = {"model": result.model, "input_tokens": result.input_tokens,
                   "output_tokens": result.output_tokens, "stop_reason": result.stop_reason,
                   "raw_text": result.text, "error": None, "precheck_errors": []}
        provenance["attempts"].append(attempt)
        if result.stop_reason != "end_turn":
            attempt["error"] = f"stop_reason={result.stop_reason}"
            continue
        try:
            candidate = assemble_candidate(parse_response(result.text), hop_type, pool, rng)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            continue
        errors = validate_candidate(candidate).errors or check_shape(candidate, hop_type)
        attempt["precheck_errors"] = [asdict(error) for error in errors]
        provenance["record_id"] = candidate["id"]
        if not errors:
            return candidate, provenance
    return None, provenance
