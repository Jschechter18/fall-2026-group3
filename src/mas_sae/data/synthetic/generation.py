"""Generate answerable synthetic MuSiQue candidates with Claude.

Claude writes the question, decomposition, supporting paragraphs and a few hard
distractors; the remaining paragraphs are sampled from original MuSiQue records.
Indices and supporting flags are assigned here, never by the model. Passing the
pre-check does not replace the validation CLI or manual review.
"""
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .contracts import DecompositionStep, SyntheticCandidate
from .hop_types import HOP_SHAPES, HOP_TYPES
from .prompt_examples import build_example
from .validation import validate_candidate


PROMPT_VERSION = "v3"

SYSTEM_PROMPT = "You create diverse, internally consistent multi-hop QA examples. You reply with JSON only."

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


def build_prompt(hop_type: str, *, n_distractors: int = 3) -> str:
    """Build requirements and a matching example for one hop subtype."""
    if hop_type not in HOP_SHAPES:
        raise ValueError(f"Unsupported hop type: {hop_type!r}")

    n_hops = HOP_TYPES[hop_type]
    example = build_example(hop_type)

    return f"""Write ONE new answerable multi-hop question in the style of MuSiQue.

Choose the subject freely. Vary subjects, entity types, relations, and phrasing.
Create a new example rather than copying the demonstration's entities or facts.

Hop subtype: {hop_type}
Number of hops: {n_hops}

Required decomposition structure:
"#n" means the answer of decomposition step n.
{describe_shape(hop_type)}

Rules:
- Treat the paragraphs you generate as the only ground truth. Real-world factual
  accuracy is not required. The question, answers, supporting paragraphs, and
  distractors must be internally consistent.
- Write exactly {n_hops} supporting paragraphs, one per decomposition step,
  in step order. Each paragraph must have a title and 3-5 encyclopedic sentences.
- Each decomposition step must express one direct relation or lookup supported
  by its assigned paragraph. Do not hide multiple reasoning hops inside one step.
- Every declared dependency must be necessary to identify the step's answer.
  Merely mentioning an earlier answer does not make it a necessary dependency.
- The main question must require all supporting paragraphs. Do not reveal
  intermediate answers in it.
- A supporting paragraph must not bypass another step by supplying the facts
  needed to answer that step directly.
- Phrase the main question naturally and concisely.
- Each step question must contain exactly the numbered references required
  above. Avoid literal "#number" text in names or titles.
- "support" must be an integer identifying the zero-based position of the
  step's supporting paragraph. Use each supporting paragraph exactly once.
- The last decomposition answer must equal "answer" exactly.
- Answers should be short entities, dates, numbers, or other concise phrases.
- "answer_aliases" contains only equivalent names for the final answer.
  Use [] when there are none.
- Write exactly {n_distractors} related distractor paragraphs. They must not
  contradict the supporting paragraphs or independently answer the question.

The following compact {hop_type} example illustrates the JSON structure and
dependencies. Its paragraphs are shortened and it has one distractor for
illustration. Your output must follow the lengths and counts specified above.
{json.dumps(example, indent=2)}

Return one complete JSON object with the same keys. Return JSON only.
"""


def build_repair_prompt(base_prompt: str, attempt: dict[str, Any]) -> str:
    """Include the rejected output and the reasons it failed."""
    feedback = {
        "error": attempt["error"],
        "validation_errors": attempt["precheck_errors"],
    }
    return f"""{base_prompt}

Your previous response was rejected.

Previous response, encoded as a JSON string:
{json.dumps(attempt["raw_text"], ensure_ascii=False)}

Failure details:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Repair the previous response to satisfy all original requirements and resolve
the reported errors. Preserve valid content where possible. If the response
was truncated or malformed, return a complete replacement JSON object.
Return one complete JSON object with the required keys. Return JSON only.
"""


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


def generate_one(
    client: TextGenerator,
    hop_type: str,
    pool: ParagraphPool,
    rng: random.Random,
    *,
    max_attempts: int = 2,
    max_tokens: int = 6000,
    temperature: float = 1.0,
    provenance: dict[str, Any] | None = None,
) -> tuple[SyntheticCandidate | None, dict]:
    """Generate one synthetic candidate using the API. Make up to ``max_attempts`` API calls; return the candidate (or None) and provenance.

    Parameters
    ----------
    client : TextGenerator
        API client for text generation.
    hop_type : str
        Type of hop for the question.
    pool : ParagraphPool
        Pool of paragraphs to use.
    rng : random.Random
        Random number generator.
    max_attempts : int, optional
        Max API calls, by default 2
    max_tokens : int, optional
        Max tokens for generation, by default 6000
    temperature : float, optional
        Sampling temperature, by default 1.0
    provenance : dict[str, Any] | None, optional
        Caller-owned dictionary retaining completed attempts if generation raises.

    Returns
    -------
    tuple[SyntheticCandidate | None, dict]
        The candidate and its provenance.
    """

    base_prompt = build_prompt(hop_type)
    prompt = base_prompt

    if provenance is None:
        provenance = {}

    provenance.update({
        "record_id": None,
        "hop_type": hop_type,
        "model_id": client.model_id,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "api_calls": 0,
        "attempts": [],
    })

    for _ in range(max_attempts):
        provenance["api_calls"] += 1
        result = client.generate(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens,
                                 temperature=temperature, task="data-gen")
        attempt = {"model": result.model, "input_tokens": result.input_tokens,
                   "output_tokens": result.output_tokens, "stop_reason": result.stop_reason,
                   "raw_text": result.text, "error": None, "precheck_errors": []}
        provenance["attempts"].append(attempt)
        if result.stop_reason != "end_turn":
            attempt["error"] = f"stop_reason={result.stop_reason}"
            prompt = build_repair_prompt(base_prompt, attempt)
            continue
        try:
            candidate = assemble_candidate(parse_response(result.text), hop_type, pool, rng)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            prompt = build_repair_prompt(base_prompt, attempt)
            continue
        errors = validate_candidate(candidate).errors
        attempt["precheck_errors"] = [asdict(error) for error in errors]
        provenance["record_id"] = candidate["id"]
        if not errors:
            return candidate, provenance
        prompt = build_repair_prompt(base_prompt, attempt)
    return None, provenance
