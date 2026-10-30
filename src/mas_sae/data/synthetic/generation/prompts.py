"""Generation requirements, subtype descriptions, and repair prompts."""
import json
from typing import Any

from ..hop_types import HOP_SHAPES, HOP_TYPES
from .examples import build_example


PROMPT_VERSION = "v3"

SYSTEM_PROMPT = "You create diverse, internally consistent multi-hop QA examples. You reply with JSON only."

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
