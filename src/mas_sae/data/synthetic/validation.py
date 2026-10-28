"""Structural checks only: passing does not establish factual or semantic quality."""
import re
from typing import Any

from .contracts import ValidationIssue, ValidationResult


HOP_TYPES = {"2hop": 2, "3hop1": 3, "3hop2": 3, "4hop1": 4, "4hop2": 4, "4hop3": 4}


def validate_candidate(candidate: Any, *, expected_paragraphs: int | None = None) -> ValidationResult:
    """Validate the answerable profile. IDs use a MuSiQue hop prefix plus ``__``.

    Step references such as #1 are one-based positions, not step IDs.
    By default allow 1–20 paragraphs, matching the local dataset; callers may
    require an exact count. Extra fields are permitted. No input is mutated.
    """
    if expected_paragraphs is not None and (type(expected_paragraphs) is not int or expected_paragraphs < 1):
        raise ValueError("expected_paragraphs must be a positive integer or None")
    result = ValidationResult()

    def issue(code: str, path: str, message: str) -> None:
        result.errors.append(ValidationIssue(code, path, message))

    def text(obj: dict, key: str, path: str) -> bool:
        value = obj.get(key)
        if not isinstance(value, str) or not value.strip():
            issue("nonempty_string", path + key, "Expected a nonempty string.")
            return False
        return True

    if not isinstance(candidate, dict):
        issue("object", "", "Expected a JSON object.")
        return result
    for key in ("id", "question", "answer"):
        text(candidate, key, "")
    identifier = candidate.get("id")
    prefix, separator, suffix = identifier.partition("__") if isinstance(identifier, str) else ("", "", "")
    hops = HOP_TYPES.get(prefix)
    if hops is None or not separator or not suffix.strip():
        issue("hop_id", "id", "Expected a supported MuSiQue hop prefix and nonempty suffix separated by __.")
    if candidate.get("answerable") is not True:
        issue("answerable", "answerable", "This profile requires answerable=true.")
    aliases = candidate.get("answer_aliases")
    if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
        issue("aliases", "answer_aliases", "Expected a list of nonempty strings (or []).")

    paragraphs = candidate.get("paragraphs")
    supporting = set()
    if not isinstance(paragraphs, list):
        issue("list", "paragraphs", "Expected a list.")
        paragraphs = []
    if expected_paragraphs is not None and len(paragraphs) != expected_paragraphs:
        issue("paragraph_count", "paragraphs", f"Expected {expected_paragraphs} paragraphs.")
    elif expected_paragraphs is None and not 1 <= len(paragraphs) <= 20:
        issue("paragraph_count", "paragraphs", "Expected between 1 and 20 paragraphs.")
    for position, paragraph in enumerate(paragraphs):
        path = f"paragraphs[{position}]."
        if not isinstance(paragraph, dict):
            issue("object", path[:-1], "Expected a paragraph object.")
            continue
        idx = paragraph.get("idx")
        if type(idx) is not int or idx != position:
            issue("paragraph_index", path + "idx", "Index must equal its zero-based list position.")
        for key in ("title", "paragraph_text"):
            text(paragraph, key, path)
        if type(paragraph.get("is_supporting")) is not bool:
            issue("boolean", path + "is_supporting", "Expected a boolean.")
        elif paragraph["is_supporting"]:
            supporting.add(position)
    if hops is not None and len(supporting) != hops:
        issue("support_count", "paragraphs", f"Expected {hops} supporting paragraphs.")

    steps = candidate.get("question_decomposition")
    if not isinstance(steps, list):
        issue("list", "question_decomposition", "Expected a list.")
        steps = []
    if hops is not None and len(steps) != hops:
        issue("step_count", "question_decomposition", f"Expected {hops} steps.")
    step_ids = set()
    used_supports = set()
    used_steps = set()
    for position, step in enumerate(steps):
        path = f"question_decomposition[{position}]."
        if not isinstance(step, dict):
            issue("object", path[:-1], "Expected a decomposition step object.")
            continue
        step_id = step.get("id")
        if type(step_id) is not int or step_id in step_ids:
            issue("step_id", path + "id", "Expected a unique integer step ID.")
        else:
            step_ids.add(step_id)
        for key in ("question", "answer"):
            text(step, key, path)
        support = step.get("paragraph_support_idx")
        if type(support) is not int or support not in supporting:
            issue("support_reference", path + "paragraph_support_idx", "Must reference a supporting paragraph index.")
        else:
            used_supports.add(support)
        question = step.get("question")
        if isinstance(question, str):
            for reference in re.findall(r"#(\d+)", question):
                number = int(reference)
                if not 1 <= number <= position:
                    issue("step_reference", path + "question", "References must point to earlier one-based step positions.")
                else:
                    used_steps.add(number)
    if used_supports != supporting:
        issue("unused_support", "question_decomposition", "Every supporting paragraph must be referenced.")
    if set(range(1, len(steps))) - used_steps:
        issue("unused_step", "question_decomposition", "Every intermediate step must feed a later step.")
    if steps and isinstance(steps[-1], dict) and steps[-1].get("answer") != candidate.get("answer"):
        issue("final_answer", "answer", "Must equal the final decomposition answer.")
    return result
