"""Conservative lexical classifications, not a semantic equivalence judge.

Historical interaction_labels remain unchanged. Containment, malformed output,
recognized refusals, and uncertain answers are not treated as candidate adoption.
Unmatched answer-like strings are candidates, never proven semantic disagreement.
"""
import re
from mas_sae.evaluation.scoring import normalize_answer

NONANSWER = re.compile(
    r"\b(?:cannot|can't|could not|unable to|not enough|insufficient|"
    r"not provided|not mentioned|not specified|unknown|don't know|do not know|"
    r"no answer|unanswerable|cannot determine|"
    r"(?:do not|don't) have (?:enough|sufficient|the required) (?:information|context|evidence)|"
    r"(?:no|lack of) (?:information|context|evidence)|"
    r"(?:decline|refuse) to answer|not (?:possible|able) to (?:answer|determine)|"
    r"(?:text|context|passage|paragraphs?) (?:does|do) not (?:say|state|contain|identify))\b", re.I
)
UNCERTAIN = re.compile(r"\b(?:maybe|perhaps|possibly|might be|either)\b", re.I)


def answer_kind(value: object) -> str:
    """'missing' (not a string, or empty once normalized), 'nonanswer' ("n/a", "none", "null" or
    a refusal such as "cannot determine"), 'uncertain' ("maybe", "either", ...) or 'answer'."""
    if not isinstance(value, str) or not normalize_answer(value):
        return "missing"
    value = value.replace("’", "\'").replace("‘", "\'")
    if normalize_answer(value) in {"na", "n a", "none", "null", "not applicable"} or NONANSWER.search(value):
        return "nonanswer"
    if UNCERTAIN.search(value):
        return "uncertain"
    return "answer"


def textual_relation(left: object, right: object) -> str:
    """'exact', 'containment' (one normalized text inside the other), 'different', or 'unresolved'
    when either side is not a string or normalizes to empty. Symmetric."""
    if not isinstance(left, str) or not isinstance(right, str):
        return "unresolved"
    a, b = normalize_answer(left), normalize_answer(right)
    if not a or not b:
        return "unresolved"
    if a == b:
        return "exact"
    if a in b or b in a:
        return "containment"
    return "different"


def classify_behavior(a1: object, a2: object, advocacy: object, *, usable: bool = True) -> dict:
    """The lexical_v2 labels of one episode from A1, A2 and the Critic's advocated answer.

    With ``usable`` True (the collection passes ``not feedback.noncommittal``) and both A1 and the
    advocated answer of kind 'answer', ``solver_behavior`` is 'no_conflict' when the two match
    exactly; when they differ as text and A2 is an answer, it is 'retained_solver_a1',
    'adopted_critic' or 'third_answer_revision' (A2 differs from both). Everything else is
    'ambiguous', and ``direct_critic_adoption`` is None outside those three outcomes.
    """
    raw = textual_relation(advocacy, a1)
    relation = "ambiguous_or_unresolved"
    outcome = "ambiguous"
    if answer_kind(advocacy) == "nonanswer":
        relation = "refusal_or_nonanswer"
    elif usable and answer_kind(advocacy) == answer_kind(a1) == "answer":
        if raw == "exact":
            relation, outcome = "same_position", "no_conflict"
        elif raw == "different":
            relation = "different_nonrefusal_candidate"
            own, critic = textual_relation(a2, a1), textual_relation(a2, advocacy)
            if answer_kind(a2) == "answer":
                if own == "exact":
                    outcome = "retained_solver_a1"
                elif critic == "exact":
                    outcome = "adopted_critic"
                elif own == critic == "different":
                    outcome = "third_answer_revision"
    return {
        "behavior_schema_version": "lexical_v2",
        "behavior_matching_basis": "normalized_text_not_semantic",
        "solver_copied_nonanswer": (
            textual_relation(a2, advocacy) == "exact"
            if relation == "refusal_or_nonanswer" and answer_kind(a2) != "missing"
            else None
        ),
        "critic_textual_relation": raw,
        "critic_position_relation": relation,
        "solver_behavior": outcome,
        "direct_critic_adoption": (
            outcome == "adopted_critic"
            if outcome in {"adopted_critic", "retained_solver_a1", "third_answer_revision"}
            else None
        ),
    }
