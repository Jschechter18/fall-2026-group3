"""Whether the Critic feedback a Solver received states one coherent position.

Behavior labels compare A1, the advocated answer and A2 as text, and never read
the Critic's verdict or explanation. Human QC (issue #115) found that the
Critic often advocated one answer while its explanation argued for A1, so a
Solver keeping A1 was following the feedback, not rejecting it.

This module reads only what the Critic delivered (verdict, advocated answer,
explanation), the Critic's raw output and telemetry, and A1. It never reads A2
or any correctness field, so a coherence decision cannot depend on the outcome
it is later used to filter.

Decisions:
  consistent    the delivered fields agree with each other
  inconsistent  the delivered fields contradict each other
  uncertain     a warning sign without a direct contradiction
  unusable      no verdict or no advocated answer reached the Solver
"""
import hashlib
import json
import re

from mas_sae.agents.critic import Critic, first_json_object_span
from mas_sae.evaluation.behavior import textual_relation
from mas_sae.evaluation.scoring import normalize_answer

# The compare prompt labels the Critic's blind answer "Your own answer", and the
# Critic then writes about it as if it were someone else's.
ROLE_CONFUSION = re.compile(
    r"\b(?:user'?s|your|my|the original|original)\s+own\s+answer\b"
    r"|\boriginal answer\b",
    re.I,
)
# The explanation endorses the proposed answer (A1).
ENDORSES_PROPOSED = re.compile(
    r"\bproposed answer (?:is|was) (?:actually |factually |indeed )?(?:correct|accurate|right)\b"
    r"|\bverdict (?:is|should be) ['\"]?agree\b"
    r"|\b(?:we|i) (?:must |should )?agree with the proposed answer\b",
    re.I,
)
# The explanation rejects the proposed answer (A1).
REJECTS_PROPOSED = re.compile(
    r"\bproposed answer (?:is|was) (?!not (?:incorrect|wrong|inaccurate)\b)"
    r"(?:\w+ )?(?:incorrect|wrong|not correct|inaccurate)\b"
    r"|\bproposed answer (?:does not|doesn't) (?:correctly )?answer\b",
    re.I,
)
# Clauses that state the Critic's position: an explicit "correct answer is ..."
# anywhere, or the final "therefore/thus/hence" sentence.
POSITION_CLAUSE = re.compile(
    r"\b(?:correct (?:advocated )?answer|advocated answer|answer) (?:is|should be|would be)\b[^.!?]*",
    re.I,
)
# An explicit "correct answer is ..." claim; under an "agree" verdict it must
# name the agreed answer.
STATED_ANSWER = re.compile(r"\bcorrect answer (?:is|should be|would be)\b[^.!?]*", re.I)
CONCLUSION_CUE = re.compile(r"\b(?:therefore|thus|hence)\b", re.I)
# A mention of A1 right after one of these is the Critic arguing against it.
NEGATION_BEFORE = re.compile(r"(?:\bnot|\brather than|\binstead of|\bincorrect|\bwrong)\W+(?:\w+\W+){0,3}$", re.I)
# ... or right before one of these ("Philippines is not supported").
NEGATION_AFTER = re.compile(
    r"^\W*(?:is|was|are)\s+(?:not|incorrect|wrong|inaccurate|unsupported|false)\b", re.I)
# Text generated after the delivered JSON that is long enough to carry a claim.
MIN_TRAILING_CHARS = 20
DECISIONS = ("consistent", "inconsistent", "uncertain", "unusable")


def _pattern(p):
    return [p.pattern, p.flags]


INCONSISTENT_IF = (
    "verdict_disagree_advocates_a1",
    "disagree_but_endorses_proposed",
    "agree_but_rejects_proposed",
    "position_names_a1_not_advocated",
    "agree_but_states_other_answer",
)
UNCERTAIN_IF = (
    "verdict_agree_advocates_other",
    "position_names_both",
    "role_confusion",
    "trailing_text_after_json",
    "critic_reached_token_budget",
)
RULES = {
    "name": "feedback_coherence",
    "role_confusion": _pattern(ROLE_CONFUSION),
    "endorses_proposed": _pattern(ENDORSES_PROPOSED),
    "rejects_proposed": _pattern(REJECTS_PROPOSED),
    "position_clause": _pattern(POSITION_CLAUSE),
    "stated_answer": _pattern(STATED_ANSWER),
    "conclusion_cue": _pattern(CONCLUSION_CUE),
    "negation_before": _pattern(NEGATION_BEFORE),
    "negation_after": _pattern(NEGATION_AFTER),
    "min_trailing_chars": MIN_TRAILING_CHARS,
    "inconsistent_if": list(INCONSISTENT_IF),
    "uncertain_if": list(UNCERTAIN_IF),
}


def rules_sha256():
    return hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()


def critic_reached_token_budget(telemetry):
    """True if any recorded Critic review generation stopped at its token budget."""
    reviews = (telemetry or {}).get("critic_review") or []
    if isinstance(reviews, dict):
        reviews = [reviews]
    return any(bool(r.get("reached_token_budget")) for r in reviews)


def _affirms(text, answer):
    """True if ``answer`` appears in ``text`` other than next to a negation."""
    a = normalize_answer(answer) if isinstance(answer, str) else ""
    if not a:
        return False
    norm = normalize_answer(text)
    start = norm.find(a)
    while start != -1:
        if not (NEGATION_BEFORE.search(norm[:start]) or NEGATION_AFTER.search(norm[start + len(a):])):
            return True
        start = norm.find(a, start + 1)
    return False


def position_clauses(explanation):
    """The explanation's statements of the Critic's position."""
    clauses = [m.group(0) for m in POSITION_CLAUSE.finditer(explanation)]
    cues = list(CONCLUSION_CUE.finditer(explanation))
    if cues:
        clauses.append(re.split(r"(?<=[.!?])\s", explanation[cues[-1].start():], maxsplit=1)[0])
    return clauses


def assess(raw_output, a1, *, telemetry=None):
    """Assess the feedback that reached the Solver.

    ``raw_output`` is the Critic's raw review output, parsed with the same
    ``Critic.parse_natural`` the collection used, so the fields assessed are
    the ones the Solver saw. Returns a flat dict of signals plus ``coherence``
    and ``coherence_reasons``.
    """
    raw_output = raw_output or ""
    feedback = Critic.parse_natural(raw_output)
    span = first_json_object_span(raw_output)
    trailing = raw_output[span[1]:].strip() if span else ""
    explanation = feedback.explanation
    advocated = feedback.advocated_answer
    clauses = position_clauses(explanation)
    advocated_vs_a1 = textual_relation(advocated, a1)
    signals = {
        "delivered_verdict": feedback.verdict,
        "delivered_advocated_answer": advocated,
        "delivered_explanation": explanation,
        "parse_found_json": span is not None,
        "delivered_noncommittal": feedback.noncommittal,
        "critic_reached_token_budget": critic_reached_token_budget(telemetry),
        "trailing_text_after_json": len(trailing) >= MIN_TRAILING_CHARS,
        "trailing_text": trailing,
        "role_confusion": bool(ROLE_CONFUSION.search(explanation)),
        "endorses_proposed": bool(ENDORSES_PROPOSED.search(explanation)),
        "rejects_proposed": bool(REJECTS_PROPOSED.search(explanation)),
        "advocated_vs_a1": advocated_vs_a1,
        "position_names_a1": any(_affirms(c, a1) for c in clauses),
        "position_names_advocated": any(_affirms(c, advocated) for c in clauses),
        "states_other_answer": any(
            not _affirms(m.group(0), a1) and not _affirms(m.group(0), advocated)
            for m in STATED_ANSWER.finditer(explanation)
        ),
    }

    if feedback.noncommittal:
        reasons, decision = ["noncommittal"], "unusable"
    else:
        disagree = feedback.verdict == "disagree"
        agree = feedback.verdict == "agree"
        found = {
            "verdict_disagree_advocates_a1": disagree and advocated_vs_a1 == "exact",
            "verdict_agree_advocates_other": agree and advocated_vs_a1 == "different",
            # A "disagree" that extracts a shorter answer from a longer A1
            # (containment) may still call A1 accurate, so require a clear difference.
            "disagree_but_endorses_proposed": (
                disagree and advocated_vs_a1 == "different" and signals["endorses_proposed"]
            ),
            "agree_but_rejects_proposed": agree and signals["rejects_proposed"],
            "agree_but_states_other_answer": agree and signals["states_other_answer"],
            # Containment between A1 and the advocated answer makes a mention
            # ambiguous, so the stated position is only used when they differ.
            "position_names_a1_not_advocated": (
                disagree and advocated_vs_a1 == "different"
                and signals["position_names_a1"] and not signals["position_names_advocated"]
            ),
            "position_names_both": (
                disagree and advocated_vs_a1 == "different"
                and signals["position_names_a1"] and signals["position_names_advocated"]
            ),
            "role_confusion": signals["role_confusion"],
            "trailing_text_after_json": signals["trailing_text_after_json"],
            "critic_reached_token_budget": signals["critic_reached_token_budget"],
        }
        reasons = [name for name in INCONSISTENT_IF + UNCERTAIN_IF if found[name]]
        if any(found[name] for name in INCONSISTENT_IF):
            decision = "inconsistent"
        elif reasons:
            decision = "uncertain"
        else:
            decision = "consistent"

    signals["coherence"] = decision
    signals["coherence_reasons"] = reasons
    return signals
