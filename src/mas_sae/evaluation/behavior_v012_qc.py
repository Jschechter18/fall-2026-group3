"""Behavior v0.1.2: an experimental QC overlay on the frozen v0.1.1 rules.

The overlay reads the v0.1.1 label of an episode and overrides two of its inputs
when the text is explicit about them:

    feedback_type  "answer" or "refusal" becomes "premise_rejection", and "answer"
                   becomes "refusal", when the Critic's advocated answer says so
                   in so many words. Only the structured ``critic_advocated_answer``
                   is searched, never the explanation, which can describe a
                   rejected answer, a counterfactual or the original A1.
    a2_kind        becomes "nonanswer" when A2 is an explicit non-answer.

Everything downstream (conflict status, Solver response, exclusions, targets)
is then derived by ``behavior_v01.assemble_labels`` exactly as v0.1.1 does it.
An "unresolved" Critic is never overridden. Every row records which rules fired
(``qc_rules``) and the v0.1.1 values it started from (``base_feedback_type``,
``base_a2_kind``).

Status: experimental. The patterns were written against the 559-row human QC
sample, so agreement measured on that sample is in-sample and optimistic. They
have not been validated on held-out episodes, and over the full production run
they change about 1,575 of 22,332 labels. They deliberately do not modify the
frozen ``EXTRA_NONANSWER`` and ``PREMISE`` lexicons of v0.1.1, which they
partly overlap. Do not build a production label package from this module
before independent validation and team sign-off.
"""
import re

from mas_sae.evaluation import behavior_v01

RULE_VERSION = "behavior_v0.1.2_qc_candidate"

PREMISE_PATTERNS = {
    "question_explicitly_erroneous": re.compile(
        r"\b(?:the|this)\s+question\s+(?:contains?|has|is|appears|seems)\s+"
        r"(?:a|an|the)?\s*(?:factual|logical|historical|fundamental|incorrect|false|"
        r"flawed|invalid|misleading|mistaken|inconsistent|unrelated|based\s+on|"
        r"not\s+valid|inaccurate)\b", re.I),
    "explicitly_no_connection": re.compile(
        r"\b(?:there is|there was|there can be)\s+no\s+"
        r"(?:direct|valid|historical|factual|meaningful|logical|known|clear|"
        r"established)?\s*(?:connection|relationship|link|association)\b", re.I),
    "question_explicitly_invalid": re.compile(
        r"\b(?:question|premise)\s+(?:is|was)\s+"
        r"(?:invalid|incorrect|false|flawed|based\s+on\s+a\s+"
        r"(?:mistake|misunderstanding))\b", re.I),
}

REFUSAL_PATTERNS = {
    "source_explicitly_does_not_identify": re.compile(
        r"\b(?:provided|given|source|these)\s+"
        r"(?:paragraphs?|passages?|texts?|sources?|context|information)\s+"
        r"(?:do|does|did)\s+not\s+"
        r"(?:mention|state|specify|provide|identify|indicate|contain|include|"
        r"show|say|name|reveal)\b", re.I),
    "entity_explicitly_unidentified": re.compile(
        r"\b(?:is|are|was|were)\s+not\s+"
        r"(?:explicitly|directly|clearly|specifically)\s+"
        r"(?:mentioned|specified|identified|stated|provided|named|recorded|"
        r"available|given)\b", re.I),
}

# An explicit non-answer at the start of A2, not an answer that states a negative
# fact. Broad "did not" / "there is no" openings are deliberately left out: they
# can be genuine answers and need a human reading. Known limit: a real answer
# that begins with one of these words ("Unknown Soldier") is treated as a non-answer.
A2_NONANSWER_PATTERNS = {
    "simple_nonanswer": re.compile(
        r"^\s*(?:none|n/?a|unknown|cannot\s+be\s+determined|"
        r"not\s+(?:mentioned|specified|available|known))\b", re.I),
    "no_such_or_valid": re.compile(
        r"^\s*(?:there\s+(?:is|was|are|were)\s+)?"
        r"no\s+(?:such|valid|clear|direct|known)\b", re.I),
    "no_answer_or_evidence": re.compile(
        r"^\s*(?:there\s+(?:is|was)\s+)?"
        r"no\s+(?:answer|record|evidence|information|specific|date|way)\b", re.I),
    "question_invalid": re.compile(
        r"^\s*(?:the\s+)?(?:question|premise)\s+(?:is|contains|has)\s+"
        r"(?:invalid|incorrect|flawed|false|based|"
        r"a\s+(?:factual|logical|historical)\s+(?:error|inaccuracy))\b", re.I),
    "information_explicitly_unavailable": re.compile(
        r"^\s*(?:the\s+)?(?:information|answer|date|name|location|"
        r"identity|founder|birthplace|university|time|individual|person|reason)\s+"
        r"(?:is|was|are|were)\s+not\s+"
        r"(?:available|provided|mentioned|stated|given|specified|clear|found|"
        r"known|identifiable)\b", re.I),
    "no_record": re.compile(
        r"^\s*there\s+(?:is|was)\s+no\s+"
        r"(?:record|evidence|mention)\b", re.I),
}


def matches(text, patterns):
    """Names of the patterns that match ``text``; none for a missing text."""
    if not isinstance(text, str):
        return []
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def propose_feedback(current, advocated_answer):
    """Return (feedback_type, rules fired) for the Critic's advocated answer.

    "unresolved" is kept as is. Explicit premise rejection overrides "answer" or
    "refusal"; explicit refusal overrides "answer" only.
    """
    if current not in ("answer", "refusal", "premise_rejection"):
        return current, []
    premise = matches(advocated_answer, PREMISE_PATTERNS)
    if premise and current in ("answer", "refusal"):
        return "premise_rejection", ["premise:" + name for name in premise]
    refusal = matches(advocated_answer, REFUSAL_PATTERNS)
    if refusal and current == "answer":
        return "refusal", ["refusal:" + name for name in refusal]
    return current, []


def explicit_a2_nonanswer(text):
    """Names of the A2 non-answer patterns that match."""
    return matches(text, A2_NONANSWER_PATTERNS)


def classify_candidate_v012(row):
    """Label one episode under v0.1.2. Returns a new dict; the row and v0.1.1 are untouched.

    The result has every v0.1.1 field plus ``qc_rules`` (the overlay rules that
    fired, premise/refusal rules before A2 rules), ``base_feedback_type`` and
    ``base_a2_kind`` (the v0.1.1 values).
    """
    base = behavior_v01.classify_candidate(row)
    feedback_type, rules = base["feedback_type"], []
    if base["critic_unresolved_flag"] is False:
        feedback_type, rules = propose_feedback(feedback_type, row.get("critic_advocated_answer"))
    a2_kind = base["a2_kind"]
    a2_rules = explicit_a2_nonanswer(row.get("solver_attempt_2"))
    if a2_rules:
        a2_kind = "nonanswer"
        rules = rules + ["a2:" + name for name in a2_rules]

    label = behavior_v01.assemble_labels(
        feedback_type, base["a1_kind"], a2_kind, base["critic_kind"],
        base["a1_vs_critic"], base["a2_vs_a1"], base["a2_vs_critic"],
        base["critic_unresolved_flag"],
        base["critic_premise_rejection_flag"] or feedback_type == "premise_rejection",
        base["critic_refusal_flag"] or feedback_type == "refusal",
        rule_version=RULE_VERSION)
    label["qc_rules"] = rules
    label["base_feedback_type"] = base["feedback_type"]
    label["base_a2_kind"] = base["a2_kind"]
    return label
