"""Behavior labels: how the Solver responds to Critic feedback, under a label policy.

Each episode has a first answer (A1), the Critic's feedback, and a second
answer (A2). This module gives every episode two separate labels:

    feedback_type    what the Critic gave
    solver_response  what the Solver did in A2

The rules compare normalized text. They do not understand meaning, so the
labels are candidates until people have checked them. They are stored next to
the original lexical_v2 labels and never replace them.

A ``LabelPolicy`` names the rule set that labelled an episode. ``V011`` is the
first, frozen rule set and the default. ``V012_QC`` adds an overlay: when the
Critic's advocated answer or A2 says in so many words that it is a premise
rejection, a refusal or a non-answer, the overlay overrides that one input and
the rest is derived exactly as before. Every policy writes its version into
``rule_version``; no rule reads the version back. A new rule set is a new
``LabelPolicy`` with its own lexicons, not a new function.
"""
from collections import Counter
from dataclasses import dataclass, field
import csv
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping

from mas_sae.evaluation import behavior

SOLVER_RESPONSES = (
    "adopted_critic",           # A2 is the Critic's answer
    "retained_a1",              # A2 is the same as A1
    "third_answer",             # A2 is different from both
    "unchanged_same_position",  # A1, the Critic and A2 all agree
    "non_answer",               # A2 is a refusal such as "cannot determine"
    "produced_answer",          # the Critic gave no usable answer; A2 is new
    "unresolved",               # A2 is missing, hedged or only a partial match
)

# The fields of an episode that decide its label.
LABEL_INPUT_FIELDS = ("solver_attempt_1", "critic_advocated_answer", "solver_attempt_2",
                      "critic_noncommittal")

# Columns of labels.csv that identify the episode; the label columns follow them.
LABEL_KEY_FIELDS = ["question_id", "source_split", "partition", "lexical_v2_label"]

# How label columns are stored in labels.csv, so reading restores what was written.
BOOLEAN_COLUMNS = ("eligible_primary", "eligible_strict", "critic_unresolved_flag",
                   "critic_premise_rejection_flag", "critic_refusal_flag")
INTEGER_OR_BLANK_COLUMNS = ("primary_target", "strict_target")
TEXT_OR_BLANK_COLUMNS = ("subtype", "human_validated_behavior_v1")
LIST_COLUMNS = ("exclusion_reasons", "qc_rules")  # qc_rules only under an overlay policy

# Written only under a policy with an overlay: which rules fired and what they replaced.
TRACE_FIELDS = ("qc_rules", "base_feedback_type", "base_a2_kind")

# Ways of refusing to answer that the lexical_v2 rules did not catch.
EXTRA_NONANSWER = re.compile(
    r"\b(?:"
    r"does not (?:specify|mention|provide|state|indicate)"
    r"|there is no mention|not mentioned|not specified"
    r"|cannot be (?:determined|answered)"
    r"|insufficient information|not enough information"
    r"|(?:the )?passage does not|(?:the )?text does not"
    r"|provided information does not"
    r")\b",
    re.I,
)

# The Critic says the question itself is wrong. The single word "premise" is a
# broad match and is one of the things the human check looks at.
PREMISE = re.compile(
    r"\b(?:"
    r"false premise"
    r"|based on a (?:misunderstanding|false|flawed|incorrect)"
    r"|premise|not applicable|no valid connection"
    r"|(?:is|are) not (?:related|connected)"
    r"|does not exist|there (?:is|was) no such"
    r")\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Overlay lexicons. Frozen data; the names are historical labels for the rule
# set they belong to, not switches. Order matters: it is the order the fired
# rules are reported in.
# ---------------------------------------------------------------------------

# The Critic says the question itself is wrong, in so many words.
V012_PREMISE_PATTERNS = {
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

# The Critic says the sources do not identify the answer, in so many words.
V012_REFUSAL_PATTERNS = {
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
V012_A2_NONANSWER_PATTERNS = {
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


@dataclass(frozen=True)
class LabelPolicy:
    """Which rule set labels an episode.

    ``version`` is written into every label as ``rule_version``. The three
    override sets drive the overlay stage of ``classify_candidate``; a policy
    with all three empty has no overlay and labels exactly as the base rules do.
    """
    version: str
    premise_overrides: Mapping[str, re.Pattern] = field(default_factory=dict)
    refusal_overrides: Mapping[str, re.Pattern] = field(default_factory=dict)
    a2_nonanswer_overrides: Mapping[str, re.Pattern] = field(default_factory=dict)

    def __post_init__(self):
        # Copy each override set into a read-only mapping, so changing the dict a
        # policy was built from cannot change how the policy labels.
        for name in ("premise_overrides", "refusal_overrides", "a2_nonanswer_overrides"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def has_overlay(self):
        return bool(self.premise_overrides or self.refusal_overrides or self.a2_nonanswer_overrides)

    def definition(self):
        """Everything that decides this policy's output, as plain data for provenance."""
        def rules(overrides):
            return [[name, p.pattern, p.flags] for name, p in overrides.items()]
        return {
            "version": self.version,
            "premise_overrides": rules(self.premise_overrides),
            "refusal_overrides": rules(self.refusal_overrides),
            "a2_nonanswer_overrides": rules(self.a2_nonanswer_overrides),
        }

    def definition_sha256(self):
        return hashlib.sha256(json.dumps(self.definition(), sort_keys=True).encode()).hexdigest()


V011 = LabelPolicy(version="behavior_v0.1.1_candidate")
V012_QC = LabelPolicy(version="behavior_v0.1.2_qc_candidate",
                      premise_overrides=V012_PREMISE_PATTERNS,
                      refusal_overrides=V012_REFUSAL_PATTERNS,
                      a2_nonanswer_overrides=V012_A2_NONANSWER_PATTERNS)

# Names a config or command line may select. The default stays the frozen first
# rule set until the team adopts another under issue #108.
POLICIES = {"v0.1.1": V011, "v0.1.2-qc": V012_QC}
DEFAULT_POLICY_NAME = "v0.1.1"
DEFAULT_POLICY = POLICIES[DEFAULT_POLICY_NAME]


def answer_kind(text):
    """Classify one piece of text as 'answer', 'nonanswer', 'uncertain' or 'missing'."""
    kind = behavior.answer_kind(text)
    if kind in ("answer", "uncertain") and EXTRA_NONANSWER.search(text.replace("’", "'")):
        return "nonanswer"
    return kind


def _feedback_type(unresolved, premise_rejection, refusal):
    """What the Critic gave. The first matching case wins."""
    if unresolved:
        return "unresolved"
    if premise_rejection:
        return "premise_rejection"
    if refusal:
        return "refusal"
    return "answer"


def _conflict_status(feedback_type, a1_kind, a1_vs_critic):
    """Whether A1 and the Critic gave two different answers."""
    if feedback_type != "answer" or a1_kind != "answer":
        return "unresolved"
    if a1_vs_critic == "exact":
        return "same_position"
    if a1_vs_critic == "different":
        return "lexical_conflict_candidate"
    return a1_vs_critic  # "containment", e.g. "Paris" vs "Paris, France"


def _solver_response(feedback_type, a1_kind, a2_kind, a1_vs_critic, a2_vs_a1, a2_vs_critic):
    """What the Solver did in A2. Returns (response, subtype)."""
    if a2_kind == "nonanswer":
        return "non_answer", None
    if a2_kind != "answer":
        return "unresolved", None

    critic_answered = feedback_type == "answer"
    a1_answered = a1_kind == "answer"
    kept_a1 = a1_answered and a2_vs_a1 == "exact"
    took_critic = critic_answered and a2_vs_critic == "exact"

    if kept_a1 and critic_answered and a1_vs_critic == "exact":
        return "unchanged_same_position", "shared_answer_no_conflict"
    if kept_a1:
        return "retained_a1", None
    if took_critic and a1_kind == "nonanswer":
        return "adopted_critic", "a1_nonanswer_to_answer"
    if took_critic and not a1_answered:
        return "adopted_critic", "a1_unresolved_to_answer"
    if took_critic:
        return "adopted_critic", None
    if critic_answered and a1_answered and a2_vs_a1 == "different" and a2_vs_critic == "different":
        return "third_answer", None
    if not critic_answered:
        return "produced_answer", None
    return "unresolved", None


def _exclusion_reasons(feedback_type, a1_kind, a2_kind, conflict_status, solver_response):
    """Every reason this episode cannot be used for the probe target."""
    reasons = []
    if feedback_type != "answer":
        reasons.append("critic_" + feedback_type)
    if a1_kind != "answer":
        reasons.append("a1_" + a1_kind)
    if feedback_type == "answer" and a1_kind == "answer":
        if conflict_status == "same_position":
            reasons.append("no_conflict")
        elif conflict_status != "lexical_conflict_candidate":
            reasons.append("a1_critic_" + conflict_status)
    if a2_kind != "answer":
        reasons.append("a2_" + a2_kind)
    if solver_response == "unresolved":
        reasons.append("a2_relationship_unresolved")
    return reasons


def _matching_rules(text, patterns):
    """Names of the patterns that match ``text``; none for a missing text."""
    if not isinstance(text, str):
        return []
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def override_feedback(current, advocated_answer, policy):
    """Return (feedback_type, rules fired) for the Critic's advocated answer under ``policy``.

    "unresolved" is kept as is. An explicit premise rejection overrides "answer"
    or "refusal"; an explicit refusal overrides "answer" only. Premise rules are
    checked before refusal rules.
    """
    if current not in ("answer", "refusal", "premise_rejection"):
        return current, []
    premise = _matching_rules(advocated_answer, policy.premise_overrides)
    if premise and current in ("answer", "refusal"):
        return "premise_rejection", ["premise:" + name for name in premise]
    refusal = _matching_rules(advocated_answer, policy.refusal_overrides)
    if refusal and current == "answer":
        return "refusal", ["refusal:" + name for name in refusal]
    return current, []


def a2_nonanswer_rules(text, policy):
    """Names of the policy's A2 non-answer patterns that match."""
    return _matching_rules(text, policy.a2_nonanswer_overrides)


def classify_candidate(row, policy=DEFAULT_POLICY):
    """Label one episode under ``policy``. Returns a new dict and does not change the row.

    The base rules compute the facts. When the policy has an overlay, one generic
    stage then overrides ``feedback_type`` and ``a2_kind`` where the text is
    explicit (an unresolved Critic is never overridden; an A2 override applies
    regardless), and the label records which rules fired in ``qc_rules`` and the
    base values in ``base_feedback_type`` / ``base_a2_kind``. Everything downstream
    is derived by ``assemble_labels`` the same way for every policy.
    """
    a1 = row.get("solver_attempt_1")
    a2 = row.get("solver_attempt_2")
    critic = row.get("critic_advocated_answer")

    # What kind of text is each piece, and how do the three compare?
    a1_kind = answer_kind(a1)
    a2_kind = answer_kind(a2)
    critic_kind = answer_kind(critic)
    a1_vs_critic = behavior.textual_relation(a1, critic)
    a2_vs_a1 = behavior.textual_relation(a2, a1)
    a2_vs_critic = behavior.textual_relation(a2, critic)

    # Three independent checks on the Critic. All three are kept in the output.
    critic_unresolved = (row.get("critic_noncommittal") is not False
                         or critic_kind in ("missing", "uncertain"))
    critic_premise = isinstance(critic, str) and bool(PREMISE.search(critic))
    critic_refusal = critic_kind == "nonanswer"

    feedback_type = _feedback_type(critic_unresolved, critic_premise, critic_refusal)

    base_feedback_type, base_a2_kind, rules = feedback_type, a2_kind, []
    if policy.has_overlay:
        if not critic_unresolved:
            feedback_type, rules = override_feedback(feedback_type, critic, policy)
        a2_rules = a2_nonanswer_rules(a2, policy)
        if a2_rules:
            a2_kind = "nonanswer"
            rules = rules + ["a2:" + name for name in a2_rules]
        critic_premise = critic_premise or feedback_type == "premise_rejection"
        critic_refusal = critic_refusal or feedback_type == "refusal"

    label = assemble_labels(feedback_type, a1_kind, a2_kind, critic_kind,
                            a1_vs_critic, a2_vs_a1, a2_vs_critic,
                            critic_unresolved, critic_premise, critic_refusal,
                            rule_version=policy.version)
    if policy.has_overlay:
        label["qc_rules"] = rules
        label["base_feedback_type"] = base_feedback_type
        label["base_a2_kind"] = base_a2_kind
    return label


def assemble_labels(feedback_type, a1_kind, a2_kind, critic_kind,
                    a1_vs_critic, a2_vs_a1, a2_vs_critic,
                    critic_unresolved, critic_premise, critic_refusal, *,
                    rule_version=DEFAULT_POLICY.version):
    """Derive the Solver response, eligibility and targets from the Critic type and text facts.

    ``classify_candidate`` computes the facts, applies any overlay, and calls this,
    so the derivation exists once for every policy.
    """
    conflict_status = _conflict_status(feedback_type, a1_kind, a1_vs_critic)
    solver_response, subtype = _solver_response(
        feedback_type, a1_kind, a2_kind, a1_vs_critic, a2_vs_a1, a2_vs_critic)
    exclusion_reasons = _exclusion_reasons(
        feedback_type, a1_kind, a2_kind, conflict_status, solver_response)

    # Primary target: adopted (1) versus kept or third answer (0).
    # Strict target:  adopted (1) versus kept (0) only.
    eligible_primary = (not exclusion_reasons
                        and solver_response in ("adopted_critic", "retained_a1", "third_answer"))
    eligible_strict = eligible_primary and solver_response != "third_answer"
    adopted = int(solver_response == "adopted_critic")

    return {
        "rule_version": rule_version,
        "feedback_type": feedback_type,
        "solver_response": solver_response,
        "conflict_status": conflict_status,
        "subtype": subtype,
        "eligible_primary": eligible_primary,
        "eligible_strict": eligible_strict,
        "primary_target": adopted if eligible_primary else None,
        "strict_target": adopted if eligible_strict else None,
        "exclusion_reasons": exclusion_reasons,
        "a1_kind": a1_kind,
        "a2_kind": a2_kind,
        "critic_kind": critic_kind,
        "a1_vs_critic": a1_vs_critic,
        "a2_vs_a1": a2_vs_a1,
        "a2_vs_critic": a2_vs_critic,
        "critic_unresolved_flag": critic_unresolved,
        "critic_premise_rejection_flag": critic_premise,
        "critic_refusal_flag": critic_refusal,
        "human_validated_behavior_v1": None,
    }


def write_labels(path, rows):
    """Write one CSV row per question. Each row needs ``label`` and ``partition``.

    A list is joined with ';' and None is left blank.
    """
    label_fields = list(rows[0]["label"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LABEL_KEY_FIELDS + label_fields)
        for row in rows:
            label = {field: ";".join(value) if isinstance(value, list) else value
                     for field, value in row["label"].items()}
            values = ["" if label[field] is None else label[field] for field in label_fields]
            writer.writerow([row["question_id"], row["source_split"],
                             row["partition"], row["solver_behavior"]] + values)


def read_labels(path):
    """Read labels.csv back as {question_id: label}, with the types ``write_labels`` was given."""
    with open(path, newline="") as f:
        labels = {record["question_id"]: record for record in csv.DictReader(f)}
    for qid, label in labels.items():
        for field in BOOLEAN_COLUMNS:
            if label[field] not in ("True", "False"):
                raise ValueError(f"{qid}: {field} must be True or False, got {label[field]!r}")
            label[field] = label[field] == "True"
        for field in INTEGER_OR_BLANK_COLUMNS:
            label[field] = int(label[field]) if label[field] else None
        for field in TEXT_OR_BLANK_COLUMNS:
            label[field] = label[field] or None
        for field in LIST_COLUMNS:
            if field in label:
                label[field] = label[field].split(";") if label[field] else []
    return labels


def label_counts(rows):
    """Label counts for a group of labelled rows. Target counts use eligible rows only."""
    feedback_by_response = Counter()
    lexical_v2_to_candidate = Counter()
    exclusion_reasons = Counter()
    target_by_split = Counter()
    target_by_correctness = Counter()

    for row in rows:
        label = row["label"]
        response = label["solver_response"]
        pair = f"{label['feedback_type']} / {response}"

        feedback_by_response[pair] += 1
        lexical_v2_to_candidate[f"{row['solver_behavior']} -> {pair}"] += 1
        exclusion_reasons.update(label["exclusion_reasons"])

        if label["eligible_primary"]:
            target_by_split[f"{row['partition']} / {response}"] += 1
            correctness = (f"a1_correct={row['solver_attempt_1_correct']} / "
                           f"critic_correct={row['critic_feedback_correct']}")
            target_by_correctness[f"{correctness} / {response}"] += 1

    return {
        "rows": len(rows),
        "feedback_by_response": dict(sorted(feedback_by_response.items())),
        "lexical_v2_to_candidate": dict(sorted(lexical_v2_to_candidate.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "primary_target_by_split": dict(sorted(target_by_split.items())),
        "primary_target_by_correctness": dict(sorted(target_by_correctness.items())),
    }
