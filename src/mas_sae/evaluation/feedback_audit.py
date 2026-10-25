"""Per-episode audit of Critic feedback quality against the behavior labels (issue #115).

One row per episode keeps four things apart, so none hides another:

  feedback   whether the delivered feedback was coherent (``feedback_coherence``)
  behavior   what the label rules say the Solver did (an official label release)
  answers    truncation and long, multi-part answers that text matching handles badly
  outcome    correctness, recorded only and never used to decide the rest

``candidate_status`` combines them only for rows the release marks eligible.
It is an automatic screen for review, not a verified label and not a new label
policy (that choice is issue #108); human adjudication is kept separately.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from mas_sae.agents import critic
from mas_sae.evaluation import behavior, feedback_coherence, scoring
from mas_sae.evaluation.scoring import normalize_answer
from mas_sae.experiments.artifacts import sha256_file

# An answer longer than this many words is usually a sentence or a list, where
# exact or containment matching cannot tell a revision from a restatement.
COMPOUND_ANSWER_WORDS = 10
CANDIDATE_STATUSES = ("rule_pass", "uncertain", "invalid")
HUMAN_ELIGIBLE = ("yes", "no", "uncertain")
# v0.1.2 feedback types other than an answer: rows the 'non_answers' review
# classes sample, so errors in these labels can be measured too.
NON_ANSWER_FEEDBACK = ("refusal", "premise_rejection", "unresolved")
REVIEW_CLASSES = ("conflicts", "non_answers")
REVIEW_ID_PREFIX = {"conflicts": "F", "non_answers": "G"}


def procedure_files():
    """Source files whose code decides an audit record: parser, matching, rules."""
    return [Path(__file__)] + [Path(m.__file__) for m in (feedback_coherence, critic, behavior, scoring)]


def procedure_sha256():
    """Hash of the whole audit procedure: rule settings plus the code that applies them.

    Validation is read only under the hash frozen after development, so any
    change to a pattern, a threshold, the parser or the decision code
    invalidates the freeze.
    """
    files = {path.name: sha256_file(path) for path in procedure_files()}
    payload = {"rules": feedback_coherence.RULES, "compound_answer_words": COMPOUND_ANSWER_WORDS,
               "files": files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _reached_budget(telemetry, step):
    record = (telemetry or {}).get(step) or {}
    return bool(record.get("reached_token_budget"))


def compound_answer(text):
    return isinstance(text, str) and len(normalize_answer(text).split()) > COMPOUND_ANSWER_WORDS


def candidate_status(label, coherence, answers_flagged):
    """Whether an eligible row stays a trustworthy example; '' for ineligible rows.

    rule_pass  eligible, coherent feedback, and no answer-level warning
    uncertain  eligible, but the feedback or an answer carries a warning sign
    invalid    eligible, but the feedback contradicts itself or never arrived
    """
    if not label["eligible_primary"]:
        return ""
    if coherence in ("inconsistent", "unusable"):
        return "invalid"
    if coherence == "uncertain" or answers_flagged:
        return "uncertain"
    return "rule_pass"


def audit_row(row):
    """One audit record for an interaction that carries ``partition`` and ``label``."""
    label = row["label"]
    telemetry = row.get("generation_telemetry")
    coherence = feedback_coherence.assess(row["critic_raw_output"], row["solver_attempt_1"],
                                          telemetry=telemetry)
    answers = {
        "a1_reached_token_budget": _reached_budget(telemetry, "solver_a1"),
        "a2_reached_token_budget": _reached_budget(telemetry, "solver_a2"),
        "a1_compound": compound_answer(row["solver_attempt_1"]),
        "a2_compound": compound_answer(row["solver_attempt_2"]),
        # A sentence-length advocated answer is often a premise objection that
        # the label rules still read as an answer.
        "advocated_compound": compound_answer(coherence["delivered_advocated_answer"]),
    }
    record = {
        "question_id": row["question_id"],
        "partition": row["partition"],
        "source_split": row["source_split"],
        "question": row["question"],
        "solver_a1": row["solver_attempt_1"],
        "critic_blind_answer": row.get("critic_blind_answer"),
        "solver_a2": row["solver_attempt_2"],
        **coherence,
        "blind_equals_advocated": (
            coherence["delivered_advocated_answer"] is not None
            and normalize_answer(coherence["delivered_advocated_answer"])
            == normalize_answer(row.get("critic_blind_answer") or "")
        ),
        **answers,
        "label_rule_version": label["rule_version"],
        "feedback_type": label["feedback_type"],
        "solver_response": label["solver_response"],
        "conflict_status": label["conflict_status"],
        "a1_vs_critic": label["a1_vs_critic"],
        "a2_vs_a1": label["a2_vs_a1"],
        "a2_vs_critic": label["a2_vs_critic"],
        "eligible_primary": label["eligible_primary"],
        "eligible_strict": label["eligible_strict"],
        "primary_target": label["primary_target"],
        "strict_target": label["strict_target"],
        "exclusion_reasons": label["exclusion_reasons"],
        "candidate_status": candidate_status(label, coherence["coherence"], any(answers.values())),
        "solver_attempt_1_correct": row.get("solver_attempt_1_correct"),
        "solver_attempt_2_correct": row.get("solver_attempt_2_correct"),
        "critic_feedback_correct": row.get("critic_feedback_correct"),
        "validator_final_correct": row.get("validator_final_correct"),
    }
    return record


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a proportion; (None, None) when total is 0."""
    if total == 0:
        return None, None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _table(records, key_a, key_b):
    table = defaultdict(Counter)
    for r in records:
        table[str(r[key_a])][str(r[key_b])] += 1
    return {a: dict(sorted(counts.items())) for a, counts in sorted(table.items())}


def audit_counts(records):
    """Counts for one partition's audit records."""
    eligible = [r for r in records if r["eligible_primary"]]
    status_by_response = _table(eligible, "solver_response", "candidate_status")
    negatives = Counter(r["candidate_status"] for r in eligible if r["solver_response"] == "retained_a1")
    total = sum(negatives.values())
    lo, hi = wilson_interval(negatives["rule_pass"], total)
    reasons = Counter(reason for r in records for reason in r["coherence_reasons"])
    return {
        "rows": len(records),
        "coherence": dict(sorted(Counter(r["coherence"] for r in records).items())),
        "coherence_reasons": dict(sorted(reasons.items())),
        "coherence_by_solver_response": _table(records, "solver_response", "coherence"),
        "coherence_by_feedback_type": _table(records, "feedback_type", "coherence"),
        "eligible_status_by_solver_response": status_by_response,
        "strict_negatives": {
            "eligible_retained_a1": total,
            **{status: negatives[status] for status in CANDIDATE_STATUSES},
            # The share of this sample that passes the rules, not a count of
            # verified negatives: rule_pass rows can still be human-rejected.
            "rule_pass_share_95ci": [lo, hi],
        },
        "possible_missed_negatives": dict(sorted(Counter(
            f"{r['conflict_status']} / {r['coherence']}" for r in records if possible_missed_negative(r)
        ).items())),
        "answer_flags": {
            flag: sum(bool(r[flag]) for r in records)
            for flag in ("a1_reached_token_budget", "a2_reached_token_budget",
                         "a1_compound", "a2_compound", "advocated_compound")
        },
    }


# Conflicts the label rules exclude because text matching cannot settle them.
UNSETTLED_CONFLICTS = ("containment", "lexical_conflict_candidate")


def possible_missed_negative(record):
    """An excluded row that may hide a genuine rejection of a coherent alternative.

    The Critic gave an answer and its feedback reached the Solver, but A1 and the
    advocated answer only partly matched, and A2 kept A1: it repeats A1, or
    contains A1 without being the advocated answer (which would be adoption).
    """
    return (not record["eligible_primary"]
            and record["feedback_type"] == "answer"
            and record["coherence"] != "unusable"
            and record["conflict_status"] in UNSETTLED_CONFLICTS
            and (record["a2_vs_a1"] == "exact"
                 or (record["a2_vs_a1"] == "containment" and record["a2_vs_critic"] != "exact")))


def choose_review_rows(records, reviewed_ids, *, per_group, seed, classes="conflicts"):
    """Pick train rows for blind human review that no human has seen yet.

    classes='conflicts': every eligible 'retained_a1' row is taken, because
    each one decides whether a strict negative class exists. Other eligible
    rows contribute up to ``per_group`` from each (solver_response, coherence)
    group, including rows the rules call clean, so missed errors can be
    estimated. Rows the label rules excluded only because text matching could
    not settle the conflict (see ``possible_missed_negative``) form their own
    groups, so genuine rejections hidden by exclusion can be estimated too.

    classes='non_answers': up to ``per_group`` from each (feedback_type,
    coherence) group of rows the release labelled refusal, premise rejection
    or unresolved, so errors in those labels can be estimated. Its review ids
    start with a different letter, so the two packets never share an id.

    Returns (chosen records, private key entries, sampling record).
    """
    if classes not in REVIEW_CLASSES:
        raise ValueError(f"classes must be one of {REVIEW_CLASSES}, got {classes!r}")
    groups = defaultdict(list)
    for r in records:
        if r["partition"] != "train":
            raise ValueError(f"{r['question_id']}: review rows come from the train partition only")
        if r["question_id"] in reviewed_ids:
            continue
        if classes == "non_answers":
            if r["feedback_type"] in NON_ANSWER_FEEDBACK:
                groups[("label_" + r["feedback_type"], r["coherence"])].append(r)
        elif r["eligible_primary"]:
            groups[(r["solver_response"], r["coherence"])].append(r)
        elif possible_missed_negative(r):
            groups[("excluded_" + r["conflict_status"], r["coherence"])].append(r)

    rng = random.Random(seed)
    picked = []
    for group, members in sorted(groups.items()):
        members = sorted(members, key=lambda r: r["question_id"])
        if group[0] != "retained_a1":
            members = rng.sample(members, min(per_group, len(members)))
        picked += [(f"{group[0]} / {group[1]}", r) for r in members]
    rng.shuffle(picked)
    chosen = [r for _, r in picked]

    key = [{"review_id": f"{REVIEW_ID_PREFIX[classes]}{position + 1:04d}",
            "question_id": r["question_id"],
            "stratum": stratum,
            "second_annotator": False}
           for position, (stratum, r) in enumerate(picked)]
    sampling = {
        "seed": seed,
        "classes": classes,
        "per_group": per_group,
        "excluded_already_reviewed": len(reviewed_ids),
        "population": {f"{a} / {b}": len(m) for (a, b), m in sorted(groups.items())},
        "selected": dict(sorted(Counter(entry["stratum"] for entry in key).items())),
    }
    return chosen, key, sampling


def human_comparison(records, human):
    """Compare audit records with human review, for rows a human reviewed.

    ``human`` maps question_id to {"eligible_primary": yes|no|uncertain,
    "feedback_type": ..., "contradiction_noted": bool}. Rule contradiction is
    coherence 'inconsistent'; flagged is 'inconsistent' or 'uncertain'.

    Annotators wrote "contradict" when they saw one, but never marked feedback
    as coherent, so a row without the note is not a verified coherent row: the
    'inconsistent' share among those rows is not a false-positive rate.
    """
    reviewed = [r for r in records if r["question_id"] in human]
    contradiction = Counter()
    eligibility = Counter()
    human_retained = Counter()
    for r in reviewed:
        h = human[r["question_id"]]
        contradiction[(h["contradiction_noted"], r["coherence"])] += 1
        if r["eligible_primary"]:
            eligibility[(h["eligible_primary"], r["candidate_status"])] += 1
        if h.get("solver_response") == "retained_a1":
            human_retained[h["eligible_primary"]] += 1

    def rate(noted, decisions):
        hits = sum(contradiction[(noted, d)] for d in decisions)
        total = sum(n for (k, _), n in contradiction.items() if k == noted)
        return {"count": hits, "of": total, "share_95ci": list(wilson_interval(hits, total))}

    return {
        "reviewed_rows": len(reviewed),
        "contradiction_noted_by_coherence": {
            f"noted={k} / {d}": n for (k, d), n in sorted(contradiction.items(), key=str)},
        "noted_called_inconsistent": rate(True, ("inconsistent",)),
        "noted_flagged": rate(True, ("inconsistent", "uncertain")),
        "not_noted_called_inconsistent": rate(False, ("inconsistent",)),
        # The human-adjudicated negative class, independent of any rule.
        "human_retained_a1_by_eligible": dict(sorted(human_retained.items())),
        "human_eligible_by_candidate_status": {
            f"human={k} / rules={s}": n for (k, s), n in sorted(eligibility.items(), key=str)},
    }
