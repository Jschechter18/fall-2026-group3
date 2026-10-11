from copy import deepcopy

import pytest

from mas_sae.evaluation.behavior_v01 import (RULE_VERSION as V011_RULE_VERSION, classify_candidate,
                                             read_labels, write_labels)
from mas_sae.evaluation.behavior_v012_qc import (RULE_VERSION, classify_candidate_v012,
                                                 explicit_a2_nonanswer, propose_feedback)

TRACE_FIELDS = ("rule_version", "qc_rules", "base_feedback_type", "base_a2_kind")


def row(a1="London", critic="Paris", a2="Paris", **kwargs):
    return {"solver_attempt_1": a1, "critic_advocated_answer": critic,
            "solver_attempt_2": a2, "critic_noncommittal": False, **kwargs}


@pytest.mark.parametrize("current,advocated,expected,rules", [
    ("answer", "The provided paragraphs do not mention who founded the university.",
     "refusal", ["refusal:source_explicitly_does_not_identify"]),
    ("answer", "The mother's name is not explicitly mentioned in the provided paragraphs.",
     "refusal", ["refusal:entity_explicitly_unidentified"]),
    ("answer", "The question contains a factual error. No date exists.",
     "premise_rejection", ["premise:question_explicitly_erroneous"]),
    ("refusal", "There is no direct connection between these events.",
     "premise_rejection", ["premise:explicitly_no_connection"]),
    ("answer", "THERE WAS NO Known Link between the two.",  # case-insensitive
     "premise_rejection", ["premise:explicitly_no_connection"]),
    # Premise rejection wins over refusal wording in the same text, and the
    # premise patterns are checked before the refusal patterns.
    ("answer", "The premise is flawed; the provided paragraphs do not mention it.",
     "premise_rejection", ["premise:question_explicitly_invalid"]),
    # An existing premise rejection is never downgraded to a refusal.
    ("premise_rejection", "The provided paragraphs do not mention it.", "premise_rejection", []),
    # Negative facts and ordinary answers are left alone.
    ("answer", "No, the defendant did not win the trial.", "answer", []),
    ("answer", "The treaty was not signed in 1919.", "answer", []),
    ("answer", "The correct answer is New Hampshire.", "answer", []),
    ("answer", None, "answer", []),
    # An unresolved Critic is never touched.
    ("unresolved", "The question contains a logical error.", "unresolved", []),
])
def test_propose_feedback(current, advocated, expected, rules):
    assert propose_feedback(current, advocated) == (expected, rules)


@pytest.mark.parametrize("a2,rules", [
    ("No such location exists.", ["no_such_or_valid"]),
    ("There is no valid date for this event.", ["no_such_or_valid"]),
    ("The information is not available.", ["information_explicitly_unavailable"]),
    ("Not mentioned in the passage.", ["simple_nonanswer"]),
    ("There is no record of it.", ["no_answer_or_evidence", "no_record"]),
    ("  none  ", ["simple_nonanswer"]),
    ("None of the above", ["simple_nonanswer"]),
    ("Unknown Soldier", ["simple_nonanswer"]),  # known limit: a real answer starting with "Unknown"
    ("No", []),
    ("No, he did not win the contest.", []),
    ("1949", []),
    ("", []),
    (None, []),
])
def test_explicit_a2_nonanswer(a2, rules):
    assert explicit_a2_nonanswer(a2) == rules


def test_overlay_never_modifies_the_row_or_the_v011_result():
    r = row(critic="The provided paragraphs do not mention a castle.", a2="No such castle is mentioned.")
    before = deepcopy(r)
    base_before = classify_candidate(r)
    label = classify_candidate_v012(r)
    assert r == before
    assert classify_candidate(r) == base_before
    assert base_before["rule_version"] == V011_RULE_VERSION
    assert base_before["feedback_type"] == "answer" and label["feedback_type"] == "refusal"


def test_explicit_refusal_makes_the_episode_a_non_answer_and_ineligible():
    label = classify_candidate_v012(row(a1="Prague Castle",
                                        critic="The provided paragraphs do not mention a castle.",
                                        a2="No such castle is mentioned."))
    assert label["rule_version"] == RULE_VERSION
    assert label["feedback_type"] == "refusal" and label["solver_response"] == "non_answer"
    assert label["critic_refusal_flag"] and not label["critic_premise_rejection_flag"]
    assert not label["eligible_primary"] and label["primary_target"] is None
    assert "critic_refusal" in label["exclusion_reasons"] and "a2_nonanswer" in label["exclusion_reasons"]
    assert label["qc_rules"] == ["refusal:source_explicitly_does_not_identify", "a2:no_such_or_valid"]
    assert label["base_feedback_type"] == "answer" and label["base_a2_kind"] == "answer"


def test_explicit_premise_rejection_sets_the_flag():
    label = classify_candidate_v012(row(critic="The question contains a factual error.", a2="London"))
    assert label["feedback_type"] == "premise_rejection" and label["critic_premise_rejection_flag"]
    assert label["solver_response"] == "retained_a1" and not label["eligible_primary"]
    assert label["qc_rules"] == ["premise:question_explicitly_erroneous"]


def test_row_without_a_rule_equals_v011_except_the_trace_fields():
    for r in [row(), row(a2="London"), row(a2="Rome"), row(a1="Paris", critic="Paris", a2="Paris"),
              row(critic="The passage does not specify", a2="Rome")]:
        base = classify_candidate(r)
        label = classify_candidate_v012(r)
        assert {k: v for k, v in label.items() if k not in TRACE_FIELDS} == \
               {k: v for k, v in base.items() if k != "rule_version"}
        assert label["rule_version"] == RULE_VERSION and label["qc_rules"] == []
        assert label["base_feedback_type"] == base["feedback_type"]
        assert label["base_a2_kind"] == base["a2_kind"]


def test_valid_disagreement_stays_eligible():
    label = classify_candidate_v012(row(a1="Paris", critic="London", a2="London"))
    assert label["feedback_type"] == "answer" and label["solver_response"] == "adopted_critic"
    assert label["eligible_primary"] and label["primary_target"] == 1 and label["qc_rules"] == []


def test_unresolved_critic_is_never_overridden():
    label = classify_candidate_v012(row(critic="The question contains a factual error.",
                                        critic_noncommittal=True))
    assert label["feedback_type"] == "unresolved" and label["qc_rules"] == []
    assert label["solver_response"] == classify_candidate(
        row(critic="The question contains a factual error.", critic_noncommittal=True))["solver_response"]


def test_a2_override_applies_even_when_the_critic_is_unresolved():
    label = classify_candidate_v012(row(critic_noncommittal=True, a2="Not mentioned anywhere."))
    assert label["feedback_type"] == "unresolved" and label["a2_kind"] == "nonanswer"
    assert label["solver_response"] == "non_answer" and label["qc_rules"] == ["a2:simple_nonanswer"]


def test_v012_has_every_v011_field():
    r = row()
    assert set(classify_candidate(r)) <= set(classify_candidate_v012(r))
    assert set(classify_candidate_v012(r)) - set(classify_candidate(r)) == set(TRACE_FIELDS[1:])


def test_labels_round_trip_through_labels_csv(tmp_path):
    rows = []
    for i, r in enumerate([row(critic="The provided paragraphs do not mention it.", a2="No such thing."),
                           row()]):
        rows.append({"question_id": f"q{i}", "source_split": "train", "partition": "train",
                     "solver_behavior": "accepted", "label": classify_candidate_v012(r)})
    path = tmp_path / "labels.csv"
    write_labels(path, rows)
    labels = read_labels(path)
    for r in rows:
        assert labels[r["question_id"]]["qc_rules"] == r["label"]["qc_rules"]
        assert labels[r["question_id"]]["exclusion_reasons"] == r["label"]["exclusion_reasons"]
        assert labels[r["question_id"]]["eligible_primary"] == r["label"]["eligible_primary"]
        assert labels[r["question_id"]]["primary_target"] == r["label"]["primary_target"]
        assert labels[r["question_id"]]["rule_version"] == RULE_VERSION
