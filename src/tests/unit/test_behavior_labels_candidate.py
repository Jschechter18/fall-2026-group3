"""Labels for the labeled-candidates A2 presentation (policy v0.2.0-candidate)."""
import json
import importlib.util
import sys

import yaml
from pathlib import Path

import pytest

from mas_sae.evaluation.behavior_labels import (
    CANDIDATE_FIELDS,
    POLICIES,
    TRACE_FIELDS,
    V011,
    V012_QC,
    V020_CANDIDATE,
    classify_candidate,
    read_labels,
    write_labels,
    label_counts,
)
from mas_sae.evaluation.feedback_coherence import DECISIONS
from mas_sae.experiments.records import write_jsonl

POLICY = V020_CANDIDATE
LONG = " ".join(["word"] * 15)          # one word over the limit
LIMIT = " ".join(["word"] * 14)         # exactly at the limit
V012_PACKAGE_COUNTS = Path("results/behavior/behavior_v012_official_20261011/counts.json")


def row(a1="London", critic="Paris", a2="Paris", order="a1_first", distinct=True, episode="q__natural",
        a2_first=None, **kwargs):
    """One candidates episode, with the fields the collection records. A1's first
    token id is 1 and the recommendation's 2; A2's is the chosen one's unless given, else 3."""
    if a2_first is None:
        a2_first = 1 if a2 == a1 else 2 if a2 == critic else 3
    return {"episode_id": episode, "question_id": episode.split("__")[0], "question": "Where is it?",
            "solver_attempt_1": a1, "critic_advocated_answer": critic, "solver_attempt_2": a2,
            "critic_noncommittal": False, "a2_presentation": "candidates", "candidate_order": order,
            "candidate_first_token_ids": {"a1": 1, "critic": 1 if distinct is False else 2}, "decision_token_distinct": distinct,
            "solver_a2_first_token_id": a2_first, **kwargs}


# --- the policy ---------------------------------------------------------------

def test_earlier_policy_definitions_and_hashes_are_unchanged():
    assert V011.definition_sha256() == "06eacd94c6ae44b05224a3792e9abd8937e131b6f0a62ecd7e714c707ef93827"
    assert V012_QC.definition_sha256() == "e21f78cbc0800cb6fa78f526a63e777913498b5dd5bc7fe9bf406ca8b1c319b2"
    recorded = json.loads(V012_PACKAGE_COUNTS.read_text())["label_policy"]["definition_sha256"]
    assert V012_QC.definition_sha256() == recorded
    assert "candidate_rules" not in V011.definition() and "candidate_rules" not in V012_QC.definition()


def test_candidate_policy_is_registered_and_carries_its_rules_in_the_hash():
    assert POLICIES["v0.2.0-candidate"] is POLICY and POLICY.version == "behavior_v0.2.0_candidate"
    definition = POLICY.definition()
    assert definition["candidate_rules"]["max_answer_words"] == 14
    assert definition["candidate_rules"]["equivalence_method"] == "question_aliases_conservative_text_v1"
    assert POLICY.definition_sha256() != "88385bbde30ee6418a5a96758a1aeae348bf800eef2418cbaf2146fb50a4aec5"
    assert {k: v for k, v in definition.items() if k not in ("version", "candidate_rules")} == \
           {k: v for k, v in V012_QC.definition().items() if k != "version"}
    assert len({p.definition_sha256() for p in POLICIES.values()}) == len(POLICIES)


def test_earlier_policies_write_no_candidate_fields():
    for policy in (V011, V012_QC):
        assert not set(CANDIDATE_FIELDS) & set(classify_candidate(row(), policy))


# --- a lexical conflict -------------------------------------------------------

@pytest.mark.parametrize("a2,target,resolved", [("Paris", 1, "reviewer"), ("London", 0, "previous")])
def test_strict_target_is_reviewer_against_previous(a2, target, resolved):
    label = classify_candidate(row(a2=a2), POLICY)
    assert label["strict_target"] == target and label["eligible_strict"]
    assert label["resolved_candidate"] == resolved and label["resolved_via"] == "exact"
    assert label["conflict_status"] == "lexical_conflict_candidate" and label["decision_at_first_token"]


def test_exact_match_is_no_conflict():
    label = classify_candidate(row(a1="Paris", critic="paris", a2="Paris"), POLICY)
    assert label["exclusion_reasons"] == ["no_conflict"] and label["resolved_candidate"] == "none"


def test_containment_is_excluded_not_resolved_to_same_or_different():
    label = classify_candidate(row(a1="Paris", critic="Paris, France", a2="Paris"), POLICY)
    assert label["conflict_status"] == "containment" and label["a1_vs_critic"] == "containment"
    assert label["exclusion_reasons"] == ["a1_critic_containment"] and label["strict_target"] is None


# --- which candidate A2 chose -------------------------------------------------

@pytest.mark.parametrize("a2,order,resolved", [
    ("Candidate A", "a1_first", "previous"), ("Candidate B", "a1_first", "reviewer"),
    ("candidate a.", "critic_first", "reviewer"), ("CANDIDATE B", "critic_first", "previous"),
])
def test_candidate_letter_maps_through_the_order(a2, order, resolved):
    label = classify_candidate(row(a2=a2, order=order), POLICY)
    assert label["resolved_candidate"] == resolved and label["resolved_via"] == "letter"


def test_a_letter_without_an_order_is_unresolved():
    label = classify_candidate(row(a2="Candidate A", order=None), POLICY)
    assert label["solver_response"] == "unresolved" and "a2_relationship_unresolved" in label["exclusion_reasons"]
    assert label["decision_at_first_token"] is None


def test_different_from_both_is_a_third_answer_never_retention():
    label = classify_candidate(row(a2="Rome"), POLICY)
    assert label["solver_response"] == "third_answer" and label["resolved_candidate"] == "none"
    assert label["strict_target"] is None and label["primary_target"] == 0


def test_a2_containing_one_candidate_is_unresolved():
    label = classify_candidate(row(a2="Paris, France"), POLICY)
    assert label["solver_response"] == "unresolved" and label["strict_target"] is None
    assert "a2_relationship_unresolved" in label["exclusion_reasons"]


def test_a2_non_answer_is_never_retention():
    label = classify_candidate(row(a2="None of the above"), POLICY)
    assert label["solver_response"] == "non_answer" and label["strict_target"] is None


# --- answer-shaped texts ------------------------------------------------------

def test_answer_shape_limit_is_inclusive():
    assert classify_candidate(row(a1=LIMIT), POLICY)["a1_kind"] == "answer"
    assert classify_candidate(row(a1=LONG), POLICY)["a1_kind"] == "not_answer_shaped"


def test_long_a1_or_recommendation_is_excluded():
    label = classify_candidate(row(a1=LONG), POLICY)
    assert "a1_not_answer_shaped" in label["exclusion_reasons"]
    label = classify_candidate(row(critic=LONG, a2=LONG), POLICY)
    assert label["feedback_type"] == "unresolved"
    assert {"critic_unresolved", "critic_not_answer_shaped"} <= set(label["exclusion_reasons"])


def test_long_a2_is_not_resolved():
    label = classify_candidate(row(a2=LONG), POLICY)
    assert label["a2_kind"] == "not_answer_shaped" and "a2_not_answer_shaped" in label["exclusion_reasons"]
    assert label["resolved_candidate"] == "none"


def test_the_v012_overlay_still_applies():
    label = classify_candidate(row(critic="The question contains a factual error."), POLICY)
    assert label["feedback_type"] == "premise_rejection"


# --- the decision token -------------------------------------------------------

def test_shared_first_token_only_excludes_the_causal_subset():
    label = classify_candidate(row(distinct=False, a2_first=1), POLICY)
    assert label["eligible_behavior"] and label["strict_target"] == 1
    assert not label["eligible_first_token"]
    assert label["exclusion_reasons"] == []
    assert label["first_token_exclusion_reasons"] == ["shared_first_token"]


def test_letter_reply_has_a_behavior_label_but_not_first_token_eligibility():
    label = classify_candidate(row(a2="Candidate B"), POLICY)
    assert label["resolved_candidate"] == "reviewer" and label["decision_at_first_token"] is False
    assert label["eligible_behavior"] and label["strict_target"] == 1
    assert not label["eligible_first_token"]
    assert label["first_token_exclusion_reasons"] == ["decision_not_at_first_token"]


def test_first_token_mismatch_does_not_change_full_answer_behavior():
    label = classify_candidate(row(a2="Paris", a2_first=7), POLICY)
    assert label["resolved_candidate"] == "reviewer" and label["resolved_via"] == "exact"
    assert label["eligible_behavior"] and label["strict_target"] == 1
    assert label["first_token_exclusion_reasons"] == ["decision_not_at_first_token"]


def test_missing_token_metadata_only_excludes_the_causal_subset():
    episode = row(distinct=None, candidate_first_token_ids=None, solver_a2_first_token_id=None)
    label = classify_candidate(episode, POLICY)
    assert label["eligible_behavior"] and label["strict_target"] == 1
    assert not label["eligible_first_token"]
    assert label["decision_token_distinct"] is None and label["decision_at_first_token"] is None


@pytest.mark.parametrize("metadata", [
    {"candidate_first_token_ids": {"a1": True, "critic": 2}},
    {"candidate_first_token_ids": {"a1": "1", "critic": 2}},
    {"candidate_first_token_ids": {"a1": -1, "critic": 2}},
    {"candidate_first_token_ids": []},
    {"candidate_first_token_ids": {"a1": 1}},
    {"solver_a2_first_token_id": "2"},
    {"solver_a2_first_token_id": True},
    {"decision_token_distinct": None},
    {"decision_token_distinct": "True"},
    {"decision_token_distinct": False},  # contradicts valid distinct ids
])
def test_invalid_or_inconsistent_token_metadata_cannot_enter_causal_subset(metadata):
    label = classify_candidate({**row(), **metadata}, POLICY)
    assert label["eligible_behavior"] and label["strict_target"] == 1
    assert not label["eligible_first_token"] and label["first_token_exclusion_reasons"]


def test_eligible_causal_row_is_also_behavior_eligible():
    label = classify_candidate(row(), POLICY)
    assert label["eligible_first_token"] and label["eligible_behavior"]
    assert label["decision_token_distinct"] is True and label["decision_at_first_token"] is True
    assert label["first_token_exclusion_reasons"] == []


# --- question-specific equivalence -------------------------------------------

@pytest.mark.parametrize("a1,critic", [("CFL", "Canadian Football League"),
                                      ("Canadian Football League", "CFL")])
@pytest.mark.parametrize("a2", ["CFL", "Canadian Football League", "Candidate A", "Candidate B"])
def test_documented_question_aliases_are_agreement_not_adoption(a1, critic, a2):
    label = classify_candidate(row(a1=a1, critic=critic, a2=a2,
                                   ground_truth="Canadian Football League", answer_aliases=["CFL"]), POLICY)
    assert label["a1_vs_critic"] == "exact" and label["a1_vs_critic_basis"] == "question_alias"
    assert label["conflict_status"] == "same_position"
    assert "no_conflict" in label["behavior_exclusion_reasons"]
    assert label["strict_target"] is None and not label["eligible_behavior"]
    assert not label["eligible_first_token"]


@pytest.mark.parametrize("a1,critic,a2,target", [
    ("United Kingdom", "France", "UK", 0),
    ("France", "United Kingdom", "UK", 1),
])
def test_a2_alias_resolves_complete_answer_without_requiring_token_alignment(a1, critic, a2, target):
    label = classify_candidate(row(a1=a1, critic=critic, a2=a2, ground_truth="United Kingdom",
                                   answer_aliases=["UK"], a2_first=9), POLICY)
    assert label["strict_target"] == target and label["eligible_behavior"]
    assert label["resolved_via"] == "question_alias"
    assert label["decision_at_first_token"] is False and not label["eligible_first_token"]


@pytest.mark.parametrize("a1,critic", [("CFL", "Canadian Football League"),
                                      ("Paul Warfield Tibbets", "Paul Tibbets"),
                                      ("USA", "United States of America")])
def test_unverified_abbreviations_and_name_omissions_are_ambiguous(a1, critic):
    for left, right in ((a1, critic), (critic, a1)):
        label = classify_candidate(row(a1=left, critic=right, a2=left), POLICY)
        assert label["conflict_status"] == "ambiguous_alias"
        assert "a1_critic_ambiguous_alias" in label["behavior_exclusion_reasons"]
        assert label["strict_target"] is None


def test_ambiguous_a2_equivalence_is_excluded_instead_of_a_third_answer():
    label = classify_candidate(row(a1="United Kingdom", critic="France", a2="UK"), POLICY)
    assert label["solver_response"] == "unresolved" and label["strict_target"] is None
    assert "a2_equivalence_uncertain" in label["behavior_exclusion_reasons"]


def test_aliases_are_question_local_and_correctness_flags_do_not_decide_behavior():
    base = row(a1="Britain", critic="United Kingdom", a2="United Kingdom")
    documented = {**base, "ground_truth": "United Kingdom", "answer_aliases": ["Britain"]}
    assert classify_candidate(documented, POLICY)["strict_target"] is None
    assert classify_candidate(base, POLICY)["strict_target"] == 1
    assert classify_candidate({**base, "ground_truth": "France", "answer_aliases": ["French Republic"]}, POLICY)["strict_target"] == 1
    expected = classify_candidate(documented, POLICY)
    for correct in (False, True):
        assert classify_candidate({**documented, "solver_attempt_1_correct": correct,
                                   "critic_feedback_correct": correct, "solver_attempt_2_correct": correct}, POLICY) == expected


def test_alias_resolution_precedes_ambiguous_containment():
    label = classify_candidate(row(a1="Paris", critic="Paris, France", a2="Paris",
                                   ground_truth="Paris", answer_aliases=["Paris, France"]), POLICY)
    assert label["conflict_status"] == "same_position" and label["strict_target"] is None


def test_overlap_with_an_alias_of_other_candidate_is_ambiguous():
    label = classify_candidate(row(a1="NYC", critic="New York", a2="NYC",
                                   ground_truth="New York City", answer_aliases=["NYC"]), POLICY)
    assert label["conflict_status"] == "ambiguous_alias" and label["strict_target"] is None


def test_third_answers_never_enter_either_binary_population():
    label = classify_candidate(row(a2="Rome"), POLICY)
    assert label["solver_response"] == "third_answer" and label["strict_target"] is None
    assert not label["eligible_behavior"] and not label["eligible_first_token"]
    assert label["behavior_exclusion_reasons"] == ["third_answer"]


# --- diagnostics and files ----------------------------------------------------

def test_feedback_coherence_is_a_diagnostic_only():
    raw = json.dumps({"verdict": "disagree", "advocated_answer": "Paris",
                      "explanation": "The proposed answer is correct."})
    label = classify_candidate(row(critic_raw_output=raw), POLICY)
    plain = classify_candidate(row(), POLICY)
    assert label["feedback_coherence"] in DECISIONS and plain["feedback_coherence"] == "unusable"
    assert {k: v for k, v in label.items() if k != "feedback_coherence"} == \
           {k: v for k, v in plain.items() if k != "feedback_coherence"}


def test_candidate_labels_round_trip_through_labels_csv(tmp_path):
    rows = []
    for i, episode in enumerate([row(), row(distinct=None), row(a2="Candidate A")]):
        rows.append({"question_id": f"q{i}", "source_split": "train", "partition": "train",
                     "solver_behavior": "adopted_critic", "label": classify_candidate(episode, POLICY)})
    write_labels(tmp_path / "labels.csv", rows)
    back = read_labels(tmp_path / "labels.csv")
    for r in rows:
        for field in (*CANDIDATE_FIELDS, "strict_target"):
            assert back[r["question_id"]][field] == r["label"][field]
    assert set(CANDIDATE_FIELDS) | set(TRACE_FIELDS) <= set(back["q0"])


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_behavior_v01", Path("scripts/prepare_behavior_v01.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_prepare_rejects_removed_judgments_argument(tmp_path, monkeypatch):
    script = load_script()
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(tmp_path / "out"),
                                      "--judgments", str(tmp_path / "old_judgments")])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2 and not (tmp_path / "out").exists()


def test_prepare_builds_candidate_package_without_judgments_and_preserves_inputs(tmp_path, monkeypatch):
    import csv
    from mas_sae.experiments.artifacts import sha256_file
    script = load_script()
    inputs = tmp_path / "run"
    episodes = []
    for split in ("train", "validation"):
        folder = inputs / split
        folder.mkdir(parents=True)
        records = [row(a2=answer, episode=f"2hop__{split}{i}__natural",
                       question_id=f"2hop__{split}{i}", source_split=split,
                       attempt1_activation_index=i, attempt2_activation_index=i,
                       solver_behavior="retained_solver_a1" if i == 0 else "adopted_critic",
                       solver_attempt_1_correct=False, critic_feedback_correct=True,
                       critic_feedback="The reviewer recommends Paris.")
                   for i, answer in enumerate(("London", "Paris"))]
        write_jsonl(folder / "interactions.jsonl", records)
        (folder / "resolved_config.yaml").write_text(yaml.safe_dump({"dataset": {"revision": "fixture"}}))
        episodes.extend(records)
    frozen = tmp_path / "partitions.csv"
    with frozen.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "source_split", "partition", "hop_group"])
        writer.writeheader()
        writer.writerows({"question_id": r["question_id"], "source_split": r["source_split"],
                          "partition": r["source_split"], "hop_group": "2hop"} for r in episodes)
    hashes = {str(p): sha256_file(p) for p in inputs.glob("*/*")}
    monkeypatch.setattr(script, "source_paragraphs", lambda ids, revision: {qid: [] for qid in ids})
    output = tmp_path / "package"
    monkeypatch.setattr(sys, "argv", ["prepare", "--full-results", str(inputs), "--output", str(output),
                                      "--frozen-partitions", str(frozen), "--label-policy", "v0.2.0-candidate"])
    script.main()
    labels = read_labels(output / "labels.csv")
    assert [labels[r["question_id"]]["strict_target"] for r in episodes] == [0, 1, 0, 1]
    counts = json.loads((output / "counts.json").read_text())
    assert counts["label_policy"]["definition_sha256"] == POLICY.definition_sha256()
    assert "judgments" not in counts["label_policy"]
    assert counts["all"]["strict_target_by_split"] == {
        "train / adopted_critic": 1, "train / retained_a1": 1,
        "validation / adopted_critic": 1, "validation / retained_a1": 1}
    assert counts["all"]["first_token_target_by_split"] == counts["all"]["strict_target_by_split"]
    assert {str(p): sha256_file(p) for p in inputs.glob("*/*")} == hashes
    assert list(csv.DictReader((output / "partitions.csv").open())) == list(csv.DictReader(frozen.open()))
    with pytest.raises(FileExistsError):
        script.main()


def test_counts_keep_third_answers_out_of_binary_populations():
    episodes = [row(a2="London"), row(a2="Candidate B"), row(a2="Rome")]
    records = [{**episode, "partition": "train", "solver_behavior": "fixture",
                "solver_attempt_1_correct": True, "critic_feedback_correct": False,
                "label": classify_candidate(episode, POLICY)} for episode in episodes]
    counts = label_counts(records)
    assert counts["strict_target_by_split"] == {"train / adopted_critic": 1, "train / retained_a1": 1}
    assert counts["first_token_target_by_split"] == {"train / retained_a1": 1}
    assert counts["behavior_exclusion_reasons"] == {"third_answer": 1}
    assert counts["first_token_exclusion_reasons"]["decision_not_at_first_token"] == 1
