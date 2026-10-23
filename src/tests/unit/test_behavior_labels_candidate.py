"""Labels for the labeled-candidates A2 presentation (policy v0.2.0-candidate)."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from mas_sae.agents.validator import Validator
from mas_sae.evaluation.behavior_labels import (
    CANDIDATE_FIELDS,
    JUDGMENT_PAIRS,
    POLICIES,
    TRACE_FIELDS,
    V011,
    V012_QC,
    V020_CANDIDATE,
    classify_candidate,
    judge_answer_pairs,
    load_run_judgments,
    read_judgments,
    read_labels,
    write_labels,
)
from mas_sae.evaluation.feedback_coherence import DECISIONS
from mas_sae.experiments.records import write_jsonl

POLICY = V020_CANDIDATE
LONG = " ".join(["word"] * 15)          # one word over the limit
LIMIT = " ".join(["word"] * 14)         # exactly at the limit
V012_PACKAGE_COUNTS = Path("results/behavior/behavior_v012_official_20261011/counts.json")


def row(a1="London", critic="Paris", a2="Paris", order="a1_first", distinct=True, episode="q__natural", **kwargs):
    """One candidates episode, with the fields the collection records."""
    return {"episode_id": episode, "question_id": episode.split("__")[0], "question": "Where is it?",
            "solver_attempt_1": a1, "critic_advocated_answer": critic, "solver_attempt_2": a2,
            "critic_noncommittal": False, "a2_presentation": "candidates", "candidate_order": order,
            "candidate_first_token_ids": {"a1": 1, "critic": 2}, "decision_token_distinct": distinct,
            "solver_a2_first_token_id": 2, **kwargs}


CONFLICT = {"a1_vs_critic": "different"}


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
    assert {k: v for k, v in definition.items() if k not in ("version", "candidate_rules")} == \
           {k: v for k, v in V012_QC.definition().items() if k != "version"}
    assert len({p.definition_sha256() for p in POLICIES.values()}) == len(POLICIES)


def test_earlier_policies_write_no_candidate_fields_and_refuse_judgments():
    for policy in (V011, V012_QC):
        label = classify_candidate(row(), policy)
        assert not set(CANDIDATE_FIELDS) & set(label)
        with pytest.raises(ValueError, match="candidate rules"):
            classify_candidate(row(), policy, judgments=CONFLICT)


def test_judgments_are_checked():
    with pytest.raises(ValueError, match="unknown judgment pairs"):
        classify_candidate(row(), POLICY, {"a1_vs_gold": "same"})
    with pytest.raises(ValueError, match="must be one of"):
        classify_candidate(row(), POLICY, {"a1_vs_critic": "yes"})
    assert classify_candidate(row(), POLICY, {"a1_vs_critic": None})["exclusion_reasons"] == ["a1_critic_unjudged"]


# --- a real disagreement ------------------------------------------------------

def test_a_conflict_needs_a_judge_verdict_and_is_never_guessed():
    label = classify_candidate(row(), POLICY)
    assert label["judgment_pairs"] == ["a1_vs_critic"]
    assert label["conflict_status"] == "unjudged" and label["exclusion_reasons"] == ["a1_critic_unjudged"]
    assert label["strict_target"] is None and not label["eligible_strict"]


@pytest.mark.parametrize("a2,target,resolved", [("Paris", 1, "reviewer"), ("London", 0, "previous")])
def test_strict_target_is_reviewer_against_previous(a2, target, resolved):
    label = classify_candidate(row(a2=a2), POLICY, CONFLICT)
    assert label["strict_target"] == target and label["eligible_strict"]
    assert label["resolved_candidate"] == resolved and label["resolved_via"] == "exact"
    assert label["conflict_status"] == "lexical_conflict_candidate" and label["a1_vs_critic_basis"] == "judge"


def test_judged_same_answer_is_no_conflict_and_a2_matching_either_matches_both():
    label = classify_candidate(row(a1="Thirty-six", critic="36", a2="36"), POLICY, {"a1_vs_critic": "same"})
    assert label["solver_response"] == "unchanged_same_position"
    assert label["exclusion_reasons"] == ["no_conflict"] and label["resolved_candidate"] == "none"
    assert label["a2_vs_a1"] == "exact" and label["a2_vs_a1_basis"] == "judge"


def test_uncertain_disagreement_is_excluded():
    label = classify_candidate(row(), POLICY, {"a1_vs_critic": "uncertain"})
    assert label["exclusion_reasons"] == ["a1_critic_uncertain"] and label["strict_target"] is None


def test_containment_is_judged_not_treated_as_a_partial_match():
    label = classify_candidate(row(a1="Paris", critic="Paris, France", a2="Paris"), POLICY,
                               {"a1_vs_critic": "same"})
    assert label["judgment_pairs"] == ["a1_vs_critic"] and label["exclusion_reasons"] == ["no_conflict"]
    assert classify_candidate(row(a1="Paris", critic="Paris, France"), V012_QC)["conflict_status"] == "containment"


# --- which candidate A2 chose -------------------------------------------------

@pytest.mark.parametrize("a2,order,resolved", [
    ("Candidate A", "a1_first", "previous"), ("Candidate B", "a1_first", "reviewer"),
    ("candidate a.", "critic_first", "reviewer"), ("CANDIDATE B", "critic_first", "previous"),
])
def test_candidate_letter_maps_through_the_order(a2, order, resolved):
    label = classify_candidate(row(a2=a2, order=order), POLICY, CONFLICT)
    assert label["resolved_candidate"] == resolved and label["resolved_via"] == "letter"
    assert label["judgment_pairs"] == ["a1_vs_critic"] and label["eligible_strict"]


def test_a_letter_without_an_order_is_unresolved():
    label = classify_candidate(row(a2="Candidate A", order=None), POLICY, CONFLICT)
    assert label["solver_response"] == "unresolved" and "a2_relationship_unresolved" in label["exclusion_reasons"]


def test_a2_matching_neither_exactly_takes_a_verdict_against_each_candidate():
    episode = row(a2="Paris, France")
    assert classify_candidate(episode, POLICY)["judgment_pairs"] == list(JUDGMENT_PAIRS)
    label = classify_candidate(episode, POLICY, {**CONFLICT, "a2_vs_a1": "different", "a2_vs_critic": "same"})
    assert label["resolved_candidate"] == "reviewer" and label["resolved_via"] == "judge"
    assert label["strict_target"] == 1


def test_different_from_both_is_a_third_answer_never_retention():
    label = classify_candidate(row(a2="Rome"), POLICY, {**CONFLICT, "a2_vs_a1": "different", "a2_vs_critic": "different"})
    assert label["solver_response"] == "third_answer" and label["resolved_candidate"] == "none"
    assert label["strict_target"] is None and label["primary_target"] == 0


@pytest.mark.parametrize("verdicts", [
    {"a2_vs_a1": "same", "a2_vs_critic": "same"},          # contradicts A1 != recommendation
    {"a2_vs_a1": "uncertain", "a2_vs_critic": "different"},
    {"a2_vs_a1": "different"},                              # a verdict is missing
])
def test_inconsistent_uncertain_or_missing_a2_verdicts_are_unresolved(verdicts):
    label = classify_candidate(row(a2="Rome"), POLICY, {**CONFLICT, **verdicts})
    assert label["solver_response"] == "unresolved" and label["strict_target"] is None


def test_a2_non_answer_is_never_retention():
    label = classify_candidate(row(a2="None of the above"), POLICY, CONFLICT)
    assert label["solver_response"] == "non_answer" and label["strict_target"] is None
    assert label["judgment_pairs"] == ["a1_vs_critic"]


# --- answer-shaped texts ------------------------------------------------------

def test_answer_shape_limit_is_inclusive():
    assert classify_candidate(row(a1=LIMIT), POLICY, CONFLICT)["a1_kind"] == "answer"
    assert classify_candidate(row(a1=LONG), POLICY, CONFLICT)["a1_kind"] == "not_answer_shaped"


def test_long_a1_or_recommendation_is_excluded_without_asking_the_judge():
    label = classify_candidate(row(a1=LONG), POLICY)
    assert label["judgment_pairs"] == [] and "a1_not_answer_shaped" in label["exclusion_reasons"]
    label = classify_candidate(row(critic=LONG, a2=LONG), POLICY)
    assert label["judgment_pairs"] == [] and label["feedback_type"] == "unresolved"
    assert {"critic_unresolved", "critic_not_answer_shaped"} <= set(label["exclusion_reasons"])


def test_long_a2_is_not_judged_and_not_resolved():
    label = classify_candidate(row(a2=LONG), POLICY, CONFLICT)
    assert label["judgment_pairs"] == ["a1_vs_critic"]
    assert label["a2_kind"] == "not_answer_shaped" and "a2_not_answer_shaped" in label["exclusion_reasons"]


def test_the_v012_overlay_still_applies():
    label = classify_candidate(row(critic="The question contains a factual error."), POLICY)
    assert label["feedback_type"] == "premise_rejection" and label["judgment_pairs"] == []


# --- the decision token -------------------------------------------------------

def test_shared_first_token_is_excluded_from_both_targets():
    label = classify_candidate(row(distinct=False), POLICY, CONFLICT)
    assert label["exclusion_reasons"] == ["shared_first_token"]
    assert label["strict_target"] is None and label["primary_target"] is None
    assert label["resolved_candidate"] == "reviewer"          # what A2 chose is still recorded


def test_episodes_without_the_field_are_not_excluded_for_it():
    episode = row(a2_presentation="feedback", order=None, distinct=None, candidate_first_token_ids=None)
    label = classify_candidate(episode, POLICY, CONFLICT)
    assert label["eligible_strict"] and label["decision_token_distinct"] is None


# --- diagnostics and files ----------------------------------------------------

def test_feedback_coherence_is_a_diagnostic_only():
    raw = json.dumps({"verdict": "disagree", "advocated_answer": "Paris",
                      "explanation": "The proposed answer is correct."})
    label = classify_candidate(row(critic_raw_output=raw), POLICY, CONFLICT)
    plain = classify_candidate(row(), POLICY, CONFLICT)
    assert label["feedback_coherence"] in DECISIONS and plain["feedback_coherence"] == "unusable"
    assert {k: v for k, v in label.items() if k != "feedback_coherence"} == \
           {k: v for k, v in plain.items() if k != "feedback_coherence"}


def test_candidate_labels_round_trip_through_labels_csv(tmp_path):
    rows = []
    for i, (episode, judgments) in enumerate([(row(), CONFLICT), (row(distinct=None), {}),
                                              (row(a2="Candidate A"), CONFLICT)]):
        rows.append({"question_id": f"q{i}", "source_split": "train", "partition": "train",
                     "solver_behavior": "adopted_critic", "label": classify_candidate(episode, POLICY, judgments)})
    write_labels(tmp_path / "labels.csv", rows)
    back = read_labels(tmp_path / "labels.csv")
    for r in rows:
        assert back[r["question_id"]]["decision_token_distinct"] == r["label"]["decision_token_distinct"]
        assert back[r["question_id"]]["judgment_pairs"] == r["label"]["judgment_pairs"]
        assert back[r["question_id"]]["strict_target"] == r["label"]["strict_target"]
    assert set(CANDIDATE_FIELDS) | set(TRACE_FIELDS) <= set(back["q0"])


# --- the judge ----------------------------------------------------------------

def judge_from(verdicts):
    calls = []

    def judge(question, left, right):
        calls.append((left, right))
        return verdicts.get((left, right), "different"), "RAW", {"generated_tokens": 1}
    return judge, calls


def test_judge_is_asked_only_about_the_pairs_labels_list():
    rows = [row(episode="a__natural"),                                   # A2 exact: one pair
            row(a2="Paris, France", episode="b__natural"),              # A2 matches neither: three
            row(a1="Paris", critic="Paris", a2="Paris", episode="c__natural"),   # exact: none
            row(a1=LONG, episode="d__natural")]                         # not answer-shaped: none
    judge, calls = judge_from({("Paris, France", "Paris"): "same"})
    records = judge_answer_pairs(rows, POLICY, judge)
    assert [(r["episode_id"], r["pair"]) for r in records] == [
        ("a__natural", "a1_vs_critic"), ("b__natural", "a1_vs_critic"),
        ("b__natural", "a2_vs_a1"), ("b__natural", "a2_vs_critic")]
    assert calls[0] == ("London", "Paris") and records[3]["relation"] == "same"


def test_judgments_written_and_read_back_finish_the_labels(tmp_path):
    rows = [row(a2="Paris, France", episode="b__natural")]
    judge, _ = judge_from({("Paris, France", "Paris"): "same"})
    write_jsonl(tmp_path / "judgments.jsonl", judge_answer_pairs(rows, POLICY, judge))
    judgments = read_judgments(tmp_path / "judgments.jsonl")
    label = classify_candidate(rows[0], POLICY, judgments["b__natural"])
    assert label["resolved_candidate"] == "reviewer" and label["strict_target"] == 1


def test_judging_refuses_repeated_episodes_and_policies_without_candidate_rules(tmp_path):
    judge, _ = judge_from({})
    with pytest.raises(ValueError, match="more than once"):
        judge_answer_pairs([row(), row()], POLICY, judge)
    with pytest.raises(ValueError, match="no candidate rules"):
        judge_answer_pairs([row()], V012_QC, judge)
    record = {"episode_id": "q__natural", "pair": "a1_vs_critic", "relation": "same"}
    write_jsonl(tmp_path / "twice.jsonl", [record, record])
    with pytest.raises(ValueError, match="judged twice"):
        read_judgments(tmp_path / "twice.jsonl")


def test_judgments_from_other_inputs_or_another_policy_are_refused(tmp_path):
    write_jsonl(tmp_path / "judgments.jsonl", [{"episode_id": "q__natural", "pair": "a1_vs_critic",
                                                "relation": "different"}])
    provenance = {"label_policy": {"definition_sha256": POLICY.definition_sha256()},
                  "interactions_sha256": {"a/train": "h1", "a/validation": "h2"},
                  "judge": {"model_id": "judge-model"}, "prompt_sha256": "p"}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    judgments, record = load_run_judgments(tmp_path, POLICY, ["h2", "h1"])
    assert judgments == {"q__natural": {"a1_vs_critic": "different"}} and record["judge_model"] == "judge-model"
    with pytest.raises(ValueError, match="different interaction files"):
        load_run_judgments(tmp_path, POLICY, ["h1", "other"])
    with pytest.raises(ValueError, match="different policy definition"):
        load_run_judgments(tmp_path, V012_QC, ["h1", "h2"])


@pytest.mark.parametrize("raw,relation", [
    ("SAME", "same"), ("Different.", "different"), ("uncertain", "uncertain"),
    ("Yes", None), ("", None),
])
def test_validator_same_answer_judge_parses_one_word(raw, relation):
    validator = Validator(Mock(device="cpu"), Mock())
    validator._generate = Mock(return_value=raw)
    verdict = validator.judge_same_answer("Who won?", "Thirty-six", "36")
    assert verdict.relation == relation and verdict.raw_output == raw
    prompt = validator._generate.call_args.args[0]
    assert "Question: Who won?" in prompt and "Answer 1: Thirty-six" in prompt and "Answer 2: 36" in prompt


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_judge_script_writes_judgments_and_provenance(tmp_path, monkeypatch):
    script = load_script("judge_answer_pairs")
    results = tmp_path / "run" / "train"
    results.mkdir(parents=True)
    write_jsonl(results / "interactions.jsonl", [row(), row(a1="Paris", episode="c__natural")])
    role = {"id": "google/gemma-3-4b-it", "loader": "gemma3", "revision": None, "dtype": "bfloat16",
            "device": "cpu", "generation": {"max_new_tokens": 4, "do_sample": False}}
    (results / "resolved_config.yaml").write_text(yaml.safe_dump(
        {"roles": {"solver": role, "critic": {**role, "loader": "qwen3"}, "validator": role}}))
    monkeypatch.setattr(script, "load_role_models",
                        lambda specs: {r: (SimpleNamespace(device="cpu"), object()) for r in specs})
    monkeypatch.setattr(Validator, "_generate", lambda self, prompt: "DIFFERENT")
    output = tmp_path / "judgments"
    monkeypatch.setattr(sys, "argv", ["judge", "--interactions", str(results / "interactions.jsonl"),
                                      "--output", str(output)])
    script.main()
    judgments = read_judgments(output / "judgments.jsonl")
    assert judgments == {"q__natural": {"a1_vs_critic": "different"}}
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["episodes"] == 2 and provenance["judgments"] == 1
    assert provenance["label_policy"]["definition_sha256"] == POLICY.definition_sha256()
    assert provenance["judge"]["model_id"] == "google/gemma-3-4b-it"
    with pytest.raises(FileExistsError):
        script.main()


@pytest.mark.parametrize("extra", [
    ["--label-policy", "v0.2.0-candidate"],                     # candidate rules need judgments
    ["--label-policy", "v0.1.2-qc", "--judgments", "somewhere"],  # other policies refuse them
])
def test_prepare_refuses_a_policy_and_judgments_that_do_not_go_together(tmp_path, monkeypatch, extra):
    script = load_script("prepare_behavior_v01")
    monkeypatch.setattr(sys, "argv", ["prepare", "--full-results", str(tmp_path / "missing"),
                                      "--frozen-partitions", str(tmp_path / "missing.csv"),
                                      "--output", str(tmp_path / "out"), *extra])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2 and not (tmp_path / "out").exists()
