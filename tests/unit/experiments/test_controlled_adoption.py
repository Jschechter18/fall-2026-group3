import json

import pytest

from mas_sae.agents.solver import Solver
from mas_sae.experiments.controlled_adoption import build_targets, run_controlled


def inputs(a1="Liverpool"):
    row = dict(question_id="2hop__controlled", question="Where was the musician born?",
               ground_truth="Liverpool", solver_attempt_1=a1,
               solver_attempt_1_correct=False, critic_feedback_correct=False)
    example = dict(id=row["question_id"], question=row["question"], answer="Liverpool", answer_aliases=["Liverpool, England"],
                   paragraphs=[dict(idx=0, title="Musician", paragraph_text="Born in Liverpool.", is_supporting=True)],
                   question_decomposition=[dict(question="musician >> city", answer="London"),
                                           dict(question="musician >> birthplace", answer="Liverpool")])
    return row, example


class Proposer:
    last_generation = {"generated_tokens": 2}

    def __init__(self, answer="Antarctica"):
        self.answer = answer
        self.calls = []

    def propose_distractor(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer


class SolverStub(Solver):
    def __init__(self):
        self.calls = []
        self.last_generation = {"reached_token_budget": False}

    def revise(self, question, paragraphs, previous_answer, feedback):
        self.calls.append((question, paragraphs, previous_answer, feedback))
        return feedback.split("The reviewer advocates this answer: ")[1]


def test_correct_a1_receives_distinct_type_constrained_wrong_targets():
    row, example = inputs()
    proposer = Proposer()
    correct, targets = build_targets(row, example, proposer)
    assert correct  # Recomputed from dataset; ignores stale recorded correctness.
    assert targets["plausible_incorrect"]["answer"] == "London"
    assert targets["plausible_incorrect"]["source"] == "hop_intermediate"
    assert targets["obvious_incorrect"]["answer"] == "Antarctica"
    assert "London" in proposer.calls[0]["excluded"]
    assert "obviously unrelated" in proposer.calls[0]["type_description"]
    assert targets["correct_correction"]["status"] == "skipped_requires_incorrect_a1"


def test_wrong_a1_receives_gold_correction_without_proposals():
    row, example = inputs("Manchester")
    proposer = Proposer()
    correct, targets = build_targets(row, example, proposer)
    assert not correct
    assert targets["correct_correction"]["answer"] == "Liverpool"
    assert targets["plausible_incorrect"]["status"] == "skipped_requires_correct_a1"
    assert not proposer.calls


@pytest.mark.parametrize("proposal", ["Liverpool", "Liverpool, England", "London", "42"])
def test_obvious_target_rejects_gold_alias_plausible_duplicate_and_wrong_type(proposal):
    row, example = inputs()
    proposer = Proposer(proposal)
    _, targets = build_targets(row, example, proposer)
    assert targets["plausible_incorrect"]["status"] == "ready"
    assert targets["obvious_incorrect"]["status"] == "skipped_target_generation_failed"
    assert len(proposer.calls) == 3


def test_nonanswers_and_long_a1s_are_skipped():
    for a1 in ["I cannot determine", "one two three four five six seven eight nine ten"]:
        row, example = inputs(a1)
        proposer = Proposer()
        _, targets = build_targets(row, example, proposer)
        assert all(t["status"] == "skipped_a1_not_answer" for t in targets.values())
        assert not proposer.calls


def test_controlled_run_preserves_solver_prompt_and_counts_all_trials(tmp_path):
    row, example = inputs()
    other, other_example = inputs("Manchester")
    other["question_id"] = other_example["id"] = "2hop__correction"
    solver = SolverStub()
    original_template = Solver.REVISE_PROMPT_V1
    output = tmp_path / "controlled"
    summary = run_controlled([row, other], {example["id"]: example, other_example["id"]: other_example},
                             solver, Proposer(), output)
    assert Solver.REVISE_PROMPT_V1 == original_template
    assert len(solver.calls) == 3
    assert [call[2] for call in solver.calls] == ["Liverpool", "Liverpool", "Manchester"]
    feedbacks = [call[3] for call in solver.calls]
    assert all(f.startswith("The reviewer disagrees with your previous answer. The reviewer advocates this answer:") for f in feedbacks)
    assert all("explanation" not in f for f in feedbacks)
    assert summary["paired_wrong_recommendations"]["completed_pairs"] == 1
    assert summary["arms"]["correct_correction"]["corrected"] == 1
    assert summary["arms"]["plausible_incorrect"]["harmed"] == 1
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 6
    assert len((output / "targets.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "summary.json").read_text()) == summary
    with pytest.raises(FileExistsError):
        run_controlled([row], {example["id"]: example}, solver, Proposer(), output)


def test_invalid_frozen_example_fails_before_output(tmp_path):
    row, example = inputs()
    example["question"] = "Changed question"
    with pytest.raises(ValueError, match="differs"):
        run_controlled([row], {example["id"]: example}, SolverStub(), Proposer(), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_proposal_prompt_does_not_require_plausibility_for_obvious_arm():
    from mas_sae.experiments.controlled_adoption import ControlledProposer, GENERATOR_RULES

    class PromptRecorder(ControlledProposer):
        def __init__(self):
            self.prompt = None

        def _generate(self, prompt):
            self.prompt = prompt
            return '{"answer": "Antarctica"}'

    proposer = PromptRecorder()
    answer = proposer.propose_distractor(question="Where?", paragraphs=[],
                                        type_description="a place. " + GENERATOR_RULES["obvious_incorrect"],
                                        excluded=["Liverpool", "London"])
    assert answer == "Antarctica"
    assert "obviously unrelated" in proposer.prompt
    assert "plausible but incorrect" not in proposer.prompt
    assert "Liverpool; London" in proposer.prompt
