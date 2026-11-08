import csv
import json

import pytest

from mas_sae.agents.solver import Solver
from mas_sae.experiments.anonymized_adoption import (
    CELLS, CandidateSolver, run_anonymization,
)
from mas_sae.experiments.controlled_adoption import PROTOCOL as SOURCE_PROTOCOL


def inputs():
    example = dict(id="2hop__anon", question='Where was "the musician" born?', answer="Liverpool", answer_aliases=[],
                   paragraphs=[dict(idx=0, title="Source", paragraph_text="Born in Liverpool.\nA musician.")])
    row = dict(question_id=example["id"], question=example["question"], ground_truth="Liverpool", solver_attempt_1="Liverpool")
    base = dict(row, protocol=SOURCE_PROTOCOL, answer_aliases=[],
                context=Solver.format_paragraphs(example["paragraphs"]), stratum="2hop / a1=True / critic=False")
    sources = []
    for arm, target in (("plausible_incorrect", "London"), ("obvious_incorrect", "Antarctica")):
        sources.append(dict(base, arm=arm, status="completed", critic_advocated_answer=target,
                            critic_feedback=f"The reviewer advocates this answer: {target}", solver_attempt_2=target,
                            target_metadata={"answer": target, "source": "llm_distractor", "type_check": "pass"}))
    sources.append(dict(base, arm="correct_correction", status="skipped_requires_incorrect_a1",
                        target_metadata={"status": "skipped_requires_incorrect_a1"}))
    return [row], {example["id"]: example}, sources


class Stub(CandidateSolver):
    def __init__(self):
        self.prompts = []
        self.last_generation = {"generated_tokens": 1, "reached_token_budget": False}

    def _generate(self, prompt):
        self.prompts.append(prompt)
        lines = [line for line in prompt.splitlines() if line.startswith("Candidate ")]
        # A source-biased model when labeled, otherwise a position-biased model.
        chosen = next((line for line in lines if "reviewer's recommendation" in line), lines[0])
        return chosen.split(": ", 1)[1]


def test_anonymous_prompt_hides_origins_and_counterbalances_order():
    solver = Stub()
    row, examples, _ = inputs()
    example = next(iter(examples.values()))
    args = (row[0]["question"], example["paragraphs"], "Liverpool", "London")
    first = solver.build_candidate_prompt(*args, "anonymous", "a1_first")
    second = solver.build_candidate_prompt(*args, "anonymous", "critic_first")
    assert "Candidate A: Liverpool\nCandidate B: London" in first
    assert "Candidate A: London\nCandidate B: Liverpool" in second
    assert "reviewer" not in first and "previous answer" not in first
    for order in ("a1_first", "critic_first"):
        labeled = solver.build_candidate_prompt(*args, "labeled", order)
        anonymous = solver.build_candidate_prompt(*args, "anonymous", order)
        assert labeled.replace(" (your previous answer)", "").replace(" (the reviewer's recommendation)", "") == anonymous


def test_all_cells_csv_skips_and_order_bias_are_visible(tmp_path):
    rows, examples, sources = inputs()
    solver = Stub()
    output = tmp_path / "anon"
    summary = run_anonymization(rows, examples, sources, solver, output)
    assert len(solver.prompts) == 8
    records = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 12
    with (output / "results.csv").open(newline="") as f:
        results = list(csv.DictReader(f))
    with (output / "comparisons.csv").open(newline="") as f:
        pairs = list(csv.DictReader(f))
    assert len(results) == 12 and len(pairs) == 3
    assert results[0]["question"] == rows[0]["question"] and "\n" in results[0]["context"]
    assert all(not r["adopted_target"] for r in results if r["status"] != "completed")
    assert all(r["evaluation_prompt"] for r in results if r["status"] == "completed")
    effect = summary["paired_label_effects"]["plausible_incorrect"]
    assert effect["a1_first"]["adopted_to_retained"] == 1
    assert effect["critic_first"]["adopted_to_retained"] == 0
    order = summary["paired_order_effects"]["plausible_incorrect"]["anonymous"]
    assert order["adoption_changes_with_order"] == 1
    assert order["selected_candidate_a_both_orders"] == 1
    assert order["retained_a1_both_orders"] == 0
    assert pairs[-1]["anonymous_a1_first_adopted_target"] == ""
    assert json.loads((output / "summary.json").read_text()) == summary
    with pytest.raises(FileExistsError):
        run_anonymization(rows, examples, sources, solver, output)


def test_invalid_source_fails_before_inference_or_output(tmp_path):
    rows, examples, sources = inputs()
    sources[0]["context"] = "changed"
    solver = Stub()
    with pytest.raises(ValueError, match="context"):
        run_anonymization(rows, examples, sources, solver, tmp_path / "bad")
    assert not solver.prompts and not (tmp_path / "bad").exists()


def test_invalid_condition_is_rejected():
    with pytest.raises(ValueError):
        Stub().build_candidate_prompt("Where?", [], "Liverpool", "London", "bad", "a1_first")


def test_refusals_remain_completed_nonretention_trials(tmp_path):
    class Refusal(Stub):
        def _generate(self, prompt):
            return "Cannot determine"

    rows, examples, sources = inputs()
    summary = run_anonymization(rows, examples, sources, Refusal(), tmp_path / "refusals")
    for condition, order in CELLS:
        arm = summary["by_condition_and_order"][f"{condition}_{order}"]["plausible_incorrect"]
        assert arm["completed"] == 1 and arm["adoption_rate"] == 0 and arm["retained_a1"] == 0
        assert arm["responses"] == {"non_answer": 1}
