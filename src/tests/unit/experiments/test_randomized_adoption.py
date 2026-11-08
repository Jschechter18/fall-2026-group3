import csv
import json
from collections import Counter, defaultdict

import pytest

from mas_sae.agents.solver import Solver
from mas_sae.experiments.anonymized_adoption import CandidateSolver
from mas_sae.experiments.controlled_adoption import PROTOCOL as SOURCE_PROTOCOL
from mas_sae.experiments.randomized_adoption import assign_orders, order_only_baseline, run_randomized_order


def test_assignment_balanced_reproducible_and_independent_of_a2():
    rows, examples = [], {}
    for hop in (2, 3, 4):
        for i in range(9):
            qid = f"{hop}hop__{i}"
            rows.append(dict(question_id=qid, solver_attempt_1="Liverpool" if i % 2 else "London"))
            examples[qid] = dict(answer="Liverpool")
    first = assign_orders(rows, examples, 42)
    assert first == assign_orders(list(reversed(rows)), examples, 42)
    assert first == assign_orders([{**r, "solver_attempt_2": "changed", "solver_accepted_feedback": True} for r in rows], examples, 42)
    assert first != assign_orders(rows, examples, 43)
    counts = Counter(first.values())
    assert abs(counts["a1_first"] - counts["critic_first"]) <= 1
    groups = defaultdict(Counter)
    for row in rows:
        groups[(row["question_id"].split("__")[0], row["solver_attempt_1"])][first[row["question_id"]]] += 1
    assert all(abs(c["a1_first"] - c["critic_first"]) <= 1 for c in groups.values())


def test_assignment_rejects_duplicate_questions():
    row = dict(question_id="2hop__1", solver_attempt_1="Liverpool")
    with pytest.raises(ValueError, match="unique"):
        assign_orders([row, row], {row["question_id"]: {"answer": "Liverpool"}}, 42)


def test_fixed_order_baseline_reports_confusion_and_handles_single_class():
    rows = [dict(candidate_order=order, adopted_target=label) for order, label in
            [("critic_first", True), ("critic_first", False), ("a1_first", True), ("a1_first", False)]]
    scores = order_only_baseline(rows)
    assert scores["accuracy"] == scores["balanced_accuracy"] == 0.5
    assert scores["confusion_matrix"] == dict(true_positive=1, false_positive=1, true_negative=1, false_negative=1)
    assert order_only_baseline([])["accuracy"] is None
    assert order_only_baseline(rows[:1])["balanced_accuracy"] is None


def test_randomized_run_saves_assignment_before_inference_and_preserves_skips(tmp_path):
    qid = "2hop__test"
    example = dict(id=qid, question="Where?", answer="Liverpool", answer_aliases=[],
                   paragraphs=[dict(idx=0, title="Source", paragraph_text="Liverpool.")])
    row = dict(question_id=qid, question=example["question"], ground_truth="Liverpool", solver_attempt_1="Liverpool")
    base = dict(row, answer_aliases=[], protocol=SOURCE_PROTOCOL, stratum="2hop / a1=True / critic=False",
                context=Solver.format_paragraphs(example["paragraphs"]))
    sources = [dict(base, arm=arm, status="completed", critic_advocated_answer=target,
                    target_metadata={"answer": target}, critic_feedback=target, solver_attempt_2=target)
               for arm, target in [("plausible_incorrect", "London"), ("obvious_incorrect", "Antarctica")]]
    sources.append(dict(base, arm="correct_correction", status="skipped_requires_incorrect_a1", target_metadata={}))
    output = tmp_path / "randomized"

    class Stub(CandidateSolver):
        def __init__(self):
            self.calls = []
            self.last_generation = {}

        def _generate(self, prompt):
            assert (output / "order_assignments.json").exists()
            assert "your previous answer" in prompt and "reviewer's recommendation" in prompt
            self.calls.append(prompt)
            return next(line.split(": ", 1)[1] for line in prompt.splitlines() if line.startswith("Candidate A"))

    solver = Stub()
    summary = run_randomized_order([row], {qid: example}, sources, solver, output, seed=42)
    assignments = json.loads((output / "order_assignments.json").read_text())
    records = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(solver.calls) == 2 and len(records) == 3
    assert {r["candidate_order"] for r in records} == {assignments[qid]}
    assert all(r["instruction_condition"] == "labeled" for r in records)
    assert summary["completed"] == 2 and summary["skipped"] == 1
    assert summary["distinct_completed_questions"] == 1
    assert summary["order_only_baseline"]["all_completed_adoption_vs_nonadoption"]["accuracy"] == 1
    with (output / "results.csv").open(newline="") as f:
        results = list(csv.DictReader(f))
    with (output / "comparisons.csv").open(newline="") as f:
        comparisons = list(csv.DictReader(f))
    assert len(results) == len(comparisons) == 3
    assert results[-1]["adopted_target"] == ""
    assert comparisons[0]["assigned_order"] == assignments[qid]
    assert json.loads((output / "provenance.json").read_text())["order_seed"] == 42
    with pytest.raises(FileExistsError):
        run_randomized_order([row], {qid: example}, sources, solver, output)
