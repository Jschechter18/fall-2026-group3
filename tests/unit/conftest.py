import json

import pytest

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.experiments.adoption_experiment import digest
from mas_sae.experiments.controlled_adoption import PROTOCOL as CONTROLLED_PROTOCOL
from mas_sae.experiments.randomized_adoption import PROTOCOL as RANDOMIZED_PROTOCOL
from mas_sae.experiments.records import write_jsonl


@pytest.fixture
def adoption_results_root(tmp_path):
    """Create a complete, small adoption-results tree for unit tests."""
    root = tmp_path / "adoption_results"
    solver = {"id": "fixture-solver"}
    dataset = {"id": "fixture-dataset"}
    assignments = {
        "2hop__wrong": "critic_first",
        "2hop__correct": "a1_first",
    }

    sample_rows = []
    for question_id, ground_truth, attempt, recommendation in (
        ("2hop__wrong", "Rome", "Paris", "Berlin"),
        ("2hop__correct", "Paris", "Paris", "Paris"),
    ):
        raw = json.dumps({
            "verdict": "disagree",
            "advocated_answer": recommendation,
            "explanation": "Fixture feedback.",
        })
        feedback = Critic.parse_natural(raw)
        sample_rows.append({
            "question_id": question_id,
            "question": f"Question for {question_id}?",
            "ground_truth": ground_truth,
            "solver_attempt_1": attempt,
            "critic_raw_output": raw,
            "critic_feedback": feedback.to_solver_text(),
            "critic_advocated_answer": recommendation,
            "critic_noncommittal": feedback.noncommittal,
        })

    sample = {
        "protocol": "adoption_feedback_ablation_v1",
        "seed": 7,
        "solver": solver,
        "dataset": dataset,
        "input_hashes": {},
        "rows": sample_rows,
    }
    sample["sha256"] = digest(sample)
    sample_directory = root / "sample_100q"
    sample_directory.mkdir(parents=True)
    (sample_directory / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")

    arms = {
        "2hop__wrong": "plausible_wrong",
        "2hop__correct": "correct_correction",
    }

    def method_row(sample_row, *, instruction_condition, candidate_order=None,
                   adopted=False, retained=False):
        return {
            "question_id": sample_row["question_id"],
            "arm": arms[sample_row["question_id"]],
            "instruction_condition": instruction_condition,
            "candidate_order": candidate_order,
            "status": "completed",
            "question": sample_row["question"],
            "context": "Fixture context.",
            "ground_truth": sample_row["ground_truth"],
            "solver_attempt_1": sample_row["solver_attempt_1"],
            "critic_advocated_answer": sample_row["critic_advocated_answer"],
            "adopted_target": adopted,
            "retained_a1": retained,
            "a2_correct": arms[sample_row["question_id"]] == "correct_correction" and adopted,
        }

    def rows_for(instruction_condition, outcomes, candidate_order=None):
        return [
            method_row(row, instruction_condition=instruction_condition,
                       candidate_order=candidate_order,
                       adopted=outcome == "adopted", retained=outcome == "retained")
            for row, outcome in zip(sample_rows, outcomes)
        ]

    controlled = rows_for("original", ("adopted", "retained"))
    verification = rows_for("evidence_verification", ("adopted", "adopted"))
    anonymized = []
    for condition, order, outcomes in (
        ("labeled", "a1_first", ("retained", "retained")),
        ("labeled", "critic_first", ("adopted", "adopted")),
        ("anonymous", "a1_first", ("other", "retained")),
        ("anonymous", "critic_first", ("adopted", "other")),
    ):
        anonymized.extend(rows_for(condition, outcomes, order))
    randomized = [
        method_row(row, instruction_condition="labeled_randomized",
                   candidate_order=assignments[row["question_id"]],
                   adopted=row["question_id"] == "2hop__wrong",
                   retained=row["question_id"] == "2hop__correct")
        for row in sample_rows
    ]

    for folder, rows, protocol in (
        ("controlled_100q", controlled, CONTROLLED_PROTOCOL),
        ("verification_100q", verification, "fixture_verification"),
        ("anonymization_100q", anonymized, "fixture_anonymization"),
        ("randomized_order_100q", randomized, RANDOMIZED_PROTOCOL),
    ):
        directory = root / folder
        directory.mkdir(parents=True)
        write_jsonl(directory / "episodes.jsonl", rows)
        provenance = {"protocol": protocol, "sample_sha256": sample["sha256"]}
        summary = {"protocol": protocol}
        if folder == "controlled_100q":
            provenance.update({
                "solver": solver,
                "dataset": dataset,
                "revise_template": Solver.REVISE_PROMPT_V1,
            })
        if folder == "randomized_order_100q":
            provenance["order_seed"] = 11
            summary["order_only_baseline"] = {
                "all_completed_adoption_vs_nonadoption": {"balanced_accuracy": 1.0}
            }
            (directory / "order_assignments.json").write_text(
                json.dumps(assignments, indent=2) + "\n"
            )
        (directory / "provenance.json").write_text(json.dumps(provenance) + "\n")
        (directory / "summary.json").write_text(json.dumps(summary) + "\n")

    return root
