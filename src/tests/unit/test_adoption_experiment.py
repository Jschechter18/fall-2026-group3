import csv
import json

import pytest

from mas_sae.experiments.adoption_experiment import ARMS, feedback_for, run_experiment, sample_rows
from mas_sae.agents.solver import Solver


def episode(qid, correct=True):
    return dict(question_id=qid, question="Where?", ground_truth="Liverpool",
                solver_attempt_1="Liverpool", solver_attempt_1_correct=correct,
                critic_condition="natural", critic_feedback_correct=False,
                critic_verdict="disagree", critic_advocated_answer="London",
                critic_noncommittal=False, solver_attempt_2="London",
                critic_feedback="The reviewer disagrees with your previous answer. "
                                "The reviewer advocates this answer: London Reviewer explanation: Look here.")


def partitions(tmp_path, rows):
    path = tmp_path / "partitions.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "source_split", "partition"])
        writer.writeheader()
        for qid, partition in rows:
            writer.writerow(dict(question_id=qid, source_split="train", partition=partition))
    return path


def test_sampling_is_order_invariant_and_excludes_heldout(tmp_path):
    rows = [episode(f"{hop}hop__{i}", bool(i % 2)) for hop in (2, 3, 4) for i in range(8)]
    heldout = episode("2hop__heldout")
    path = partitions(tmp_path, [(r["question_id"], "train") for r in rows] + [(heldout["question_id"], "test")])
    first, stats = sample_rows(rows + [heldout], path, size=12)
    second, other_stats = sample_rows(list(reversed(rows + [heldout])), path, size=12)
    assert first == second
    assert stats == other_stats
    assert len({r["question_id"] for r in first}) == 12
    assert all(s["selected"] == 2 for s in stats.values())
    assert heldout not in first
    # A2/adoption must never influence selection.
    altered = [{**r, "solver_attempt_2": "different"} for r in rows]
    other, _ = sample_rows(altered, path, size=12)
    assert [r["question_id"] for r in first] == [r["question_id"] for r in other]


def test_sampling_fails_closed(tmp_path):
    row = episode("2hop__1")
    path = partitions(tmp_path, [(row["question_id"], "train")])
    with pytest.raises(ValueError, match="Duplicate Natural"):
        sample_rows([row, row], path, size=1)
    with pytest.raises(ValueError, match="absent"):
        sample_rows([episode("2hop__missing")], path, size=1)
    with pytest.raises(ValueError, match="Not enough"):
        sample_rows([row], path, size=2)


def test_feedback_ablations_preserve_target_and_skip_unknown_formats():
    row = episode("2hop__1")
    assert feedback_for(row, "original") == row["critic_feedback"]
    assert feedback_for(row, "no_verdict").startswith("The reviewer advocates")
    assert feedback_for(row, "answer_only") == "The reviewer advocates this answer: London"
    assert feedback_for({**row, "critic_advocated_answer": None}, "answer_only") is None
    assert feedback_for({**row, "critic_feedback": "unfamiliar"}, "no_verdict") is None


class FakeSolver(Solver):
    def __init__(self):
        self.calls = []
        self.last_generation = {"budget_reached": False}

    def revise(self, question, paragraphs, previous_answer, feedback):
        self.calls.append((question, paragraphs, previous_answer, feedback))
        return "Liverpool" if feedback.startswith("No additional") else "London"


def test_paired_replay_and_automatic_outputs(tmp_path):
    row = episode("2hop__1")
    example = dict(id=row["question_id"], question="Where?", answer="Liverpool",
                   paragraphs=[dict(idx=0, title="Source", paragraph_text="Liverpool.")])
    solver = FakeSolver()
    output = tmp_path / "run"
    summary = run_experiment([row], {example["id"]: example}, solver, output)
    assert len(solver.calls) == len(ARMS)
    assert all(call[2] == "Liverpool" for call in solver.calls)
    assert summary["arms"]["original"]["candidate_adoption_rate"] == 1
    assert summary["arms"]["no_feedback"]["candidate_adoption_rate"] is None
    assert summary["arms"]["no_feedback"]["retained_a1"] == 1
    assert summary["arms"]["original"]["harmed"] == 1
    assert summary["arms"]["answer_only"]["paired_with_original"]["both_candidate_eligible"] == 1
    records = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all("candidate_label" in record for record in records)
    assert json.loads((output / "summary.json").read_text()) == summary
    assert not (output / "review.csv").exists()
    assert not (output / "review_private_key.json").exists()
    with pytest.raises(FileExistsError):
        run_experiment([row], {example["id"]: example}, solver, output)


def test_load_sample_accepts_existing_manifest_and_rejects_tampering(tmp_path):
    from pathlib import Path
    from mas_sae.experiments.adoption_experiment import load_sample

    # Compatibility with the existing checksum format, not a newly generated one.
    original = Path(__file__).resolve().parents[3] / "results/adoption_experiment/sample_100q/sample.json"
    packet, checksum = load_sample(original)
    assert len(packet["rows"]) == 100
    assert checksum == json.loads(original.read_text())["sha256"]
    packet["seed"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps({**packet, "sha256": checksum}))
    with pytest.raises(ValueError, match="Sample/config checksum mismatch"):
        load_sample(path)
    with pytest.raises(ValueError, match="Excluded sample checksum mismatch"):
        load_sample(path, checksum_error="Excluded sample checksum mismatch")
