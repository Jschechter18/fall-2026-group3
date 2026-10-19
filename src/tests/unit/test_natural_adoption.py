import json

import pytest

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.experiments.anonymized_adoption import CandidateSolver
from mas_sae.experiments.natural_adoption import run_natural, screen_natural


def inputs(target="Rome", a1="Paris"):
    raw = json.dumps(dict(verdict="disagree", advocated_answer=target, explanation="A natural explanation."))
    feedback = Critic.parse_natural(raw)
    row = dict(question_id="2hop__1", question="Where?", ground_truth="Rome",
               solver_attempt_1=a1, solver_attempt_2="Historical output", critic_condition="natural",
               critic_raw_output=raw, critic_feedback=feedback.to_solver_text(), critic_advocated_answer=target)
    audit = dict(question_id=row["question_id"], partition="train", solver_a1=a1,
                 delivered_verdict="disagree", delivered_advocated_answer=target,
                 delivered_explanation=feedback.explanation, coherence="consistent", coherence_reasons="")
    return row, audit


def test_screen_is_independent_of_a2_and_existing_eligibility():
    row, audit = inputs()
    selected, counts = screen_natural([row], [audit])
    altered = dict(row, solver_attempt_2="Paris", solver_attempt_2_correct=True,
                   label={"eligible_primary": False})
    altered_audit = dict(audit, candidate_status="invalid", a2_compound="True", eligible_primary="False")
    again, again_counts = screen_natural([altered], [altered_audit])
    assert len(selected) == len(again) == 1
    assert counts == again_counts
    assert selected[0]["natural_screen"] == again[0]["natural_screen"]


def test_screen_rejects_stale_missing_duplicate_or_nontrain_audit():
    row, audit = inputs()
    for entries, message in (([], "Missing"), ([audit, audit], "Duplicate"),
                             ([dict(audit, solver_a1="Changed")], "Stale"),
                             ([dict(audit, partition="validation")], "train-only")):
        with pytest.raises(ValueError, match=message):
            screen_natural([row], entries)


def test_screen_excludes_agreement_and_uncertain_feedback():
    row, audit = inputs(target="Paris")
    assert screen_natural([row], [audit])[1]["excluded"] == {"no_clear_lexical_conflict": 1}
    row, audit = inputs()
    assert screen_natural([row], [dict(audit, coherence="uncertain")])[0] == []


@pytest.mark.parametrize("use_audit", [True, False])
def test_paired_run_preserves_natural_inputs_and_exports_separate_table(tmp_path, monkeypatch, use_audit):
    row, audit = inputs()
    rows, _ = screen_natural([row], [audit] if use_audit else None)
    output = tmp_path / "natural"

    class Original(Solver):
        def __init__(self):
            self.last_generation = {}

        def _generate(self, prompt):
            assert (output / "order_assignments.json").exists()
            assert row["critic_feedback"] in prompt
            return "Rome"

    class Candidate(CandidateSolver):
        def __init__(self):
            self.last_generation = {}

        def _generate(self, prompt):
            assert "your previous answer" in prompt
            assert "reviewer's recommendation" in prompt
            assert "A natural explanation" not in prompt
            return "Paris"

    ex = dict(question="Where?", answer="Rome", answer_aliases=[], paragraphs=[])
    summary = run_natural(rows, {row["question_id"]: ex}, Original(), Candidate(), output)
    assert summary["conditions"]["original"]["adopted"] == 1
    assert summary["conditions"]["labeled_randomized"]["retained_a1"] == 1
    assert summary["paired"]["response_transitions"] == {"adopted -> retained_a1": 1}
    assert summary["paired"]["accuracy_losses"] == 1
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 2
    assert "natural,labeled_randomized" in (output / "slide_table.csv").read_text()
    assert (output / "comparisons.csv").exists()
    import runpy
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    report = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["summarize_adoption_results.py",
        "--results-root", str(root / "results/adoption_experiment"),
        "--natural-run", str(output), "--output", str(report)])
    runpy.run_path(str(root / "scripts/experiments/summarize_adoption_results.py"), run_name="__main__")
    assert (report / "natural_slide_table.csv").exists()
    assert "separate paired sample" in (report / "presentation_summary.md").read_text()
    from mas_sae.experiments.natural_adoption import load_natural_run
    assert load_natural_run(output) == summary
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({**summary, "paired": {}}))
    with pytest.raises(ValueError, match="summary does not match"):
        load_natural_run(output)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with pytest.raises(FileExistsError):
        run_natural(rows, {row["question_id"]: ex}, Original(), Candidate(), output)


def test_gold_alias_overlap_is_explicit_not_silently_other(tmp_path):
    row, audit = inputs(target="Saint Paul", a1="St. Paul")
    rows, _ = screen_natural([row], [audit])

    class AliasSolver(CandidateSolver):
        def __init__(self):
            self.last_generation = {}

        def _generate(self, prompt):
            return "St. Paul"

    row_ex = dict(question="Where?", answer="Rome", answer_aliases=["Saint Paul", "St. Paul"], paragraphs=[])
    result = run_natural(rows, {row["question_id"]: row_ex}, AliasSolver(), AliasSolver(), tmp_path / "alias")
    assert result["conditions"]["original"]["ambiguous_overlap"] == 1
    assert result["conditions"]["original"]["other"] == 0


def test_prepare_natural_cli_freezes_fresh_train_only_sample(tmp_path, monkeypatch):
    import csv
    from pathlib import Path
    import runpy
    import sys
    from mas_sae.experiments.adoption_experiment import digest

    root = Path(__file__).resolve().parents[3]
    row, audit = inputs()
    previous_row = dict(row, question_id="2hop__old")
    heldout = dict(row, question_id="2hop__heldout")
    source = tmp_path / "interactions.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in (row, previous_row, heldout)) + "\n")
    partitions = tmp_path / "partitions.csv"
    partitions.write_text("question_id,source_split,partition\n2hop__1,train,train\n2hop__old,train,train\n2hop__heldout,train,test\n")
    audit_path = tmp_path / "audit.csv"
    with audit_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit))
        writer.writeheader()
        writer.writerow(audit)
    prior = {"rows": [previous_row]}
    prior["sha256"] = digest(prior)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior))
    output = tmp_path / "sample"
    monkeypatch.setattr(sys, "argv", ["adoption_experiment.py", "prepare-natural",
        "--interactions", str(source), "--partitions", str(partitions),
        "--config", str(root / "configs/collection/v2/natural_4b_full_train.yaml"),
        "--audit", str(audit_path), "--exclude-sample", str(prior_path),
        "--size", "1", "--output", str(output)])
    runpy.run_path(str(root / "scripts/experiments/adoption_experiment.py"), run_name="__main__")
    packet = json.loads((output / "sample.json").read_text())
    checksum = packet.pop("sha256")
    assert digest(packet) == checksum
    assert packet["protocol"] == "adoption_natural_sample_v1"
    assert [r["question_id"] for r in packet["rows"]] == ["2hop__1"]
    assert packet["screening"]["input_questions"] == 1
    assert str(audit_path) in packet["input_hashes"]


def test_screen_without_audit_uses_saved_feedback_and_marks_coherence_unassessed():
    row, _ = inputs()
    selected, counts = screen_natural([row])
    assert selected[0]["natural_screen"]["coherence"] == "not_assessed"
    assert counts["method"] == "saved_feedback_lexical_conflict"
    changed = dict(row, solver_attempt_2="Paris", label={"eligible_primary": False})
    assert screen_natural([changed])[1] == counts
    agreed, _ = inputs(target="Paris")
    assert screen_natural([agreed])[0] == []
    missing = dict(row, critic_raw_output="invalid JSON", critic_advocated_answer=None,
                   critic_feedback=Critic.parse_natural("invalid JSON").to_solver_text())
    assert screen_natural([missing])[1]["excluded"] == {"missing_recommendation": 1}


@pytest.mark.parametrize("a1,target,reason", [
    ("I don't know", "Rome", "a1_nonanswer"),
    ("Paris", "I don't know", "critic_refusal"),
])
def test_input_screen_reuses_answer_kind_rules_without_historical_a2(a1, target, reason):
    row, audit = inputs(target=target, a1=a1)
    for audit_rows in (None, [audit]):
        assert screen_natural([row], audit_rows)[1]["excluded"] == {reason: 1}
        row["solver_attempt_2"] = "A completely different historical output"
        assert screen_natural([row], audit_rows)[1]["excluded"] == {reason: 1}
