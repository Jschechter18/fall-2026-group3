import csv
import json

import pytest

from mas_sae.agents.solver import Solver
from mas_sae.experiments.controlled_adoption import PROTOCOL as SOURCE_PROTOCOL
from mas_sae.experiments.verification_adoption import (
    EvidenceVerificationSolver, VERIFICATION_INSTRUCTION, load_controlled_source,
    run_verification,
)


def fixture_inputs():
    example = dict(id="2hop__test", question='Where was "the musician" born?', answer="Liverpool",
                   answer_aliases=[], paragraphs=[dict(idx=0, title="Source", paragraph_text="Born in Liverpool.\nA musician.")])
    row = dict(question_id=example["id"], question=example["question"], ground_truth=example["answer"], solver_attempt_1="Liverpool")
    base = dict(question_id=example["id"], question=example["question"], ground_truth=example["answer"],
                answer_aliases=[], solver_attempt_1="Liverpool", a1_correct=True,
                context=Solver.format_paragraphs(example["paragraphs"]), protocol=SOURCE_PROTOCOL,
                stratum="2hop / a1=True / critic=False")
    sources = []
    for arm, target in (("plausible_incorrect", "London"), ("obvious_incorrect", "Antarctica")):
        sources.append(dict(base, arm=arm, status="completed", critic_advocated_answer=target,
                            critic_feedback=f"The reviewer disagrees with your previous answer. The reviewer advocates this answer: {target}",
                            solver_attempt_2=target, target_metadata={"answer": target, "source": "llm_distractor", "type_check": "pass"}))
    sources.append(dict(base, arm="correct_correction", status="skipped_requires_incorrect_a1",
                        target_metadata={"status": "skipped_requires_incorrect_a1"}))
    return [row], {example["id"]: example}, sources


class BaselineStub(Solver):
    def __init__(self):
        self.calls = []
        self.last_generation = {"generated_tokens": 1, "reached_token_budget": False}

    def revise(self, question, paragraphs, previous_answer, feedback):
        self.calls.append((question, paragraphs, previous_answer, feedback))
        return feedback.split("The reviewer advocates this answer: ")[1]


class VerificationStub(EvidenceVerificationSolver, BaselineStub):
    def revise(self, question, paragraphs, previous_answer, feedback):
        self.calls.append((question, paragraphs, previous_answer, feedback))
        return previous_answer


def test_only_instruction_changes_and_csv_contains_all_pairs(tmp_path):
    rows, examples, sources = fixture_inputs()
    solvers = {"original": BaselineStub(), "evidence_verification": VerificationStub()}
    summary = run_verification(rows, examples, sources, solvers, tmp_path / "run")
    assert solvers["original"].calls == solvers["evidence_verification"].calls
    for call in solvers["original"].calls:
        old = solvers["original"].build_revise_prompt(*call)
        new = solvers["evidence_verification"].build_revise_prompt(*call)
        assert new.replace(VERIFICATION_INSTRUCTION, "") == old
    for arm in ("plausible_incorrect", "obvious_incorrect"):
        assert summary["paired_comparisons"][arm]["adopted_to_retained"] == 1
        assert summary["paired_comparisons"][arm]["accuracy_gains"] == 1
        assert summary["paired_comparisons"][arm]["original_matches_source_a2"] == 1
    with (tmp_path / "run/results.csv").open(newline="") as f:
        results = list(csv.DictReader(f))
    with (tmp_path / "run/comparisons.csv").open(newline="") as f:
        pairs = list(csv.DictReader(f))
    assert len(results) == 6 and len(pairs) == 3
    assert results[0]["question"] == examples[rows[0]["question_id"]]["question"]
    assert "\n" in results[0]["context"]
    assert results[0]["solver_attempt_2"] == "London"
    assert results[1]["solver_attempt_2"] == "Liverpool"
    assert results[-1]["status"] == "skipped_requires_incorrect_a1"
    assert results[-1]["adopted_target"] == ""
    assert pairs[-1]["accuracy_gain"] == ""
    assert pairs[0]["adopted_to_retained"] == "True"
    assert json.loads((tmp_path / "run/provenance.json").read_text())["revision_templates"]["original"] == Solver.REVISE_PROMPT_V1
    with pytest.raises(FileExistsError):
        run_verification(rows, examples, sources, solvers, tmp_path / "run")


@pytest.mark.parametrize("change", ["missing_arm", "duplicate", "changed_a1", "changed_context", "changed_target"])
def test_source_mismatches_fail_before_output(tmp_path, change):
    rows, examples, sources = fixture_inputs()
    if change == "missing_arm":
        sources.pop()
    elif change == "duplicate":
        sources.append(sources[0])
    elif change == "changed_a1":
        sources[0]["solver_attempt_1"] = "Different"
    elif change == "changed_context":
        sources[0]["context"] = "Different"
    else:
        sources[0]["critic_advocated_answer"] = "Different"
    with pytest.raises(ValueError):
        run_verification(rows, examples, sources, {"original": BaselineStub(), "evidence_verification": VerificationStub()}, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_completed_source_requires_matching_provenance(tmp_path):
    _, _, sources = fixture_inputs()
    packet = {"solver": {"id": "solver"}, "dataset": {"revision": "pinned"}}
    metadata = dict(packet, protocol=SOURCE_PROTOCOL, sample_sha256="abc", revise_template=Solver.REVISE_PROMPT_V1)
    (tmp_path / "provenance.json").write_text(json.dumps(metadata))
    (tmp_path / "summary.json").write_text(json.dumps({"protocol": SOURCE_PROTOCOL}))
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in sources))
    records, hashes = load_controlled_source(tmp_path, packet, "abc")
    assert records == sources and len(hashes) == 3
    with pytest.raises(ValueError, match="sample_sha256"):
        load_controlled_source(tmp_path, packet, "other")
    with pytest.raises(ValueError, match="solver"):
        load_controlled_source(tmp_path, {**packet, "solver": {}}, "abc")


def test_refusal_does_not_count_as_retention_or_leave_denominator(tmp_path):
    class RefusalStub(VerificationStub):
        def revise(self, *args):
            return "Cannot determine"

    rows, examples, sources = fixture_inputs()
    summary = run_verification(rows, examples, sources,
                               {"original": BaselineStub(), "evidence_verification": RefusalStub()}, tmp_path / "refusal")
    arm = summary["conditions"]["evidence_verification"]["plausible_incorrect"]
    assert arm["completed"] == 1
    assert arm["adoption_rate"] == 0
    assert arm["retained_a1"] == 0
    assert arm["responses"] == {"non_answer": 1}
    assert summary["paired_comparisons"]["plausible_incorrect"]["adopted_to_retained"] == 0


def test_correct_correction_can_be_lost_under_verification(tmp_path):
    rows, examples, sources = fixture_inputs()
    rows[0]["solver_attempt_1"] = "Manchester"
    for source in sources:
        source["solver_attempt_1"] = "Manchester"
        if source["arm"] != "correct_correction":
            source["status"] = "skipped_requires_correct_a1"
        else:
            source.update(status="completed", critic_advocated_answer="Liverpool",
                          critic_feedback="The reviewer disagrees with your previous answer. The reviewer advocates this answer: Liverpool",
                          solver_attempt_2="Liverpool", target_metadata={"answer": "Liverpool", "source": "dataset_gold"})
    summary = run_verification(rows, examples, sources,
                               {"original": BaselineStub(), "evidence_verification": VerificationStub()}, tmp_path / "correction")
    assert summary["conditions"]["original"]["correct_correction"]["corrected"] == 1
    assert summary["conditions"]["evidence_verification"]["correct_correction"]["corrected"] == 0
    assert summary["paired_comparisons"]["correct_correction"]["accuracy_losses"] == 1
