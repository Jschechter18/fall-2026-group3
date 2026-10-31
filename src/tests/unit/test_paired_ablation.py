"""Paired ablation analysis (#174, #156): outcome rules, statistics, the three analyses and the CLI."""
import importlib.util
import json
from pathlib import Path

import pytest

from mas_sae.evaluation.paired_ablation import (analyses, arm_outcomes, exact_mcnemar, paired_difference_ci,
                                                paired_test)
from mas_sae.experiments.artifacts import sha256_file

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/experiments/compare_paired_ablation.py"
IDS = ["2hop__a", "3hop1__b", "3hop2__c", "4hop1__d"]


def episode(qid, a1=False, critic=True):
    return {"question_id": qid, "episode_id": f"{qid}__natural__candidates", "solver_attempt_1_correct": a1,
            "critic_feedback_correct": critic}


def label(qid, value):
    return {"question_id": qid, "episode_id": f"{qid}__natural__candidates", "behavior_label": value}


def exclusion(qid, stage="critic_recommendation"):
    return {"question_id": qid, "stage": stage}


def test_interval_matches_the_published_example():
    # Newcombe hybrid score interval for the 20/15/6/5 table (CAMIS, paired proportions): 0.002602 to 0.369943
    low, high = paired_difference_ci(20, 15, 6, 5, z=1.959964)
    assert low == pytest.approx(0.002602, abs=5e-7) and high == pytest.approx(0.369943, abs=5e-7)


def test_interval_is_antisymmetric_and_contains_the_estimate():
    low, high = paired_difference_ci(13, 22, 3, 62)
    assert low < 0.19 < high
    assert paired_difference_ci(13, 3, 22, 62) == pytest.approx((-high, -low))
    assert paired_difference_ci(0, 0, 0, 0) is None


def test_exact_mcnemar():
    assert exact_mcnemar(22, 3) == pytest.approx(2 * (1 + 25 + 300 + 2300) / 2 ** 25)
    assert exact_mcnemar(3, 22) == exact_mcnemar(22, 3)
    assert exact_mcnemar(5, 0) == pytest.approx(2 / 32) and exact_mcnemar(0, 0) == 1.0 == exact_mcnemar(4, 4)


def test_outcome_rules():
    outcomes = arm_outcomes(IDS, [episode(IDS[0]), episode(IDS[3])], [exclusion(IDS[1])],
                            [label(IDS[0], "adoption"), label(IDS[3], None)])
    assert outcomes == {IDS[0]: "adoption", IDS[1]: "out", IDS[2]: "not_collected", IDS[3]: "label_failed"}


@pytest.mark.parametrize("interactions, exclusions, labels, message", [
    ([episode(IDS[0]), episode(IDS[0])], [], [label(IDS[0], "adoption")], "repeated question_id"),
    ([episode("2hop__z")], [], [label("2hop__z", "adoption")], "were not assigned"),
    ([episode(IDS[0])], [exclusion(IDS[0])], [label(IDS[0], "adoption")], "both collected and excluded"),
    ([], [exclusion(IDS[0], stage="cuda_oom")], [], "unknown collection exclusion stages"),
    ([], [], [label(IDS[0], "adoption")], "have no collected episode"),
    ([episode(IDS[0])], [], [], "incomplete label run"),
    ([episode(IDS[0])], [], [{**label(IDS[0], "adoption"), "episode_id": "other"}], "the label is for episode"),
    ([episode(IDS[0])], [], [label(IDS[0], "error")], "unknown label"),
])
def test_outcome_refusals(interactions, exclusions, labels, message):
    with pytest.raises(ValueError, match=message):
        arm_outcomes(IDS, interactions, exclusions, labels)


def test_technical_failures_leave_the_primary_analysis_and_count_against_itt():
    baseline = {IDS[0]: "agreement", IDS[1]: "agreement", IDS[2]: "retention", IDS[3]: "agreement"}
    treated = {IDS[0]: "adoption", IDS[1]: "not_collected", IDS[2]: "out", IDS[3]: "label_failed"}
    result = analyses(baseline, treated, IDS, treated_ids=[IDS[0], IDS[1]])
    assert result["primary"]["n"] == 2 and result["primary"]["discordant"] == {"gained": 1, "lost": 1}
    assert result["intention_to_treat"]["n"] == 4 and result["intention_to_treat"]["treated"]["binary"] == 1
    assert result["treated_only"]["n"] == 1 and result["treated_only"]["discordant"] == {"gained": 1, "lost": 0}
    assert result["technical_failures"] == {"baseline": {}, "treated": {"not_collected": 1, "label_failed": 1}}


def test_analysis_refusals():
    arm = dict.fromkeys(IDS, "agreement")
    with pytest.raises(ValueError, match="exactly the assigned"):
        analyses(arm, {**arm, "2hop__z": "adoption"}, IDS)
    with pytest.raises(ValueError, match="must be assigned"):
        analyses(arm, arm, IDS, treated_ids=["2hop__z"])
    with pytest.raises(ValueError, match="repeats"):
        analyses(arm, arm, IDS, treated_ids=[IDS[0], IDS[0]])


def test_paired_test_reports_the_change_in_points():
    baseline = dict.fromkeys(IDS, False)
    result = paired_test(baseline, {**baseline, IDS[0]: True}, IDS)
    assert result["change_pp"] == 25.0 and result["table"] == {"both": 0, "treated_only": 1, "baseline_only": 0,
                                                               "neither": 3}


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("compare_paired_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LABELING = {"prompt_version": "claude_binary_v3_candidates", "model_id": "opus", "client": "AdaptiveThinkingClaudeClient",
            "effort": "medium", "max_tokens": 8000, "temperature": None, "a2_presentation": "candidates",
            "max_attempts": 3}


def write_arm(root: Path, labels: dict[str, str | None], excluded=(), orders=None, settings=None):
    """A collection folder and its label run: ``labels`` for collected questions, ``excluded`` by protocol."""
    collection, run = root / "collection", root / "labels"
    collection.mkdir(parents=True)
    run.mkdir()
    jsonl = lambda path, rows: path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    jsonl(collection / "interactions.jsonl", [episode(q) for q in labels])
    jsonl(collection / "exclusions.jsonl", [exclusion(q) for q in excluded])
    (collection / "sampled_questions.json").write_text(json.dumps(
        [{"position": i, "question_id": q, "candidate_order": (orders or {}).get(q, "a1_first")}
         for i, q in enumerate(IDS)]))
    jsonl(run / "raw_results.jsonl", [label(q, value) for q, value in labels.items()])
    (run / "summary.json").write_text(json.dumps({"episodes_sha256": sha256_file(collection / "interactions.jsonl"),
                                                  **LABELING, **(settings or {})}))
    return [str(collection), str(run)]


def test_cli_end_to_end(cli, tmp_path, capsys):
    base = write_arm(tmp_path / "base", dict.fromkeys(IDS, "agreement"))
    treated = write_arm(tmp_path / "treated", {IDS[0]: "adoption", IDS[1]: "retention", IDS[2]: "agreement"},
                        excluded=[IDS[3]])
    (tmp_path / "treated_ids.json").write_text(json.dumps(IDS[:2]))
    cli.main(["--baseline", *base, "--treated", *treated, "--treated-ids", str(tmp_path / "treated_ids.json"),
              "--output", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["primary"]["discordant"] == {"gained": 2, "lost": 0} and summary["primary"]["n"] == 4
    assert summary["treated_only"]["n"] == 2
    assert summary["descriptives"]["treated"]["correctness_cells"]["no episode"] == {"out": 1}
    assert "exact McNemar" in capsys.readouterr().out


@pytest.mark.parametrize("change, message", [
    ({"orders": {IDS[2]: "critic_first"}}, "different questions, positions or candidate orders"),
    ({"settings": {"prompt_version": "claude_binary_v2"}}, r"labeled differently \(prompt_version\)"),
    ({"settings": {"effort": "high"}}, r"labeled differently \(effort\)"),
    ({"settings": {"max_tokens": 4000}}, r"labeled differently \(max_tokens\)"),
    ({"settings": {"temperature": 1.0, "client": "ClaudeClient"}}, r"labeled differently \(client, temperature\)"),
])
def test_cli_refuses_arms_that_do_not_pair(cli, tmp_path, change, message):
    base = write_arm(tmp_path / "base", dict.fromkeys(IDS, "agreement"))
    treated = write_arm(tmp_path / "treated", dict.fromkeys(IDS, "adoption"), **change)
    with pytest.raises(ValueError, match=message):
        cli.main(["--baseline", *base, "--treated", *treated, "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_cli_pairs_arms_that_differ_only_in_retries_of_technical_errors(cli, tmp_path):
    base = write_arm(tmp_path / "base", dict.fromkeys(IDS, "agreement"))
    treated = write_arm(tmp_path / "treated", dict.fromkeys(IDS, "adoption"), settings={"max_attempts": 5})
    cli.main(["--baseline", *base, "--treated", *treated, "--output", str(tmp_path / "out")])
    assert json.loads((tmp_path / "out/summary.json").read_text())["primary"]["n"] == 4


def test_cli_refuses_labels_made_from_other_episodes(cli, tmp_path):
    base = write_arm(tmp_path / "base", dict.fromkeys(IDS, "agreement"))
    treated = write_arm(tmp_path / "treated", dict.fromkeys(IDS, "adoption"))
    with open(Path(treated[0]) / "interactions.jsonl", "a") as fh:
        fh.write("\n")
    with pytest.raises(ValueError, match="were not made from"):
        cli.main(["--baseline", *base, "--treated", *treated, "--output", str(tmp_path / "out")])


def test_cli_refuses_inputs_that_change_while_read(cli, tmp_path, monkeypatch):
    base = write_arm(tmp_path / "base", dict.fromkeys(IDS, "agreement"))
    treated = write_arm(tmp_path / "treated", dict.fromkeys(IDS, "adoption"))
    real = cli.analyses

    def analyses_then_edit(*args, **kwargs):
        Path(base[1], "summary.json").write_text("{}")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "analyses", analyses_then_edit)
    with pytest.raises(RuntimeError, match="changed while reading"):
        cli.main(["--baseline", *base, "--treated", *treated, "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
