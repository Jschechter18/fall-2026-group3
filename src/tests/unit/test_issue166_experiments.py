"""Tests for the issue #166 experiment helpers: condition export, summaries and intervals."""

import importlib.util
import json
from pathlib import Path

import pytest

from mas_sae.evaluation.claude_agreement import label_distribution, wilson
from mas_sae.experiments.records import write_jsonl

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wilson_and_distribution():
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and wilson(0, 0) is None
    d = label_distribution(["adoption", "adoption", "retention", "agreement", None])
    assert d["n"] == 5 and d["adoption"]["count"] == 2 and d["error"]["count"] == 1
    assert d["retention_share_of_binary"]["share"] == pytest.approx(1 / 3)


class TestConditionExport:
    @pytest.fixture(scope="class")
    def export(self):
        pytest.importorskip("torch")  # the reused natural_protocol_comparison imports the agents
        return _load("export_adoption_conditions")

    def test_shown_order(self, export):
        a = {"q1": "critic_first"}
        assert export.shown_order("Original revision baseline", "q1", a) is None
        assert export.shown_order("Labeled candidates: A1 first", "q1", a) == "a1_first"
        assert export.shown_order("Anonymous candidates: critic first", "q1", a) == "critic_first"
        assert export.shown_order("Labeled candidates: randomized order", "q1", a) == "critic_first"

    def test_validator_episode_hides_lexical_outcome(self, export):
        row = {"question_id": "q1", "question": "Q?", "solver_attempt_1": "a", "critic_advocated_answer": "b",
               "solver_attempt_2": "b", "adopted_target": True, "retained_a1": False, "a2_correct": True}
        ep = export.validator_episode(row, "Labeled candidates: A1 first", "a1_first")
        assert ep["a2_presentation"] == "candidates" and ep["candidate_order"] == "a1_first"
        assert ep["episode_id"] == "q1__labeled_candidates_a1_first"
        assert not {"adopted_target", "retained_a1", "a2_correct"} & set(ep)


def test_condition_summary_requires_the_exported_file(tmp_path):
    summ = _load("summarize_condition_labels")
    condition = {"condition": "X", "slug": "x", "a2_presentation": "feedback", "sampled_episodes_sha256": "abc",
                 "lexical": [{"episode_id": "e1", "adopted_target": True, "retained_a1": False},
                             {"episode_id": "e2", "adopted_target": False, "retained_a1": True}]}
    run = tmp_path / "x"
    write_jsonl(run / "raw_results.jsonl", [{"episode_id": "e1", "behavior_label": "adoption"},
                                            {"episode_id": "e2", "behavior_label": "retention"}])
    (run / "summary.json").write_text(json.dumps({"episodes_sha256": "abc", "total_input_tokens": 1, "total_output_tokens": 1}))
    s = summ.condition_summary(condition, run)
    assert s["claude"]["retention_share_of_binary"]["share"] == 0.5
    assert s["lexical"] == {"adopted": 1, "retained_a1": 1, "overlap": 0, "other": 0}
    (run / "summary.json").write_text(json.dumps({"episodes_sha256": "zzz", "total_input_tokens": 1, "total_output_tokens": 1}))
    with pytest.raises(SystemExit, match="did not label"):
        summ.condition_summary(condition, run)
