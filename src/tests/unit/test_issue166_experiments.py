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


class TestOrderBreakdown:
    @pytest.fixture
    def order(self):
        return _load("summarize_order_labels")

    @staticmethod
    def _source(order, a1=False, critic=True):
        return {"a2_presentation": "candidates", "candidate_order": order,
                "solver_attempt_1_correct": a1, "critic_feedback_correct": critic}

    def test_correctness_case(self, order):
        assert order.correctness_case(self._source("a1_first", False, True)) == "useful_correction"
        assert order.correctness_case(self._source("a1_first", True, False)) == "harmful_correction"
        assert order.correctness_case(self._source("a1_first", True, None)) == "unscored"

    def test_split_and_reconcile(self, order):
        labels = [{"question_id": "q1", "partition": "train", "label": "adoption", "collected": True},
                  {"question_id": "q2", "partition": "train", "label": "retention", "collected": True},
                  {"question_id": "q3", "partition": "train", "label": "out", "collected": False},
                  {"question_id": "q4", "partition": None, "label": "out", "collected": True}]
        sources = {"q1": self._source("critic_first"), "q2": self._source("a1_first", True, False)}
        rows, not_collected = order.join_orders(labels, sources)
        assert [(r["order"], r["case"]) for r in rows] == [("critic_first", "useful_correction"),
                                                           ("a1_first", "harmful_correction")]
        assert not_collected == {"train": {"out": 1}}
        zero = dict.fromkeys(order.CLASSES, 0)
        approved = {"labels_by_partition": {p: dict(zero) for p in order.PARTITIONS}}
        approved["labels_by_partition"]["train"].update(adoption=1, retention=1, out=1)
        order.reconcile(rows, not_collected, approved)
        b = order.breakdown(rows)["by_order"]["train"]
        assert b["critic_first"]["adoption"]["count"] == 1 and b["a1_first"]["retention"]["count"] == 1
        assert b["both"]["n"] == 2
        approved["labels_by_partition"]["train"]["out"] = 2
        with pytest.raises(SystemExit, match="differ from counts.json"):
            order.reconcile(rows, not_collected, approved)

    def test_refuses_missing_interaction_and_unknown_order(self, order):
        collected = [{"question_id": "q1", "partition": "train", "label": "adoption", "collected": True}]
        with pytest.raises(SystemExit, match="no interaction"):
            order.join_orders(collected, {})
        with pytest.raises(SystemExit, match="known order"):
            order.join_orders(collected, {"q1": self._source("random")})


def test_order_figure_and_tables(tmp_path):
    pytest.importorskip("matplotlib")
    order, plot = _load("summarize_order_labels"), _load("plot_order_labels")
    rows = [{"question_id": f"q{i}", "partition": p, "label": label, "order": o, "case": case}
            for i, (p, label, o, case) in enumerate(
                [(p, label, o, case) for p in order.PARTITIONS for o in order.ORDERS
                 for label, case in (("adoption", "useful_correction"), ("retention", "harmful_correction"),
                                     ("adoption", "harmful_correction"), ("agreement", "both_right"))])]
    summary = {**order.breakdown(rows), "labels_sha256": "f" * 64}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    plot.main(["--summary", str(summary_path), "--figures", str(tmp_path / "fig"), "--tables", str(tmp_path / "tab")])
    by_order, by_case = plot.table_rows(summary)
    both = next(r for r in by_order if r["partition"] == "all" and r["order"] == "both")
    assert (both["adoption"], both["retention"], both["agreement"]) == (16, 8, 8)
    assert both["adoption_share"] == pytest.approx(2 / 3) and both["ci_low"] < 2 / 3 < both["ci_high"]
    assert {r["case"] for r in by_case} == {"useful_correction", "harmful_correction"}
    tex = (tmp_path / "tab/candidate_order_labels.tex").read_text()
    assert "Both orders & 16 & 8 & 8 & 0 & 66.7\\%" in tex
    assert (tmp_path / "fig/candidate_order_adoption.pdf").stat().st_size > 0
