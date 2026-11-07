"""Tests for the issue #166 candidate-order breakdown of the PR #153 labels.

The adoption-condition tests from the same issue are in src/tests/unit/scripts/experiments/adoption/test_adoption_conditions.py."""

import importlib.util
import json

import pytest

from repo_paths import REPO_ROOT

SCRIPTS = REPO_ROOT / "scripts/archive/lexical_labels_v01"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
