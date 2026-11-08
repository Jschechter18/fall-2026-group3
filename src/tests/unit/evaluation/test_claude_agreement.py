"""Tests for Claude-vs-reference agreement metrics."""

import pytest

from mas_sae.evaluation.claude_agreement import (
    binary_side_agreement,
    cohen_kappa,
    confusion,
    per_class,
    weighted_accuracy,
)

PAIRS = ([("adoption", "adoption")] * 8 + [("adoption", "retention")] * 2
         + [("retention", "retention")] * 9 + [("retention", None)]
         + [("agreement", "agreement")] * 10 + [("out", "out")] * 5 + [("out", "agreement")] * 5)


def test_confusion_counts_errors_separately():
    m = confusion(PAIRS)
    assert m["adoption"] == {"adoption": 8, "retention": 2, "agreement": 0, "out": 0, "error": 0}
    assert m["retention"]["error"] == 1
    with pytest.raises(ValueError):
        confusion([("unknown", "out")])


def test_per_class_precision_recall_f1():
    pc = per_class(confusion(PAIRS))
    assert pc["adoption"]["recall"] == 0.8 and pc["adoption"]["precision"] == 1.0
    assert pc["retention"]["recall"] == 0.9 and pc["retention"]["precision"] == pytest.approx(9 / 11)
    assert pc["agreement"]["precision"] == pytest.approx(10 / 15)
    assert pc["out"]["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_kappa_perfect_and_chance():
    perfect = confusion([(l, l) for l in ("adoption", "retention", "agreement", "out")] * 5)
    assert cohen_kappa(perfect) == 1.0
    assert 0 < cohen_kappa(confusion(PAIRS)) < 1


def test_weighted_accuracy_uses_population_mix():
    m = confusion(PAIRS)
    acc = weighted_accuracy(m, {"adoption": 100, "retention": 100, "agreement": 600, "out": 200})
    assert acc == pytest.approx(0.1 * 0.8 + 0.1 * 0.9 + 0.6 * 1.0 + 0.2 * 0.5)


def test_binary_side_agreement_ignores_other_labels():
    assert binary_side_agreement(PAIRS) == {"n": 19, "same": 17, "rate": pytest.approx(17 / 19)}
