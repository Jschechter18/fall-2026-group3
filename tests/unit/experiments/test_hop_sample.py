"""Tests for the hop-proportional Natural sample (no model or data downloads)."""

import pytest

pytest.importorskip("datasets")  # mas_sae.data.musique (hop_group) imports it

from mas_sae.data.musique import hop_group
from mas_sae.experiments.hop_sample import (
    proportional_allocation,
    proportional_hop_sample,
    stratified_top_up,
    top_up_allocation,
)


def _rows(n2, n3, n4):
    """Rows with only a question_id: selection must not need A2 or labels."""
    return ([{"question_id": f"2hop__{i}"} for i in range(n2)]
            + [{"question_id": f"3hop1__{i}"} for i in range(n3)]
            + [{"question_id": f"4hop1__{i}"} for i in range(n4)])


def test_allocation_matches_frozen_train_mix():
    # Frozen train partition hop counts (partitions.csv): 11491 / 3506 / 939.
    assert proportional_allocation({"2hop": 11491, "3hop": 3506, "4hop": 939}, 100) == {
        "2hop": 72, "3hop": 22, "4hop": 6,
    }


@pytest.mark.parametrize("size", [0, 11])
def test_allocation_rejects_impossible_sizes(size):
    with pytest.raises(ValueError):
        proportional_allocation({"2hop": 6, "3hop": 4}, size)


def test_sample_is_proportional_and_distinct():
    selected, allocation = proportional_hop_sample(_rows(700, 200, 100), 50, seed=1)
    assert {g: a["selected"] for g, a in allocation.items()} == {"2hop": 35, "3hop": 10, "4hop": 5}
    assert len({r["question_id"] for r in selected}) == 50


def test_same_seed_same_sample_and_order_independent():
    rows = _rows(300, 100, 50)
    first, _ = proportional_hop_sample(rows, 40, seed=20261029)
    again, _ = proportional_hop_sample(list(reversed(rows)), 40, seed=20261029)
    other, _ = proportional_hop_sample(rows, 40, seed=7)
    assert first == again
    assert first != other


def test_excluded_questions_never_drawn():
    rows = _rows(30, 10, 10)
    exclude = {r["question_id"] for r in rows[:20]}
    selected, allocation = proportional_hop_sample(rows, 30, seed=3, exclude_ids=exclude)
    assert not exclude & {r["question_id"] for r in selected}
    assert allocation["2hop"]["population"] == 10


def test_rows_returned_unmodified():
    rows = [dict(r, solver_attempt_2="x", label={"a": 1}) for r in _rows(5, 3, 2)]
    snapshot = [dict(r) for r in rows]
    selected, _ = proportional_hop_sample(rows, 10, seed=0)
    assert sorted(selected, key=lambda r: r["question_id"]) == sorted(snapshot, key=lambda r: r["question_id"])


def _hop(row):
    return hop_group(row["question_id"])


def test_top_up_from_empty_matches_proportional():
    pops = {"2hop": 11491, "3hop": 3506, "4hop": 939}
    assert top_up_allocation(pops, {}, 100) == proportional_allocation(pops, 100)


def test_top_up_skips_over_covered_strata():
    # quota at 10 is a=5, b=5; cohort already holds 6 of a
    assert top_up_allocation({"a": 50, "b": 50}, {"a": 6}, 10) == {"a": 0, "b": 4}


def test_top_up_respects_capacity():
    assert top_up_allocation({"a": 2, "b": 98}, {}, 50) == {"a": 1, "b": 49}
    with pytest.raises(ValueError):
        top_up_allocation({"a": 2}, {"a": 1}, 3)


def test_stratified_top_up_never_redraws_cohort_and_reaches_proportion():
    rows = _rows(600, 300, 100)
    cohort = {r["question_id"] for r in rows[:50]}  # all 2hop: over-covers 2hop at 100
    new, strata = stratified_top_up(rows, _hop, cohort, additional=50, seed=20261029)

    new_ids = {r["question_id"] for r in new}
    assert len(new_ids) == 50 and not new_ids & cohort
    assert {s: v["total"] for s, v in strata.items()} == {"2hop": 60, "3hop": 30, "4hop": 10}
    assert strata["2hop"] == {"population": 600, "cohort": 50, "added": 10, "total": 60, "quota": 60.0}


def test_stratified_top_up_excludes_and_validates_cohort():
    rows = _rows(100, 50, 50)
    exclude = {r["question_id"] for r in rows[100:110]}
    new, strata = stratified_top_up(rows, _hop, set(), 40, seed=1, exclude_ids=exclude)
    assert not exclude & {r["question_id"] for r in new}
    assert strata["3hop"]["population"] == 40
    with pytest.raises(ValueError, match="not in the pool"):
        stratified_top_up(rows, _hop, {"2hop__missing"}, 5, seed=1)


def test_stratified_top_up_is_deterministic():
    rows = _rows(200, 80, 20)
    cohort = {r["question_id"] for r in rows[::7]}
    a, _ = stratified_top_up(rows, _hop, cohort, 60, seed=5)
    b, _ = stratified_top_up(list(reversed(rows)), _hop, cohort, 60, seed=5)
    assert a == b


def test_namespaced_ids_fail_loudly_instead_of_collapsing():
    rows = [{"question_id": f"musique:2hop__{i}"} for i in range(10)]
    with pytest.raises(ValueError, match="no parseable hop group"):
        proportional_hop_sample(rows, 5, seed=1)
