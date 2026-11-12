"""The hand-edit rules and the edits.py record (mas_sae.llm_edit.edit_rules)."""

import numpy as np
import pytest

from mas_sae.llm_edit.edit_rules import (
    COSINE_CAP, Edit, apply_edits, cosine_similarity, edit_problems, read_edits_py, write_edits_py,
)

SELECTED = {"adoption": [1], "retention": [2]}


def decoder(seed=0, latents=5, width=16):
    rows = np.random.default_rng(seed).normal(size=(latents, width)).astype(np.float32)
    return rows * np.float32(3.0) / np.linalg.norm(rows, axis=1, keepdims=True)     # length 3, not 1


def rotated(row, cosine, seed=1):
    """`row` rotated to the given cosine with itself, same length."""
    unit = row / np.linalg.norm(row)
    other = np.random.default_rng(seed).normal(size=row.shape)
    perpendicular = other - (other @ unit) * unit
    perpendicular /= np.linalg.norm(perpendicular)
    return (np.linalg.norm(row) * (cosine * unit + np.sqrt(1 - cosine ** 2) * perpendicular)).astype(np.float32)


def test_a_small_rotation_of_a_selected_row_is_valid():
    w = decoder()
    assert edit_problems(w, SELECTED, 1, rotated(w[1], 0.97)) == []


def test_rotation_past_the_cap_is_rejected():
    w = decoder()
    problems = edit_problems(w, SELECTED, 1, rotated(w[1], COSINE_CAP - 0.02))
    assert any("below the cap" in p for p in problems)


def test_changing_the_length_is_rejected():
    w = decoder()
    assert any("length" in p for p in edit_problems(w, SELECTED, 2, w[2] * 1.01))


def test_a_row_outside_the_selected_features_is_rejected():
    w = decoder()
    assert edit_problems(w, SELECTED, 3, w[3]) == ["feature 3 is not one of the 20 selected features"]


def test_the_cap_is_against_the_original_row_not_the_previous_edit():
    # Two 0.97 steps in the same direction end up past 0.95 from the original: the second must fail.
    w = decoder()
    first = rotated(w[1], 0.97, seed=4)
    second = rotated(first, 0.97, seed=4)
    assert cosine_similarity(second, w[1]) < COSINE_CAP
    assert edit_problems(w, SELECTED, 1, first) == []
    assert edit_problems(w, SELECTED, 1, second) != []


def test_apply_edits_changes_only_edited_rows_and_a_later_edit_wins():
    w = decoder()
    a, b = rotated(w[1], 0.98, seed=2), rotated(w[1], 0.96, seed=3)
    edited = apply_edits(w, [Edit(1, 1, a, ""), Edit(2, 1, b, "")])
    assert np.array_equal(edited[1], b)
    assert np.array_equal(np.delete(edited, 1, axis=0), np.delete(w, 1, axis=0))
    assert np.array_equal(w[1], decoder()[1])            # the original is not modified


def test_edits_py_round_trips_and_detects_a_changed_row(tmp_path):
    w = decoder()
    edits = [Edit(1, 1, rotated(w[1], 0.97), "Point it further from the 'fictional' cluster."),
             Edit(2, 2, rotated(w[2], 0.99), "")]
    write_edits_py(edits, tmp_path)
    assert "Point it further" in (tmp_path / "edits.py").read_text()
    replayed = read_edits_py(tmp_path)
    assert [(e.number, e.feature) for e in replayed] == [(1, 1), (2, 2)]
    assert np.array_equal(apply_edits(w, replayed), apply_edits(w, edits))

    np.save(tmp_path / "rows" / "edit_002.npy", w[2])    # swap a row behind edits.py's back
    with pytest.raises(ValueError, match="sha256"):
        read_edits_py(tmp_path)
