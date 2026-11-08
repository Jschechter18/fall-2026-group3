import pytest
import torch

from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.alignment import (
    validate_activation_alignment,
)


CONDITIONS = {
    "natural",
    "controlled_correct",
    "controlled_incorrect",
}


def make_alignment_data(tmp_path):
    store = ActivationStore(tmp_path / "layer_08")

    store.save_activations(
        "attempt1",
        torch.ones(2, 4),
    )
    store.save_activations(
        "attempt2",
        torch.ones(6, 4),
    )

    conditions = [
        "natural",
        "controlled_correct",
        "controlled_incorrect",
    ]

    rows = []

    for question_index in range(2):
        for offset, condition in enumerate(conditions):
            rows.append({
                "question_id": f"q{question_index}",
                "solver_attempt_1": f"answer-{question_index}",
                "critic_condition": condition,
                "solver_accepted_feedback": True,
                "attempt1_activation_index": question_index,
                "attempt2_activation_index": (
                    question_index * len(conditions) + offset
                ),
            })

    return rows


def validate(tmp_path, rows):
    return validate_activation_alignment(
        store_root=tmp_path,
        interactions=rows,
        layers=[8],
        attempt1_split="attempt1",
        attempt2_split="attempt2",
        expected_conditions=CONDITIONS,
        hidden_dim=4,
        expected_num_questions=2,
    )


def test_alignment_passes(tmp_path):
    report = validate(
        tmp_path,
        make_alignment_data(tmp_path),
    )

    assert report["status"] == "pass"
    assert report["questions"] == 2
    assert report["episodes"] == 6


def test_bad_attempt2_index_fails(tmp_path):
    rows = make_alignment_data(tmp_path)
    rows[4]["attempt2_activation_index"] = 99

    with pytest.raises(ValueError):
        validate(tmp_path, rows)


def test_shared_attempt1_index_fails(tmp_path):
    rows = make_alignment_data(tmp_path)

    for row in rows[3:]:
        row["attempt1_activation_index"] = 0

    with pytest.raises(ValueError):
        validate(tmp_path, rows)


def test_interaction_order_does_not_change_alignment(tmp_path):
    rows = make_alignment_data(tmp_path)
    rows.reverse()

    report = validate(tmp_path, rows)

    assert report["status"] == "pass"
    assert report["episodes"] == 6
