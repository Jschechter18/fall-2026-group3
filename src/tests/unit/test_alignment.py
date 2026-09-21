import pytest
import torch

from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.alignment import validate_pilot_alignment


def make_pilot(tmp_path):
    store = ActivationStore(tmp_path / "layer_08")
    store.save_activations("pilot_attempt1", torch.ones(2, 4))
    store.save_activations("pilot_attempt2", torch.ones(6, 4))

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
                "attempt2_activation_index": question_index * 3 + offset,
            })

    return rows


def validate(tmp_path, rows):
    return validate_pilot_alignment(
        tmp_path,
        rows,
        [8],
        2,
        hidden_dim=4,
    )


def test_alignment_passes(tmp_path):
    assert validate(
        tmp_path,
        make_pilot(tmp_path),
    )["status"] == "pass"


def test_bad_attempt2_index_fails(tmp_path):
    rows = make_pilot(tmp_path)
    rows[4]["attempt2_activation_index"] = 99

    with pytest.raises(ValueError):
        validate(tmp_path, rows)


def test_shared_attempt1_index_fails(tmp_path):
    rows = make_pilot(tmp_path)

    for row in rows[3:]:
        row["attempt1_activation_index"] = 0

    with pytest.raises(ValueError):
        validate(tmp_path, rows)
