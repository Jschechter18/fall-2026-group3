import numpy as np
import torch

from mas_sae.data.activation_store import ActivationStore
from mas_sae.probe.data import load_acceptance_probe_data


def test_load_acceptance_probe_data_filters_none(tmp_path):
    store = ActivationStore(tmp_path)
    activations = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(3, 4)
    store.save_activations(
        "pilot_attempt2",
        activations,
    )

    interactions = [
        {
            "attempt2_activation_index": 2,
            "solver_accepted_feedback": True,
        },
        {
            "attempt2_activation_index": 1,
            "solver_accepted_feedback": None,
        },
        {
            "attempt2_activation_index": 0,
            "solver_accepted_feedback": False,
        },
    ]

    X, y = load_acceptance_probe_data(
        tmp_path,
        interactions,
    )

    np.testing.assert_array_equal(
        X,
        activations[[2, 0]].numpy(),
    )
    assert y.tolist() == [1, 0]
