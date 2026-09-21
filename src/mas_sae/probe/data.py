from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from mas_sae.data.activation_store import ActivationStore
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


def make_sample_sae_features(
    n_samples: int,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate sample SAE features by pushing random raw activations through
    a real (untrained) SparseAutoencoder encoder. Stand-in for real
    Solver-Critic activations until the real activation pipeline exists."""
    torch.manual_seed(seed)
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    sae.eval()
    raw_activations = torch.randn(n_samples, input_dim)
    with torch.no_grad():
        sparse_features, _ = sae(raw_activations)
    return sparse_features.numpy()


def make_sample_labels(
    sparse_features: np.ndarray,
    signal_dims: list[int],
    seed: int = 42,
) -> np.ndarray:
    """Bake a known signal into a few latent dims so probe/SHAP/LIME results
    can be checked against ground truth. Stand-in for solver_accepted_feedback."""
    rng = np.random.default_rng(seed)
    default_weights = [3.0, -2.0, 1.5, 1.0, -1.0]
    weights = default_weights[: len(signal_dims)]
    logit = sum(w * sparse_features[:, d] for w, d in zip(weights, signal_dims))
    logit = logit - logit.mean()
    prob = 1 / (1 + np.exp(-logit))
    return (rng.random(sparse_features.shape[0]) < prob).astype(int)


def load_acceptance_probe_data(
    location: str | Path,
    interactions: list[dict[str, Any]],
    split: str = "pilot_attempt2",
) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled Attempt 2 activations for the acceptance probe."""
    activations = ActivationStore(location).load_activations(split)

    indices = []
    labels = []

    for row in interactions:
        label = row["solver_accepted_feedback"]

        if label is None:
            continue

        indices.append(row["attempt2_activation_index"])
        labels.append(int(label))

    X = activations[indices].numpy()
    y = np.asarray(labels, dtype=np.int64)

    return X, y
