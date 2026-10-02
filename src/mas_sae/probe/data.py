from __future__ import annotations

import numpy as np
import torch 

import json
from pathlib import Path

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

#added after Israel's dependencies completed 

def load_real_layer_split(
    run_name: str,
    split: str,
    layer: str,
    data_root: Path = Path("data/activations"),
    results_root: Path = Path("results/collection"),
) -> tuple[np.ndarray, np.ndarray]:
    """Load real Solver-Critic Attempt-2 activations and solver_accepted_feedback
    labels for one candidate layer and split, aligned via each interaction
    record's attempt2_activation_index.
 
    Episodes with critic_noncommittal=True are excluded, since the Critic
    never gave a clear stance to accept or reject -- see the proposal's
    "defining feedback acceptance consistently" open issue.
 
    Parameters
    ----------
    run_name : str
        The collection run's name, e.g. "scale_smoke_1q".
    split : str
        "train" or "validation" (matches the real folder names produced by
        the collection pipeline).
    layer : str
        Layer identifier as it appears in the folder name, e.g. "08", "17".
    data_root : Path
        Root directory containing <run_name>/layer_<N>/*.pt activation files.
    results_root : Path
        Root directory containing <run_name>/<split>/interactions.jsonl.
 
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X: Attempt-2 activations for the retained episodes, shape
           (n_retained, activation_dim).
        y: Binary solver_accepted_feedback labels, shape (n_retained,).
    """
    layer_dir = data_root / run_name / f"layer_{layer}"
    activations = torch.load(layer_dir / f"{split}_attempt2.pt", weights_only=False).numpy()
 
    interactions_path = results_root / run_name / split / "interactions.jsonl"
 
    X_rows, y_rows = [], []
    with open(interactions_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("critic_noncommittal"):
                continue
            idx = record["attempt2_activation_index"]
            X_rows.append(activations[idx])
            y_rows.append(int(record["solver_accepted_feedback"]))
 
    return np.array(X_rows), np.array(y_rows)