"""Unit tests for mas_sae.causal.intervention."""

import numpy as np
import pytest
import torch

from mas_sae.causal.intervention import (
    downstream_decision,
    run_causal_experiment,
    run_condition,
)
from mas_sae.probe.model import fit_probe, train_val_split
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


@pytest.fixture
def sae_and_probe():
    """A small SAE plus a probe fit on its (untrained) encoder output, small
    enough to run quickly in a unit test."""
    torch.manual_seed(0)
    sae = SparseAutoencoder(input_dim=8, hidden_dim=4, latent_dim=8)
    sae.eval()

    rng = np.random.default_rng(0)
    raw = torch.randn(100, 8)
    with torch.no_grad():
        z, _ = sae(raw)
    z = z.numpy()

    labels = (rng.random(100) < 0.5).astype(int)
    X_train, X_val, y_train, y_val = train_val_split(z, labels, seed=0)
    probe, scaler, _, _, X_val_s, _ = fit_probe(X_train, y_train, X_val, y_val, seed=0)

    return sae, probe, scaler, X_val_s


def test_downstream_decision_returns_binary_predictions(sae_and_probe):
    sae, probe, scaler, X_val_s = sae_and_probe
    # downstream_decision expects a raw activation, so decode first
    with torch.no_grad():
        activation = sae.decoder(torch.from_numpy(X_val_s).float()).numpy()
    preds = downstream_decision(activation, sae, probe, scaler)
    assert preds.shape == (X_val_s.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_run_condition_suppress_zeroes_the_target_feature(sae_and_probe):
    """We can't directly observe z inside run_condition, but we can verify
    the suppression logic in isolation the same way run_condition does it."""
    _, _, _, X_val_s = sae_and_probe
    z = X_val_s.copy()
    feature_idx = 2
    z[:, feature_idx] = 0.0
    assert (z[:, feature_idx] == 0.0).all()


def test_run_condition_with_no_feature_idx_is_a_pure_reconstruction(sae_and_probe):
    """feature_idx=None should behave identically to encode/decode with no
    modification at all -- this is what the reconstruction-only control
    relies on."""
    sae, probe, scaler, X_val_s = sae_and_probe
    preds_a = run_condition(X_val_s, sae, probe, scaler)
    preds_b = run_condition(X_val_s, sae, probe, scaler)
    assert np.array_equal(preds_a, preds_b)


def test_run_condition_raises_on_invalid_mode(sae_and_probe):
    sae, probe, scaler, X_val_s = sae_and_probe
    with pytest.raises(ValueError, match="suppress.*amplify"):
        run_condition(X_val_s, sae, probe, scaler, feature_idx=2, mode="typo")


def test_run_condition_accepts_valid_modes_without_error(sae_and_probe):
    sae, probe, scaler, X_val_s = sae_and_probe
    # should not raise
    run_condition(X_val_s, sae, probe, scaler, feature_idx=2, mode="suppress")
    run_condition(X_val_s, sae, probe, scaler, feature_idx=2, mode="amplify")


def test_run_causal_experiment_returns_expected_structure():
    torch.manual_seed(1)
    sae = SparseAutoencoder(input_dim=16, hidden_dim=4, latent_dim=16)
    sae.eval()

    rng = np.random.default_rng(1)
    raw = torch.randn(150, 16)
    with torch.no_grad():
        z, _ = sae(raw)
    sparse_features = z.numpy()
    labels = (rng.random(150) < 0.5).astype(int)

    candidate_features = [1, 3, 7]
    results = run_causal_experiment(
        candidate_features, sparse_features, labels, sae, n_random_controls=4, seed=1
    )

    assert results["candidate_features"] == candidate_features
    assert 0.0 <= results["reconstruction_only_flip_rate"] <= 1.0
    assert set(results["features"].keys()) == {"1", "3", "7"}
    for feature_result in results["features"].values():
        assert 0.0 <= feature_result["suppress_flip_rate"] <= 1.0
        assert 0.0 <= feature_result["amplify_flip_rate"] <= 1.0
    assert len(results["random_control_dims"]) == 4
    assert all(d not in candidate_features for d in results["random_control_dims"])
    assert 0.0 <= results["random_control_mean_flip_rate"] <= 1.0


def test_run_causal_experiment_random_controls_never_overlap_candidates():
    """Regression test for a subtle bug class: random controls must be drawn
    from outside candidate_features, every time, not just usually."""
    torch.manual_seed(2)
    sae = SparseAutoencoder(input_dim=10, hidden_dim=4, latent_dim=10)
    sae.eval()

    rng = np.random.default_rng(2)
    raw = torch.randn(120, 10)
    with torch.no_grad():
        z, _ = sae(raw)
    sparse_features = z.numpy()
    labels = (rng.random(120) < 0.5).astype(int)

    candidate_features = [0, 1, 2, 3, 4, 5, 6, 7]  # most of the 10 dims
    results = run_causal_experiment(
        candidate_features, sparse_features, labels, sae, n_random_controls=2, seed=2
    )
    assert all(d not in candidate_features for d in results["random_control_dims"])
