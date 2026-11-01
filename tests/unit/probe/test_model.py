"""Unit tests for mas_sae.probe.model."""

import numpy as np
import pytest

from mas_sae.probe.model import fit_probe, train_val_split


@pytest.fixture
def synthetic_data():
    """Reusable synthetic latent features + labels with a known planted
    signal, so tests can assert the probe actually recovers real structure
    rather than just checking it runs."""
    rng = np.random.default_rng(42)
    n_samples, dim = 200, 16
    X = rng.normal(size=(n_samples, dim))
    signal_dims = [1, 5]
    logit = 2.0 * X[:, 1] - 1.5 * X[:, 5]
    prob = 1 / (1 + np.exp(-(logit - logit.mean())))
    y = (rng.random(n_samples) < prob).astype(int)
    return X, y, signal_dims


def test_train_val_split_shapes_and_stratification(synthetic_data):
    X, y, _ = synthetic_data
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2, seed=42)

    assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    # roughly 20% held out
    assert abs(X_val.shape[0] / X.shape[0] - 0.2) < 0.05


def test_train_val_split_is_reproducible_with_same_seed(synthetic_data):
    X, y, _ = synthetic_data
    split_a = train_val_split(X, y, seed=7)
    split_b = train_val_split(X, y, seed=7)
    for a, b in zip(split_a, split_b):
        assert np.array_equal(a, b)


def test_train_val_split_differs_with_different_seed(synthetic_data):
    X, y, _ = synthetic_data
    X_train_a, *_ = train_val_split(X, y, seed=1)
    X_train_b, *_ = train_val_split(X, y, seed=2)
    assert not np.array_equal(X_train_a, X_train_b)


def test_fit_probe_returns_expected_shapes_and_types(synthetic_data):
    X, y, _ = synthetic_data
    X_train, X_val, y_train, y_val_split = train_val_split(X, y, seed=42)
    probe, scaler, metrics, X_train_s, X_val_s, y_val = fit_probe(
        X_train, y_train, X_val, y_val_split, seed=42
    )

    assert X_train_s.shape == X_train.shape
    assert X_val_s.shape == X_val.shape
    assert np.array_equal(y_val, y_val_split)
    assert set(metrics.keys()) == {"auroc", "f1", "balanced_accuracy"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_fit_probe_scaler_produces_standardized_training_data(synthetic_data):
    X, y, _ = synthetic_data
    X_train, X_val, y_train, y_val = train_val_split(X, y, seed=42)
    _, _, _, X_train_s, _, _ = fit_probe(X_train, y_train, X_val, y_val, seed=42)

    # StandardScaler should center training data near mean 0, std 1 per feature
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X_train_s.std(axis=0), 1, atol=1e-6)


def test_fit_probe_recovers_planted_signal_dims(synthetic_data):
    """The probe should assign larger |coefficient| to the dims that actually
    drive the label than to unrelated noise dims -- a correctness check, not
    just a shape check."""
    X, y, signal_dims = synthetic_data
    X_train, X_val, y_train, y_val = train_val_split(X, y, seed=42)
    probe, *_ = fit_probe(X_train, y_train, X_val, y_val, seed=42)

    coefs = np.abs(probe.coef_[0])
    signal_strength = coefs[signal_dims].mean()
    noise_dims = [d for d in range(X.shape[1]) if d not in signal_dims]
    noise_strength = coefs[noise_dims].mean()

    assert signal_strength > noise_strength


def test_fit_probe_is_reproducible_with_same_seed(synthetic_data):
    X, y, _ = synthetic_data
    X_train, X_val, y_train, y_val = train_val_split(X, y, seed=42)
    _, _, metrics_a, _, _, _ = fit_probe(X_train, y_train, X_val, y_val, seed=42)
    _, _, metrics_b, _, _, _ = fit_probe(X_train, y_train, X_val, y_val, seed=42)
    assert metrics_a == metrics_b
