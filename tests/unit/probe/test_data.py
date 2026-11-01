"""Unit tests for mas_sae.probe.data."""

import numpy as np

from mas_sae.probe.data import make_sample_labels, make_sample_sae_features


def test_make_sample_sae_features_shape_and_nonnegativity():
    features = make_sample_sae_features(
        n_samples=50, input_dim=8, hidden_dim=4, latent_dim=16, seed=42
    )
    assert features.shape == (50, 16)
    # encoder ends in ReLU, so every value must be >= 0
    assert (features >= 0).all()


def test_make_sample_sae_features_is_reproducible_with_same_seed():
    a = make_sample_sae_features(n_samples=20, input_dim=8, hidden_dim=4, latent_dim=16, seed=1)
    b = make_sample_sae_features(n_samples=20, input_dim=8, hidden_dim=4, latent_dim=16, seed=1)
    assert np.array_equal(a, b)


def test_make_sample_sae_features_differs_with_different_seed():
    a = make_sample_sae_features(n_samples=20, input_dim=8, hidden_dim=4, latent_dim=16, seed=1)
    b = make_sample_sae_features(n_samples=20, input_dim=8, hidden_dim=4, latent_dim=16, seed=2)
    assert not np.array_equal(a, b)


def test_make_sample_labels_shape_and_binary():
    features = make_sample_sae_features(
        n_samples=100, input_dim=8, hidden_dim=4, latent_dim=16, seed=42
    )
    labels = make_sample_labels(features, signal_dims=[2, 7], seed=42)
    assert labels.shape == (100,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_make_sample_labels_uses_default_weights_up_to_signal_dims_length():
    """default_weights has 5 entries; passing fewer signal_dims than that
    should not error, and should use the first N weights."""
    features = make_sample_sae_features(
        n_samples=30, input_dim=8, hidden_dim=4, latent_dim=16, seed=1
    )
    labels_2_dims = make_sample_labels(features, signal_dims=[0, 1], seed=1)
    labels_5_dims = make_sample_labels(features, signal_dims=[0, 1, 2, 3, 4], seed=1)
    assert labels_2_dims.shape == (30,)
    assert labels_5_dims.shape == (30,)


def test_make_sample_labels_is_reproducible_with_same_seed():
    features = make_sample_sae_features(
        n_samples=40, input_dim=8, hidden_dim=4, latent_dim=16, seed=1
    )
    a = make_sample_labels(features, signal_dims=[2, 7], seed=3)
    b = make_sample_labels(features, signal_dims=[2, 7], seed=3)
    assert np.array_equal(a, b)


def test_make_sample_labels_signal_dims_correlate_more_than_noise_dims():
    """The label should be more predictable from the signal dims than from
    an arbitrary unrelated dim -- a lightweight correctness check that the
    planted-signal mechanism actually works."""
    features = make_sample_sae_features(
        n_samples=300, input_dim=8, hidden_dim=4, latent_dim=16, seed=1
    )
    signal_dims = [2, 7, 15]
    labels = make_sample_labels(features, signal_dims=signal_dims, seed=1)

    corr_signal = abs(np.corrcoef(features[:, 2], labels)[0, 1])
    corr_noise = abs(np.corrcoef(features[:, 0], labels)[0, 1])
    assert corr_signal > corr_noise