"""Unit tests for mas_sae.probe.interpretability."""

import numpy as np
import pytest
import shap
from sklearn.linear_model import LogisticRegression

from mas_sae.probe.interpretability import (
    run_lime,
    run_shap,
    top_features_by_mean_abs_shap,
)


@pytest.fixture
def trained_probe():
    """A small, fast, real logistic regression with a known planted signal,
    fit directly here (not via fit_probe) to keep this test file's
    dependencies isolated to what interpretability.py itself needs."""
    rng = np.random.default_rng(0)
    n_samples, dim = 150, 10
    X = rng.normal(size=(n_samples, dim))
    signal_dim = 3
    logit = 2.5 * X[:, signal_dim]
    prob = 1 / (1 + np.exp(-(logit - logit.mean())))
    y = (rng.random(n_samples) < prob).astype(int)

    probe = LogisticRegression(max_iter=1000, random_state=0).fit(X, y)
    return probe, X, signal_dim


def test_run_shap_returns_correct_shape(trained_probe):
    probe, X, _ = trained_probe
    shap_values = run_shap(probe, X, X)
    assert shap_values.shape == X.shape


def test_run_shap_values_satisfy_local_accuracy(trained_probe):
    """Shapley values satisfy 'local accuracy': each example's attributions,
    summed, plus the baseline (expected_value), equal the model's actual
    output for that example (in decision_function / log-odds space). This
    holds regardless of whether the explainer accounts for correlation
    between features, unlike a naive coefficient*(x-mean) formula -- which
    only holds under an independence assumption real data rarely satisfies
    exactly."""
    probe, X, _ = trained_probe
    explainer = shap.LinearExplainer(probe, X)
    shap_values = explainer.shap_values(X)

    model_output = probe.decision_function(X)
    reconstructed = explainer.expected_value + shap_values.sum(axis=1)

    assert np.allclose(reconstructed, model_output, atol=1e-6)


def test_top_features_by_mean_abs_shap_ranks_planted_signal_first(trained_probe):
    probe, X, signal_dim = trained_probe
    shap_values = run_shap(probe, X, X)
    top_features, top_values = top_features_by_mean_abs_shap(shap_values, top_k=3)

    assert len(top_features) == 3
    assert len(top_values) == 3
    assert top_features[0] == signal_dim
    # values should be sorted descending
    assert top_values == sorted(top_values, reverse=True)


def test_run_lime_returns_expected_structure(trained_probe):
    probe, X, _ = trained_probe
    results = run_lime(probe, X, X, n_examples=3, seed=0)

    assert len(results) == 3
    for i, entry in enumerate(results):
        assert entry["example_index"] == i
        assert isinstance(entry["explanation"], list)
        assert len(entry["explanation"]) > 0
        for feature_name, weight in entry["explanation"]:
            assert isinstance(feature_name, str)
            assert isinstance(weight, float)


def test_run_lime_respects_n_examples_cap(trained_probe):
    """Requesting more examples than exist in X_val should cap at len(X_val),
    not error."""
    probe, X, _ = trained_probe
    small_X = X[:2]
    results = run_lime(probe, X, small_X, n_examples=10, seed=0)
    assert len(results) == 2