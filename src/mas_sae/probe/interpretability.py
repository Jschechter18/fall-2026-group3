from __future__ import annotations

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression


def run_shap(probe: LogisticRegression, X_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    """Exact linear SHAP values for a logistic regression probe."""
    explainer = shap.LinearExplainer(probe, X_train)
    return explainer.shap_values(X_val)


def run_lime(
    probe: LogisticRegression,
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_prefix: str = "latent",
    n_examples: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Local LIME explanations for a handful of validation examples, as a
    cross-check against SHAP's global feature ranking."""
    explainer = LimeTabularExplainer(
        X_train,
        mode="classification",
        feature_names=[f"{feature_prefix}_{i}" for i in range(X_train.shape[1])],
        discretize_continuous=False,
        random_state=seed,
    )
    results = []
    for i in range(min(n_examples, len(X_val))):
        exp = explainer.explain_instance(X_val[i], probe.predict_proba, num_features=10)
        results.append({"example_index": i, "explanation": exp.as_list()})
    return results


def top_features_by_mean_abs_shap(
    shap_values: np.ndarray, top_k: int = 10
) -> tuple[list[int], list[float]]:
    """Rank features by mean |SHAP value| across validation examples."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    return top_indices.tolist(), mean_abs_shap[top_indices].tolist()
