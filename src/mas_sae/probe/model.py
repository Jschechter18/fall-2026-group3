from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Default random train/val split. Kept separate from fit_probe so callers
    can substitute their own split (e.g. the discovery/validation/intervention
    splits from the proposal) instead of always getting a fresh random one."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    C: float = 0.5,
):
    """Standardize and fit an L1-regularized logistic regression probe on an
    already-split train/val set. Use train_val_split() above to get a default
    random split, or pass in your own split (e.g. real discovery/validation
    interactions) once that exists.

    Returns (probe, scaler, metrics, X_train_scaled, X_val_scaled, y_val).
    """
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    probe = LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=2000, random_state=seed)
    probe.fit(X_train_s, y_train)

    val_probs = probe.predict_proba(X_val_s)[:, 1]
    val_preds = probe.predict(X_val_s)
    metrics = {
        "auroc": float(roc_auc_score(y_val, val_probs)),
        "f1": float(f1_score(y_val, val_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_preds)),
    }
    return probe, scaler, metrics, X_train_s, X_val_s, y_val
