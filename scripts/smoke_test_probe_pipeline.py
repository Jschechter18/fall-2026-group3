"""
Smoke test: linear probe + interpretability (SHAP/LIME) pipeline.

Since no real Solver-Critic activation data exists yet, this script generates
synthetic "sample SAE features" by passing random raw activations through the
real (untrained) SparseAutoencoder class. This exercises the exact interface
the rest of the team will produce, while letting this issue's pipeline be
built and validated end-to-end right now.

Run from the repo root with the `capstone` conda env active:
    python scripts/smoke_test_probe_pipeline.py

Requires shap and lime in addition to the project's existing deps:
    pip install shap lime
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap
from lime.lime_tabular import LimeTabularExplainer

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 500
INPUT_DIM = 64      # placeholder, matches current hyperparamters.py
LATENT_DIM = 64
HIDDEN_DIM = 8
SIGNAL_DIMS = [2, 7, 15]   # latent dims we bake ground-truth signal into
OUTPUT_DIR = Path("logs/probe_smoke_test")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def make_sample_sae_features() -> np.ndarray:
    """Generate sample SAE features by pushing random raw activations through
    the real (untrained) SparseAutoencoder encoder. Stand-in for real
    Solver-Critic activations until Objective 1's pipeline produces them."""
    sae = SparseAutoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    sae.eval()
    raw_activations = torch.randn(N_SAMPLES, INPUT_DIM)
    with torch.no_grad():
        sparse_features, _ = sae(raw_activations)
    return sparse_features.numpy()


def make_sample_labels(sparse_features: np.ndarray) -> np.ndarray:
    """Bake a known signal into a few latent dims so the probe/SHAP results
    can be checked against ground truth. Stand-in for solver_accepted_feedback."""
    logit = (
        3.0 * sparse_features[:, SIGNAL_DIMS[0]]
        - 2.0 * sparse_features[:, SIGNAL_DIMS[1]]
        + 1.5 * sparse_features[:, SIGNAL_DIMS[2]]
    )
    logit = logit - logit.mean()
    prob = 1 / (1 + np.exp(-logit))
    return (rng.random(N_SAMPLES) < prob).astype(int)


def train_probe(X_train, y_train, X_val, y_val) -> tuple[LogisticRegression, dict]:
    probe = LogisticRegression(l1_ratio=1.0, solver="saga", C=0.5, max_iter=2000, random_state=SEED)
    probe.fit(X_train, y_train)

    val_probs = probe.predict_proba(X_val)[:, 1]
    val_preds = probe.predict(X_val)
    metrics = {
        "auroc": float(roc_auc_score(y_val, val_probs)),
        "f1": float(f1_score(y_val, val_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_preds)),
    }
    return probe, metrics


def run_shap(probe, X_train, X_val) -> np.ndarray:
    explainer = shap.LinearExplainer(probe, X_train)
    shap_values = explainer.shap_values(X_val)
    return shap_values


def run_lime(probe, X_train, X_val, n_examples: int = 5) -> list[dict]:
    explainer = LimeTabularExplainer(
        X_train,
        mode="classification",
        feature_names=[f"latent_{i}" for i in range(X_train.shape[1])],
        discretize_continuous=False,
        random_state=SEED,
    )
    results = []
    for i in range(min(n_examples, len(X_val))):
        exp = explainer.explain_instance(X_val[i], probe.predict_proba, num_features=10)
        results.append({"example_index": i, "explanation": exp.as_list()})
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sparse_features = make_sample_sae_features()
    labels = make_sample_labels(sparse_features)

    X_train, X_val, y_train, y_val = train_test_split(
        sparse_features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    probe, metrics = train_probe(X_train_s, y_train, X_val_s, y_val)
    print("Probe metrics:", metrics)

    shap_values = run_shap(probe, X_train_s, X_val_s)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(mean_abs_shap)[::-1][:10].tolist()
    print("Top 10 features by mean |SHAP|:", top_features)
    print("Ground-truth signal dims were:", SIGNAL_DIMS)

    lime_results = run_lime(probe, X_train_s, X_val_s, n_examples=5)

    # --- log everything for the smoke test record ---
    np.save(OUTPUT_DIR / "sample_sae_features.npy", sparse_features)
    np.save(OUTPUT_DIR / "sample_labels.npy", labels)
    np.save(OUTPUT_DIR / "shap_values_val.npy", shap_values)

    with open(OUTPUT_DIR / "probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUT_DIR / "top_shap_features.json", "w") as f:
        json.dump(
            {
                "top_10_feature_indices": top_features,
                "mean_abs_shap_top_10": mean_abs_shap[top_features].tolist(),
                "known_signal_dims": SIGNAL_DIMS,
            },
            f,
            indent=2,
        )

    with open(OUTPUT_DIR / "lime_explanations_sample.json", "w") as f:
        json.dump(lime_results, f, indent=2)

    print(f"\nAll smoke-test outputs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()