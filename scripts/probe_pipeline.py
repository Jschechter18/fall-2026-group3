"""
Probe pipeline: linear probe + SHAP/LIME interpretability smoke test.

Uses sample SAE features (generated with the real, untrained SparseAutoencoder)
since no real Solver-Critic activation data exists yet. The reusable logic
lives in mas_sae.probe.{data,model,interpretability} so it can also be reused
by the activation-selection probe script (which needs the same probe/SHAP/LIME
machinery, just pointed at raw Solver-Critic forward-pass activations instead
of SAE features).

Run from the repo root with the `capstone` conda env active:
    python scripts/probe_pipeline.py
"""

import json
from pathlib import Path

import numpy as np

from mas_sae.probe.data import make_sample_labels, make_sample_sae_features
from mas_sae.probe.interpretability import run_lime, run_shap, top_features_by_mean_abs_shap
from mas_sae.probe.model import fit_probe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 500
INPUT_DIM = 64      # synthetic smoke-test dimension, not tied to hyperparamters.py
LATENT_DIM = 64
HIDDEN_DIM = 8
SIGNAL_DIMS = [2, 7, 15]   # latent dims we bake ground-truth signal into
OUTPUT_DIR = Path("logs/probe_smoke_test")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sparse_features = make_sample_sae_features(
        N_SAMPLES, INPUT_DIM, HIDDEN_DIM, LATENT_DIM, seed=SEED
    )
    labels = make_sample_labels(sparse_features, SIGNAL_DIMS, seed=SEED)

    probe, scaler, metrics, X_train_s, X_val_s, y_val = fit_probe(
        sparse_features, labels, seed=SEED
    )
    print("Probe metrics:", metrics)

    shap_values = run_shap(probe, X_train_s, X_val_s)
    top_features, top_shap_vals = top_features_by_mean_abs_shap(shap_values, top_k=10)
    print("Top 10 features by mean |SHAP|:", top_features)
    print("Ground-truth signal dims were:", SIGNAL_DIMS)

    lime_results = run_lime(probe, X_train_s, X_val_s, n_examples=5, seed=SEED)

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
                "mean_abs_shap_top_10": top_shap_vals,
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