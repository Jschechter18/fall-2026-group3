"""
Probe pipeline: linear probe + SHAP/LIME interpretability smoke test.

Uses sample SAE features (generated with the real, untrained SparseAutoencoder)
since no real Solver-Critic activation data has been wired into this pipeline
yet (see issue #39). The reusable logic lives in
mas_sae.probe.{data,model,interpretability} so it can also be reused by the
activation-selection probe script.

Run from the repo root with the `capstone` conda env active:
    python scripts/probe_pipeline.py
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mas_sae.probe.data import make_sample_labels, make_sample_sae_features
from mas_sae.probe.interpretability import run_lime, run_shap, top_features_by_mean_abs_shap
from mas_sae.probe.model import fit_probe, train_val_split


@dataclass
class ProbePipelineConfig:
    seed: int = 42
    n_samples: int = 500              # TEMPORARY PLACEHOLDER: synthetic count until real interactions replace this (see #39)
    input_dim: int = 64               # synthetic smoke-test dimension, not tied to hyperparamters.py
    latent_dim: int = 64
    hidden_dim: int = 8
    signal_dims: list = field(default_factory=lambda: [2, 7, 15])   # TEMPORARY PLACEHOLDER: synthetic signal dims until real labels exist (see #39)
    n_lime_examples: int = 5
    top_k_shap_features: int = 10
    output_dir: Path = Path("logs/probe_smoke_test")


def main():
    config = ProbePipelineConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # TEMPORARY PLACEHOLDER: synthetic data until real Solver-Critic interactions replace this (see #39)
    sparse_features = make_sample_sae_features(
        config.n_samples, config.input_dim, config.hidden_dim, config.latent_dim, seed=config.seed
    )
    labels = make_sample_labels(sparse_features, config.signal_dims, seed=config.seed)

    X_train, X_val, y_train, y_val = train_val_split(sparse_features, labels, seed=config.seed)
    probe, scaler, metrics, X_train_s, X_val_s, y_val = fit_probe(
        X_train, y_train, X_val, y_val, seed=config.seed
    )
    print("Probe metrics:", metrics)

    shap_values = run_shap(probe, X_train_s, X_val_s)
    top_features, top_shap_vals = top_features_by_mean_abs_shap(
        shap_values, top_k=config.top_k_shap_features
    )
    print(f"Top {config.top_k_shap_features} features by mean |SHAP|:", top_features)
    print("Ground-truth signal dims were:", config.signal_dims)

    lime_results = run_lime(
        probe, X_train_s, X_val_s, n_examples=config.n_lime_examples, seed=config.seed
    )

    # --- log everything for the smoke test record ---
    np.save(config.output_dir / "sample_sae_features.npy", sparse_features)
    np.save(config.output_dir / "sample_labels.npy", labels)
    np.save(config.output_dir / "shap_values_val.npy", shap_values)

    with open(config.output_dir / "probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(config.output_dir / "top_shap_features.json", "w") as f:
        json.dump(
            {
                "top_10_feature_indices": top_features,
                "mean_abs_shap_top_10": top_shap_vals,
                "known_signal_dims": config.signal_dims,
            },
            f,
            indent=2,
        )

    with open(config.output_dir / "lime_explanations_sample.json", "w") as f:
        json.dump(lime_results, f, indent=2)

    print(f"\nAll smoke-test outputs written to {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
