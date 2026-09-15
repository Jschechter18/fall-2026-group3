"""
Causal pipeline: SAE feature intervention smoke test.

Pipeline (per the issue spec):
    activations -> SAE encoder -> sparse latent vector -> ablate/amplify a
    feature -> SAE decoder -> activations that get passed into the rest of
    the LLM.

Reuses the dataset saved by probe_pipeline.py (logs/probe_smoke_test/) --
the candidate features it flagged as promising, plus the sample SAE
features/labels -- and the reusable mas_sae.causal.intervention module for
the actual ablate/decode/measure logic.

Since no real Solver forward-pass hook exists yet, downstream_decision() in
mas_sae/causal/intervention.py is a clearly-marked mock: it re-encodes the
reconstructed activation and asks the trained probe for a decision instead
of continuing a real generation. Swap that one function for Israel's real
Solver continuation once it exists -- everything else stays the same.

Run from the repo root with the `capstone` conda env active:
    python scripts/causal_pipeline.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from mas_sae.causal.intervention import run_causal_experiment
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
INPUT_DIM = 64 #synthetic smoke-test dimension, not tied to hyperparamters.py
LATENT_DIM = 64
HIDDEN_DIM = 8
N_CANDIDATE_FEATURES = 3   # how many top features (from probe_pipeline.py) to test causally
N_RANDOM_CONTROLS = 4
AMPLIFY_SCALE = 3.0
AMPLIFY_SHIFT = 1.0

PROBE_DATA_DIR = Path("logs/probe_smoke_test")
OUTPUT_DIR = Path("logs/causal_intervention_smoke_test")

torch.manual_seed(SEED)


def load_promising_features_and_data():
    """Load the dataset saved by probe_pipeline.py: which features looked
    most promising, plus the sample SAE features/labels to reuse."""
    with open(PROBE_DATA_DIR / "top_shap_features.json") as f:
        top_features = json.load(f)
    sparse_features = np.load(PROBE_DATA_DIR / "sample_sae_features.npy")
    labels = np.load(PROBE_DATA_DIR / "sample_labels.npy")
    candidate_features = top_features["top_10_feature_indices"][:N_CANDIDATE_FEATURES]
    return candidate_features, sparse_features, labels


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_features, sparse_features, labels = load_promising_features_and_data()
    print("Candidate features loaded from probe_pipeline.py:", candidate_features)

    sae = SparseAutoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    sae.eval()

    results = run_causal_experiment(
        candidate_features,
        sparse_features,
        labels,
        sae,
        n_random_controls=N_RANDOM_CONTROLS,
        amplify_scale=AMPLIFY_SCALE,
        amplify_shift=AMPLIFY_SHIFT,
        seed=SEED,
    )

    print(f"Reconstruction-only control flip rate: {results['reconstruction_only_flip_rate']:.3f}")
    for f, r in results["features"].items():
        print(f"feature {f}: suppress flip={r['suppress_flip_rate']:.3f}, amplify flip={r['amplify_flip_rate']:.3f}")
    print(f"Random control dims {results['random_control_dims']}: "
          f"mean flip rate = {results['random_control_mean_flip_rate']:.3f}")

    with open(OUTPUT_DIR / "causal_intervention_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll causal smoke-test outputs written to {OUTPUT_DIR.resolve()}")
    print(
        "\nNOTE: the SAE here is untrained (no real activation data exists yet), so "
        "suppress/amplify flip rates will look similar to the reconstruction-only and "
        "random-feature controls -- that's expected. This confirms the pipeline mechanics "
        "work end-to-end; a real causal signal (targeted features flipping decisions more "
        "than random ones) requires a trained SAE on real Solver-Critic activations."
    )


if __name__ == "__main__":
    main()
