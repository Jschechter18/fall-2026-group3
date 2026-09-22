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

downstream_decision() in mas_sae/causal/intervention.py is a clearly-marked mock: 
it re-encodes the reconstructed activation and asks the trained probe for a decision instead
of continuing a real generation. 

Run from the repo root with the `capstone` conda env active:
    python scripts/causal_pipeline.py
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from mas_sae.causal.intervention import run_causal_experiment
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


@dataclass
class CausalPipelineConfig:
    seed: int = 42
    input_dim: int = 64        # synthetic smoke-test dimension, not tied to hyperparamters.py
    latent_dim: int = 64
    hidden_dim: int = 8
    n_candidate_features: int = 3   # how many top features (from probe_pipeline.py) to test causally
    n_random_controls: int = 4
    amplify_scale: float = 3.0
    amplify_shift: float = 1.0
    probe_data_dir: Path = Path("logs/probe_smoke_test")
    results_root: Path = Path("results/causal_intervention")


def create_versioned_run_dir(results_root: Path) -> Path:
    """Create a fresh, timestamped subdirectory under results_root for this
    run's outputs, so repeated runs don't overwrite each other. Mirrors the
    versioning style already used in mas_sae/experiments/artifacts.py for SAE
    training runs, for consistency across the repo."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = results_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_promising_features_and_data(config: CausalPipelineConfig):
    """Load the dataset saved by probe_pipeline.py: which features looked
    most promising, plus the sample SAE features/labels to reuse."""
    with open(config.probe_data_dir / "top_shap_features.json") as f:
        top_features = json.load(f)
    sparse_features = np.load(config.probe_data_dir / "sample_sae_features.npy")
    labels = np.load(config.probe_data_dir / "sample_labels.npy")
    candidate_features = top_features["top_10_feature_indices"][: config.n_candidate_features]
    return candidate_features, sparse_features, labels


def main():
    config = CausalPipelineConfig()
    torch.manual_seed(config.seed)
    run_dir = create_versioned_run_dir(config.results_root)

    candidate_features, sparse_features, labels = load_promising_features_and_data(config)
    print("Candidate features loaded from probe_pipeline.py:", candidate_features)

    sae = SparseAutoencoder(
        input_dim=config.input_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim
    )
    sae.eval()

    results = run_causal_experiment(
        candidate_features,
        sparse_features,
        labels,
        sae,
        n_random_controls=config.n_random_controls,
        amplify_scale=config.amplify_scale,
        amplify_shift=config.amplify_shift,
        seed=config.seed,
    )

    print(f"Reconstruction-only control flip rate: {results['reconstruction_only_flip_rate']:.3f}")
    for f, r in results["features"].items():
        print(f"feature {f}: suppress flip={r['suppress_flip_rate']:.3f}, amplify flip={r['amplify_flip_rate']:.3f}")
    print(f"Random control dims {results['random_control_dims']}: "
          f"mean flip rate = {results['random_control_mean_flip_rate']:.3f}")

    with open(run_dir / "causal_intervention_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll causal smoke-test outputs written to {run_dir.resolve()}")
    print(
        "\nNOTE: the SAE here is still untrained. Real Solver-Critic activation data now "
        "exists (the 12-question pilot, logs/issue14/pilot_12q/), but this script hasn't "
        "been wired up to use it yet -- that's tracked in issue #39. Until then, "
        "suppress/amplify flip rates will look similar to the reconstruction-only and "
        "random-feature controls, which is expected on synthetic data with an untrained SAE. "
        "A real causal signal (targeted features flipping decisions more than random ones) "
        "requires a trained SAE on the real activations."
    )


if __name__ == "__main__":
    main()
