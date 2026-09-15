from __future__ import annotations

import numpy as np
import torch

from mas_sae.probe.model import fit_probe
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


def downstream_decision(
    activation: np.ndarray,
    sae: SparseAutoencoder,
    probe,
    scaler,
) -> np.ndarray:
    """MOCK for 'continue the Solver forward pass and observe its decision.'
    Re-encodes the (possibly modified) activation and asks the probe. Replace
    this with a real call into the Solver once that hook exists."""
    with torch.no_grad():
        z = sae.encoder(torch.from_numpy(activation).float()).numpy()
    return probe.predict(scaler.transform(z))


def run_condition(
    z: np.ndarray,
    sae: SparseAutoencoder,
    probe,
    scaler,
    feature_idx: int | None = None,
    mode: str | None = None,
    amplify_scale: float = 3.0,
    amplify_shift: float = 1.0,
) -> np.ndarray:
    """Encode -> optionally ablate a feature -> decode -> downstream decision."""
    z = z.copy()
    if feature_idx is not None:
        if mode == "suppress":
            z[:, feature_idx] = 0.0
        elif mode == "amplify":
            z[:, feature_idx] = z[:, feature_idx] * amplify_scale + amplify_shift
    with torch.no_grad():
        activation = sae.decoder(torch.from_numpy(z).float()).numpy()
    return downstream_decision(activation, sae, probe, scaler)


def run_causal_experiment(
    candidate_features: list[int],
    sparse_features: np.ndarray,
    labels: np.ndarray,
    sae: SparseAutoencoder,
    n_random_controls: int = 4,
    amplify_scale: float = 3.0,
    amplify_shift: float = 1.0,
    seed: int = 42,
) -> dict:
    """Full causal intervention experiment: no-intervention baseline,
    reconstruction-only control, targeted suppress/amplify per candidate
    feature, and a matched random-feature control. Returns a results dict
    ready to be logged as JSON."""
    rng = np.random.default_rng(seed)

    probe, scaler, _, _, X_val, _ = fit_probe(sparse_features, labels, seed=seed)

    baseline_preds = probe.predict(scaler.transform(X_val))  # no-intervention replay

    recon_only_preds = run_condition(X_val, sae, probe, scaler)
    recon_only_flip = float((recon_only_preds != baseline_preds).mean())

    results = {
        "candidate_features": candidate_features,
        "no_intervention_baseline": "predictions on original latents, used as the reference for all flip rates",
        "reconstruction_only_flip_rate": recon_only_flip,
        "features": {},
    }

    for f in candidate_features:
        suppress_preds = run_condition(X_val, sae, probe, scaler, f, "suppress", amplify_scale, amplify_shift)
        amplify_preds = run_condition(X_val, sae, probe, scaler, f, "amplify", amplify_scale, amplify_shift)
        results["features"][str(f)] = {
            "suppress_flip_rate": float((suppress_preds != baseline_preds).mean()),
            "amplify_flip_rate": float((amplify_preds != baseline_preds).mean()),
        }

    all_dims = list(range(sparse_features.shape[1]))
    remaining = [d for d in all_dims if d not in candidate_features]
    random_controls = rng.choice(remaining, size=n_random_controls, replace=False).tolist()
    random_flips = [
        float((run_condition(X_val, sae, probe, scaler, d, "suppress", amplify_scale, amplify_shift) != baseline_preds).mean())
        for d in random_controls
    ]
    results["random_control_dims"] = random_controls
    results["random_control_mean_flip_rate"] = float(np.mean(random_flips))

    return results
