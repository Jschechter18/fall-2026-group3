"""Offline (probe-read) SAE interventions on real activations, and the paired statistics that the
causal results table reports.

Each condition splices an SAE output back in place of the layer's activation, as in issue #90:

  no ablation   x' = decode(encode(x))                         (the SAE control)
  targeted      x' = decode(encode(x) with chosen features = 0)
  random        x' = decode(encode(x) with frequency-matched random features = 0), averaged over draws

The outcome for every row is P(accept the Critic) from a READOUT probe trained on raw (non-SAE)
activations of the training partition. This is a proxy: it says how the modified activation looks
to a probe of acceptance, not what the Solver would generate. The real experiment replaces the
readout with the Solver's regenerated answer; the table and statistics stay the same.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mas_sae.probe.cv_eval import grouped_bootstrap_indices, make_group_folds


# ----------------------------------------------------------------------------- SAE helpers

@torch.no_grad()
def encode(sae, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """SAE features for raw activations ``X`` (rows x input_dim), in batches."""
    out = [sae.encoder(torch.from_numpy(X[i:i + batch_size]).float()).numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out).astype(np.float32, copy=False)


@torch.no_grad()
def decode(sae, Z: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = [sae.decoder(torch.from_numpy(Z[i:i + batch_size]).float()).numpy()
           for i in range(0, len(Z), batch_size)]
    return np.concatenate(out).astype(np.float32, copy=False)


def ablate(Z: np.ndarray, features) -> np.ndarray:
    """Copy of ``Z`` with the given feature columns set to zero."""
    Z = Z.copy()
    Z[:, list(features)] = 0.0
    return Z


def frequency_matched_controls(freq: np.ndarray, targeted, n_draws: int, seed: int,
                               pool_size: int = 50) -> list[list[int]]:
    """Random feature sets the same size as ``targeted``: each targeted feature is replaced by a
    random live, non-targeted feature among the ``pool_size`` with the closest training firing
    frequency, so the control removes about as much activity as the targeted ablation."""
    targeted = [int(f) for f in targeted]
    candidates = np.setdiff1d(np.flatnonzero(freq > 0), targeted)
    if len(candidates) < len(targeted):
        raise ValueError("Not enough live features to draw a matched random control.")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        chosen: list[int] = []
        for f in targeted:
            pool = candidates[~np.isin(candidates, chosen)]
            near = pool[np.argsort(np.abs(np.log(freq[pool]) - np.log(freq[f])))[:pool_size]]
            chosen.append(int(rng.choice(near)))
        draws.append(chosen)
    return draws


# ----------------------------------------------------------------------------- readout probe

def _readout_model(C: float, seed: int) -> Pipeline:
    # Unweighted so predicted probabilities stay on the scale of the real acceptance rate.
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed))])


def fit_readout(X: np.ndarray, y: np.ndarray, groups: np.ndarray, c_grid, inner_folds: int,
                seed: int) -> tuple[Pipeline, float]:
    """Acceptance readout on raw activations; C chosen by inner grouped CV log loss on these rows
    only (ties go to the smaller C). ``y`` is 1 = accepted the Critic."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds, _ = make_group_folds(y, groups, inner_folds, seed)
        best_c, best = c_grid[0], np.inf
        for C in c_grid:
            losses = [log_loss(y[va], _readout_model(C, seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1],
                               labels=[0, 1]) for tr, va in folds]
            if np.mean(losses) < best - 1e-12:
                best_c, best = C, float(np.mean(losses))
        return _readout_model(best_c, seed).fit(X, y), best_c


def p_accept(readout: Pipeline, X: np.ndarray) -> np.ndarray:
    return readout.predict_proba(X)[:, list(readout.classes_).index(1)]


# ----------------------------------------------------------------------------- statistics

def paired_effect(outcome: np.ndarray, reference: np.ndarray, groups: np.ndarray,
                  n_bootstrap: int, n_permutations: int, seed: int) -> dict:
    """Difference in mean outcome (condition - reference) on the same rows, in percentage points,
    with a question-grouped paired bootstrap 95% CI and a two-sided paired sign-flip p-value."""
    d = np.asarray(outcome, float) - np.asarray(reference, float)
    groups = np.asarray(groups)
    if len(np.unique(groups)) != len(groups):
        raise ValueError("paired_effect expects one row per question.")
    boot = grouped_bootstrap_indices(groups, n_bootstrap, seed)
    means = np.array([d[i].mean() for i in boot])
    rng = np.random.default_rng(seed)
    observed = d.mean()
    null = (rng.choice([-1.0, 1.0], size=(n_permutations, len(d))) * d).mean(axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(observed) - 1e-15)) / (1 + n_permutations)
    return {"delta_pp": 100 * float(observed),
            "ci95_pp": [100 * float(np.quantile(means, 0.025)), 100 * float(np.quantile(means, 0.975))],
            "p_value": float(p)}


def targeted_vs_random(targeted: np.ndarray, random: np.ndarray, control: np.ndarray, groups: np.ndarray,
                       n_bootstrap: int, n_permutations: int, seed: int) -> dict:
    """Targeted ablation against the random-ablation controls, paired by question, in percentage points.

    ``targeted`` and ``control`` are one value per question; ``random`` is questions x draws. The
    direct comparison is targeted - mean over draws, with a question bootstrap CI and sign-flip
    p-value (both conditional on the draws used), plus a CI that also resamples the draws, so it
    includes the uncertainty from which random feature sets were chosen. ``draw_rank_p`` is the
    share of random sets (plus the targeted set itself) whose effect against the control is at
    least as large in magnitude as the targeted set's: a test over feature sets, limited by the
    number of draws (its smallest value is 1 / (draws + 1)).
    """
    targeted, control = np.asarray(targeted, float), np.asarray(control, float)
    random = np.asarray(random, float)
    direct = paired_effect(targeted, random.mean(1), groups, n_bootstrap, n_permutations, seed)
    rng = np.random.default_rng(seed + 1)
    n_draws = random.shape[1]
    both = [targeted[i].mean() - random[i][:, rng.integers(0, n_draws, n_draws)].mean()
            for i in grouped_bootstrap_indices(groups, n_bootstrap, seed)]
    t_effect = (targeted - control).mean()
    draw_effects = (random - control[:, None]).mean(0)
    return {**direct,
            "ci95_pp_questions_and_draws": [100 * float(np.quantile(both, q)) for q in (0.025, 0.975)],
            "n_random_draws": int(n_draws),
            "targeted_vs_control_pp": 100 * float(t_effect),
            "random_draw_vs_control_pp": [100 * float(e) for e in draw_effects],
            "draw_rank_p": float((1 + np.sum(np.abs(draw_effects) >= abs(t_effect) - 1e-15)) / (1 + n_draws))}


def holm(p_values) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control), in the input order."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
