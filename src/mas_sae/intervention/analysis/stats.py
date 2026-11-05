"""Every comparison is the same paired test, on one value per question.

  Δ        mean over questions of (a − b), in percentage points
  95% CI   percentile bootstrap, resampling questions
  p        paired permutation test: swapping a and b within a question flips the sign of its
           difference, so the null distribution is the mean difference under random sign flips.
           "greater" tests a > b (one-sided); "two-sided" tests a ≠ b.
  Holm     step-down adjustment within one family of tests.
"""

from __future__ import annotations

import numpy as np

CHUNK = 1000        # bootstrap / permutation replicates computed at a time, to bound memory


def bootstrap_ci(d: np.ndarray, n_bootstrap: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_bootstrap, CHUNK):
        rows = rng.integers(0, len(d), size=(min(CHUNK, n_bootstrap - start), len(d)))
        means.append(d[rows].mean(axis=1))
    lo, hi = np.quantile(np.concatenate(means), [0.025, 0.975])
    return [float(lo), float(hi)]


def permutation_p(d: np.ndarray, alternative: str, n_permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed + 1)
    observed = d.mean()
    as_extreme = 0
    for start in range(0, n_permutations, CHUNK):
        signs = rng.choice([-1.0, 1.0], size=(min(CHUNK, n_permutations - start), len(d)))
        null = (signs * d).mean(axis=1)
        if alternative == "greater":
            as_extreme += int((null >= observed - 1e-12).sum())
        elif alternative == "two-sided":
            as_extreme += int((np.abs(null) >= abs(observed) - 1e-12).sum())
        else:
            raise ValueError(f"Unknown alternative {alternative!r}.")
    return (1 + as_extreme) / (1 + n_permutations)


def paired_difference(a, b, alternative: str, n_bootstrap: int, n_permutations: int, seed: int) -> dict:
    """a − b paired by question (one value of each per question, in the same order)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("Paired values need one value of each per question.")
    d = a - b
    return {"n": int(len(d)), "delta_pp": 100 * float(d.mean()),
            "ci95_pp": [100 * v for v in bootstrap_ci(d, n_bootstrap, seed)],
            "p_value": permutation_p(d, alternative, n_permutations, seed), "alternative": alternative}


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
