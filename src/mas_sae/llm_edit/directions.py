"""Set directions and the mean-difference direction d, as host-side diagnostics of an edited decoder.

Stage 6 steers along set directions: u_S = normalize(Σ_{i∈S} unit decoder row i), and the "both" arm
along normalize(u_A − u_R) (edits.sae_feature_directions / edits.set_direction). For a Gemma Scope SAE
decode(e_i) − decode(0) is exactly row i of w_dec, so these are computed straight from w_dec here.

d = normalize(mean(train adoption activations) − mean(train retention activations)) on the stage-5
train rows (the sandbox inputs' X_train, y_train). It is the mechanical control's target, so it is
only ever computed on the host: the LLM is never shown d or cosines with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def rotated_toward(row: np.ndarray, target: np.ndarray, cosine: float) -> np.ndarray:
    """|r|·(c·u + sqrt(1 − c²)·p): cosine c with r, as far toward `target` as c allows, same length.

    p is the unit part of `target` perpendicular to u = r/|r|; moving along p is the shortest way from u
    toward `target`, so for a fixed cosine with r this is the row closest to `target`. The controls use
    it with target ±d (mechanical edit) and with a random Gaussian vector (random edit)."""
    length = np.linalg.norm(np.asarray(row, dtype=np.float64))
    u = unit(row)
    p = unit(target - (target @ u) * u)
    return length * (cosine * u + np.sqrt(1 - cosine ** 2) * p)


def mean_difference_direction(inputs_dir: Path) -> np.ndarray:
    """d from the sandbox inputs (the stage-5 train rows): adoption mean minus retention mean, unit length."""
    X = np.load(Path(inputs_dir) / "X_train.npy", mmap_mode="r")
    y = np.load(Path(inputs_dir) / "y_train.npy")
    return unit(np.asarray(X[y == 1], dtype=np.float64).mean(0) - np.asarray(X[y == 0], dtype=np.float64).mean(0))


def set_directions(w_dec: np.ndarray, selected: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """u_adoption, u_retention and the 'both'-arm direction normalize(u_adoption − u_retention)."""
    u = {label: unit(sum(unit(w_dec[f]) for f in features)) for label, features in selected.items()}
    return {"adoption": u["adoption"], "retention": u["retention"], "both": unit(u["adoption"] - u["retention"])}


def direction_report(w_dec: np.ndarray, original_w_dec: np.ndarray, selected: dict[str, list[int]],
                     d: np.ndarray) -> dict:
    """How far each set direction moved, and how close it is to d (retention is compared with −d)."""
    now, before = set_directions(w_dec, selected), set_directions(original_w_dec, selected)
    toward = {"adoption": d, "retention": -d, "both": d}
    return {name: {"cosine_with_original": float(now[name] @ before[name]),
                   "cosine_with_d": float(now[name] @ toward[name]),
                   "original_cosine_with_d": float(before[name] @ toward[name])} for name in now}
