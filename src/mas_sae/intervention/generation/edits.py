"""The edits applied to the Solver's layer-25 activation x at the final prompt token.

Every edit is a steering edit of one fixed size: x' = x + Δ with |Δ| = r·|x|. Fixing the size makes
every arm, and every targeted and random feature set, the same size of push, so a difference in flip
rate can only come from the direction. (Zeroing features was rejected: an ablation is a no-op when
the features are not firing, and its size differs from set to set.)

A feature set S has one direction, u_S = normalize(Σ_{i∈S} d_i), where d_i is the direction that
means "more of feature i":
  - SAE: feature i's unit decoder row, i.e. what one unit of the feature adds to the reconstruction.
  - Raw activations: the unit basis vector e_i, times the sign of neuron i's probe coefficient
    toward its set's label, so that +d_i means "more of that label" for neurons too.

For an episode, `toward` is the set of the label we try to flip to and `away` the set of its stored
label (adoption episode: toward = retention, away = adoption). The three arms:
  increase   Δ = +r·|x|·u_toward
  decrease   Δ = −r·|x|·u_away              (a steering edit, never a no-op)
  both       Δ = r·|x|·normalize(u_toward − u_away)   one combined edit of size r, not 2r
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

ARMS = ("decrease", "increase", "both")


def normalize(v: Tensor) -> Tensor:
    """``v`` divided by its norm along the last axis. Raises ValueError if any vector is zero."""
    norm = v.norm(dim=-1, keepdim=True)
    if (norm == 0).any():
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm


@torch.no_grad()
def sae_feature_directions(sae: torch.nn.Module, features: list[int]) -> Tensor:
    """Unit decoder rows of ``features``, one per row, in activation units.

    Both SAE kinds decode affinely, so decode(e_i) − decode(0) is feature i's decoder row with any
    input scaling applied, whichever way the SAE stores its decoder weights."""
    unit = torch.zeros(len(features), sae.latent_dim)
    unit[torch.arange(len(features)), torch.tensor(features)] = 1.0
    rows = sae.decoder(unit) - sae.decoder(torch.zeros(1, sae.latent_dim))
    return normalize(rows.float())


def raw_feature_directions(features: list[int], signs: list[int], width: int) -> Tensor:
    """sign_i·e_i for each neuron, one per row. ``signs`` is the sign of each neuron's probe
    coefficient toward its set's label (+1 or −1), so +d_i always means "more of that label"."""
    if len(signs) != len(features) or any(s not in (1, -1) for s in signs):
        raise ValueError("Each neuron needs a sign of +1 or -1.")
    rows = torch.zeros(len(features), width)
    rows[torch.arange(len(features)), torch.tensor(features)] = torch.tensor(signs, dtype=torch.float32)
    return rows


def set_direction(feature_directions: Tensor) -> Tensor:
    """u_S = normalize(Σ d_i): the one direction a feature set is steered along."""
    return normalize(feature_directions.sum(dim=0))


def edit_delta(arm: str, x: Tensor, toward: Tensor, away: Tensor, size: float) -> Tensor:
    """Δ for one arm, with |Δ| = size·|x| for every row of ``x``.

    ``toward`` and ``away`` are the unit set directions of the label we push toward and of the
    stored label we push away from."""
    if arm == "increase":
        direction = toward
    elif arm == "decrease":
        direction = -away
    elif arm == "both":
        direction = normalize(toward - away)
    else:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}.")
    return size * x.norm(dim=-1, keepdim=True) * direction


@torch.no_grad()
def reconstruction_splice(sae: torch.nn.Module, x: Tensor) -> Tensor:
    """x' = decode(encode(x)): swaps x for the SAE's reconstruction, with no edit."""
    return sae.decoder(sae.encoder(x))


@dataclass(frozen=True)
class EditDiagnostics:
    """What a reader needs to check that an edit did what was intended."""
    applied_relative_size: float          # |Δ| / |x|; should equal the configured r
    edited_features_active: int | None    # edited features firing at x before the edit (None: raw neurons, dense)
    reconstruction_error: float | None    # |x − decode(encode(x))| / |x| (None: raw activations)


@torch.no_grad()
def diagnose(x: Tensor, delta: Tensor, edited_features: list[int],
             z: Tensor | None = None, reconstruction: Tensor | None = None) -> EditDiagnostics:
    """Diagnostics for one edit of a single activation ``x`` (shape [d]).

    ``z`` = encode(x) and ``reconstruction`` = decode(z) are computed once per episode and SAE and
    passed in; both are None for raw activations, whose neurons are dense (so "firing" and
    reconstruction error are undefined)."""
    x, delta = x.float(), delta.float()
    applied = float(delta.norm() / x.norm())
    if z is None:
        return EditDiagnostics(applied, None, None)
    return EditDiagnostics(
        applied_relative_size=applied,
        edited_features_active=int((z[edited_features] > 0).sum()),
        reconstruction_error=float((x - reconstruction.float()).norm() / x.norm()),
    )


def set_cosine(u_adoption: Tensor, u_retention: Tensor) -> float:
    """Cosine between the two set directions, once per representation. Near −1 means "decrease
    adoption" and "increase retention" are nearly the same edit."""
    return float(normalize(u_adoption) @ normalize(u_retention))
