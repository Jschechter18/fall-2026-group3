"""Dictionary geometry and thresholds, from the weights alone (no activations needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn.functional import cosine_similarity, normalize

from .stats import quantiles

if TYPE_CHECKING:
    from mas_sae.sae.pretrained import JumpReLUSAE

NEAR_DUPLICATE_COSINES = (0.9, 0.95)
CHUNK = 2048     # decoder rows compared at once: CHUNK x n_features similarities in memory


@torch.no_grad()
def nearest_neighbour_cosine(w_dec: torch.Tensor, among: torch.Tensor | None = None,
                             chunk: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Per feature: the largest |cosine| between its decoder row and any other feature's.

    Returns (max |cos| over all features, max |cos| over `among` (nan outside it), mean |cos| over all
    distinct pairs). `among` is an optional bool mask, e.g. the features alive on our data.
    """
    directions = normalize(w_dec.float(), dim=1)
    n = len(directions)
    nearest = torch.empty(n, device=directions.device)
    nearest_among = torch.full((n,), float('nan'), device=directions.device)
    total = 0.0
    for start in range(0, n, chunk):
        rows = torch.arange(start, min(start + chunk, n), device=directions.device)
        similarity = (directions[rows] @ directions.T).abs()
        local = torch.arange(len(rows), device=directions.device)
        total += (similarity.sum() - similarity[local, rows].sum()).item()
        similarity[local, rows] = -1.0                        # a feature is not its own neighbour
        nearest[rows] = similarity.max(dim=1).values
        if among is not None and int(among.sum()) > 1 and among[rows].any():
            keep = among[rows]
            nearest_among[rows[keep]] = similarity[keep][:, among].max(dim=1).values
    return nearest, nearest_among, total / (n * (n - 1)) if n > 1 else None


def _near_duplicates(nearest):
    nearest = nearest[~nearest.isnan()]
    return {'mean': nearest.mean().item() if len(nearest) else None,
            'max': nearest.max().item() if len(nearest) else None,
            **quantiles(nearest, (0.5, 0.9, 0.99)),
            **{f'count_over_{c:g}': int((nearest > c).sum()) for c in NEAR_DUPLICATE_COSINES}}


@torch.no_grad()
def parameter_health(model: JumpReLUSAE,
                     alive: torch.Tensor | None = None) -> tuple[dict, dict[str, torch.Tensor]]:
    """Returns (summary, per-feature arrays) for a JumpReLU SAE's weights.

    `alive` (optional bool mask) adds versions restricted to features that fire on our data: a dead
    feature's direction matters for neither reconstruction nor interventions.
    """
    w_dec = model.w_dec.detach()
    norms = w_dec.float().norm(dim=1)
    nearest, nearest_alive, mean_abs_cos = nearest_neighbour_cosine(w_dec, alive)
    alignment = cosine_similarity(model.w_enc.detach().float().T, w_dec.float(), dim=1)
    summary = {
        'decoder_norm': {**quantiles(norms, (0.01, 0.5, 0.99)), 'min': norms.min().item(), 'max': norms.max().item()},
        'mean_pairwise_abs_cosine': mean_abs_cos,
        'nearest_neighbour_abs_cosine': _near_duplicates(nearest),
        'encoder_decoder_alignment': {**quantiles(alignment, (0.01, 0.1, 0.5)), 'min': alignment.min().item()},
        'threshold': quantiles(model.threshold, (0.01, 0.1, 0.5, 0.9, 0.99)),
    }
    if alive is not None:
        summary['nearest_neighbour_abs_cosine_alive'] = _near_duplicates(nearest_alive[alive])
        summary['encoder_decoder_alignment_alive'] = quantiles(alignment[alive], (0.01, 0.1, 0.5))
    per_feature = {'decoder_norm': norms, 'nearest_neighbour_abs_cosine': nearest,
                   'encoder_decoder_alignment': alignment, 'threshold': model.threshold.detach()}
    return summary, per_feature
