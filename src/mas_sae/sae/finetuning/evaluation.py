"""Reconstruction quality and sparsity, and the bias-only reference point."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch

from .data import features_for, iterate_batches

if TYPE_CHECKING:
    from mas_sae.sae.pretrained import JumpReLUSAE

    from .data import Split


def evaluate_splits(model: JumpReLUSAE, splits: dict[str, Split], batch_size: int,
                    device: str | torch.device) -> dict[str, dict[str, Any]]:
    """evaluate() on every split. Returns {'train': metrics, 'validation': metrics}."""
    return {name: evaluate(model, split, batch_size, device) for name, split in splits.items()}


@torch.no_grad()
def evaluate(model: JumpReLUSAE, split: Split, batch_size: int,
             device: str | torch.device) -> dict[str, Any]:
    """Reconstruction quality and sparsity of `model` on one split.

    Returns {'n', 'mse', 'variance_explained', 'l0', 'dead_fraction'}, plus 'probe_cohort' (n, mse,
    variance_explained on the probe's rows) when the split has one.

      - mse: mean squared error per activation dimension.
      - variance_explained = 1 - (squared error) / (squared distance from the split's mean). The
        probe cohort uses its own mean. The means are fixed per split, so before/after numbers compare.
      - l0: mean number of active features per example.
      - dead_fraction: share of features that never fire anywhere in the split.
    """
    everyone = ReconstructionTotals()
    cohort = ReconstructionTotals()
    total_active = 0
    ever_active = torch.zeros(model.latent_dim, dtype=torch.bool, device=device)
    split_mean = split.mean.to(device)
    for rows, batch in iterate_batches(split.activations, batch_size, device):
        features = features_for(model, split, rows, batch)
        is_active = features > 0
        total_active += int(is_active.sum())
        ever_active |= is_active.any(dim=0)
        squared_error = (model.decoder(features) - batch).double().square().sum(dim=1)   # one per example
        everyone.add(squared_error, (batch.double() - split_mean).square().sum(dim=1))
        if split.probe_cohort is not None:
            in_cohort = torch.from_numpy(split.probe_cohort[rows]).to(device)
            cohort_batch = batch[in_cohort].double()
            cohort.add(squared_error[in_cohort],
                       (cohort_batch - split.probe_cohort_mean.to(device)).square().sum(dim=1))
    n_dims = split.activations.shape[1]
    result = everyone.metrics(n_dims)
    result['l0'] = total_active / len(split.activations)
    result['dead_fraction'] = 1 - ever_active.float().mean().item()
    if split.probe_cohort is not None:
        result['probe_cohort'] = cohort.metrics(n_dims)
    return result


class ReconstructionTotals:
    """Running sums of squared error and squared distance from the mean, for MSE and variance explained."""

    def __init__(self):
        self.n = 0
        self.squared_error = 0.0
        self.squared_from_mean = 0.0

    def add(self, squared_error: torch.Tensor, squared_from_mean: torch.Tensor) -> None:
        """Add per-example values for one batch."""
        self.n += len(squared_error)
        self.squared_error += squared_error.sum().item()
        self.squared_from_mean += squared_from_mean.sum().item()

    def metrics(self, n_dims: int) -> dict[str, float | None]:
        """{'n', 'mse' (per activation dimension), 'variance_explained' (None if no spread)}."""
        variance_explained = (1 - self.squared_error / self.squared_from_mean
                              if self.squared_from_mean > 0 else None)
        return {'n': self.n, 'mse': self.squared_error / (self.n * n_dims),
                'variance_explained': variance_explained}


@torch.no_grad()
def with_bias_only_fix(model: JumpReLUSAE, train_split: Split, batch_size: int,
                       device: str | torch.device) -> JumpReLUSAE:
    """A copy of `model` whose decoder bias absorbs the mean train reconstruction error.

    This is the cheapest possible adaptation: one constant shift, fitted on train only. If fine-tuning
    barely beats it, most of the gain is just re-centring. `model` itself is not changed.
    """
    total_residual = torch.zeros(model.input_dim, dtype=torch.float64, device=device)
    for rows, batch in iterate_batches(train_split.activations, batch_size, device):
        reconstruction = model.decoder(features_for(model, train_split, rows, batch))
        total_residual += (batch - reconstruction).double().sum(dim=0)
    shifted = copy.deepcopy(model)
    shifted.b_dec.add_((total_residual / len(train_split.activations)).to(shifted.b_dec.dtype))
    return shifted
