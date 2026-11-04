"""Dictionary health and stability for fine-tuning runs: mas_sae.sae.diagnostics applied to Splits."""

from dataclasses import dataclass

import numpy as np
import torch

from mas_sae.sae import diagnostics

from .data import features_for, iterate_batches


@dataclass
class Health:
    """dictionary_health() output: the JSON summary, and per-feature arrays for feature_health.npz."""
    summary: dict
    per_feature: dict


def dictionary_health(model, splits, batch_size, device):
    """Utilization, concentration and activation health per split, plus weight geometry.

    The near-duplicate search is also run among features alive on train (the larger split).
    Per-feature arrays are taken from validation, the split stability is measured on.
    """
    stats = {name: feature_stats(model, split, batch_size, device) for name, split in splits.items()}
    activation = {name: diagnostics.activation_health(s) for name, s in stats.items()}
    parameters, parameter_arrays = diagnostics.parameter_health(model, alive=stats['train'].alive())
    summary = {name: report for name, (report, _) in activation.items()}
    summary['parameters'] = parameters
    return Health(summary, {**activation['validation'][1], **parameter_arrays})


def stability_report(source, model, split, batch_size, device):
    """Adapted vs original on one split, including split/merge detection (a second pass)."""
    paired = paired_stats(source, model, split, batch_size, device)
    summary, per_feature = diagnostics.stability(paired, source, model)
    summary['split_merge'], split_merge_arrays = diagnostics.split_merge(
        lambda: paired_batches(source, model, split, batch_size, device), paired)
    return summary, {**per_feature, **split_merge_arrays}


def save_feature_arrays(path, original_health, adapted_health, stability_per_feature):
    arrays = {**{f'original_{k}': v for k, v in original_health.per_feature.items()},
              **{f'adapted_{k}': v for k, v in adapted_health.per_feature.items()},
              **stability_per_feature}
    np.savez(path, **{name: value.detach().cpu().numpy() for name, value in arrays.items()})


@torch.no_grad()
def feature_stats(model, split, batch_size, device):
    stats = diagnostics.FeatureStats(model.latent_dim, device)
    for rows, batch in iterate_batches(split.activations, batch_size, device):
        stats.add(features_for(model, split, rows, batch))
    return stats


def paired_batches(source, model, split, batch_size, device):
    """Yield (original features, current features) for every batch of a split."""
    with torch.no_grad():
        for rows, batch in iterate_batches(split.activations, batch_size, device):
            yield source.encoder(batch), features_for(model, split, rows, batch)


def paired_stats(source, model, split, batch_size, device):
    paired = diagnostics.PairedStats(model.latent_dim, device)
    for original_features, features in paired_batches(source, model, split, batch_size, device):
        paired.add(original_features, features)
    return paired
