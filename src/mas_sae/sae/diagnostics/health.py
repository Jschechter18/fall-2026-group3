"""Dictionary utilization, usage concentration and activation health, from one model's FeatureStats."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .stats import quantiles

if TYPE_CHECKING:
    import torch

    from .stats import FeatureStats

FREQUENCY_QUANTILES = (0.5, 0.9, 0.99)
FREQUENCY_FLOORS = (1e-4, 1e-3, 1e-2)        # "used in at least 0.01%, 0.1%, 1% of examples"
TOP_SHARES = (0.01, 0.05, 0.1)               # mass held by the top 1/5/10% of all features
MAGNITUDE_QUANTILES = (0.1, 0.5, 0.9, 0.99)
DENSE_FREQUENCIES = (0.5, 0.9)               # "always active": fires on more than this share of examples


def entropy_and_effective_size(mass: torch.Tensor) -> tuple[float | None, float | None]:
    """Shannon entropy (nats) of `mass` normalised to a distribution, and exp(entropy)."""
    total = mass.sum()
    if total <= 0:
        return None, None
    p = mass[mass > 0] / total
    entropy = -(p * p.log()).sum().item()
    return entropy, math.exp(entropy)


def top_share(mass: torch.Tensor, fraction: float) -> float | None:
    """Share of total mass held by the top `fraction` of all features (at least one feature)."""
    total = mass.sum()
    if total <= 0:
        return None
    return (mass.topk(max(1, math.ceil(fraction * len(mass)))).values.sum() / total).item()


def utilization(stats: FeatureStats) -> dict:
    """How much of the dictionary is used: alive count and fraction, the fraction of features with
    firing frequency >= each FREQUENCY_FLOORS value, and frequency quantiles (all and alive)."""
    frequency, alive = stats.frequency(), stats.alive()
    return {'alive_count': int(alive.sum()),
            'alive_fraction': alive.double().mean().item(),
            **{f'fraction_frequency_at_least_{floor:g}': (frequency >= floor).double().mean().item()
               for floor in FREQUENCY_FLOORS},
            'frequency_all': {**quantiles(frequency, FREQUENCY_QUANTILES), 'max': frequency.max().item()},
            'frequency_alive': quantiles(frequency[alive], FREQUENCY_QUANTILES)}


def concentration(stats: FeatureStats) -> dict[str, float | None]:
    """Is a small subset monopolizing the representation? Mass = summed activation values."""
    mass_entropy, mass_size = entropy_and_effective_size(stats.mass)
    count_entropy, count_size = entropy_and_effective_size(stats.count)
    return {'activation_mass_entropy': mass_entropy, 'activation_mass_effective_size': mass_size,
            'firing_count_entropy': count_entropy, 'firing_count_effective_size': count_size,
            **{f'top_{fraction * 100:g}pct_mass_share': top_share(stats.mass, fraction) for fraction in TOP_SHARES}}


def activations(stats: FeatureStats) -> dict:
    """Activation magnitudes (mean and approximate quantiles of nonzero values, quantiles of
    per-feature means) and counts of near-always-active features (DENSE_FREQUENCIES)."""
    frequency = stats.frequency()
    fired = stats.count.sum()
    return {'mean_nonzero_magnitude': (stats.mass.sum() / fired).item() if fired > 0 else None,
            'nonzero_magnitude': stats.magnitude_quantiles(MAGNITUDE_QUANTILES),
            'per_feature_mean_magnitude': quantiles(stats.mean_magnitude(), MAGNITUDE_QUANTILES),
            **{f'dense_over_{share * 100:g}pct_count': int((frequency > share).sum()) for share in DENSE_FREQUENCIES}}


def activation_health(stats: FeatureStats) -> tuple[dict, dict[str, torch.Tensor]]:
    """Returns (summary dict, per-feature arrays) for one model on one split."""
    summary = {'n_examples': stats.n, 'utilization': utilization(stats),
               'concentration': concentration(stats), 'activations': activations(stats)}
    per_feature = {'frequency': stats.frequency(), 'mean_magnitude': stats.mean_magnitude(),
                   'max_activation': stats.max}
    return summary, per_feature


def compact_health(stats: FeatureStats) -> dict[str, float | None]:
    """A few activation_health() numbers, cheap enough to log every epoch."""
    return {'alive_fraction': stats.alive().double().mean().item(),
            'activation_mass_effective_size': entropy_and_effective_size(stats.mass)[1],
            'top_1pct_mass_share': top_share(stats.mass, 0.01),
            'dense_over_50pct_count': int((stats.frequency() > 0.5).sum())}
