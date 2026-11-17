"""How far an adapted SAE moved from the original it was fine-tuned from.

Features are matched by index: fine-tuning starts from the original weights, so adapted feature i
descends from original feature i.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from typing import TYPE_CHECKING

import torch
from torch.nn.functional import cosine_similarity

from .stats import quantiles

if TYPE_CHECKING:
    from mas_sae.sae.pretrained import JumpReLUSAE

    from .stats import PairedStats

TAIL_QUANTILES = (0.01, 0.05, 0.1, 0.5)      # for stability the bad tail matters more than the mean
RATIO_QUANTILES = (0.01, 0.1, 0.5, 0.9, 0.99)
FREQUENCY_CHANGE_FACTOR = 2.0                # count features whose firing rate moved by more than 2x
LOW_CORRELATION = 0.9
LINK_CORRELATION = 0.5                       # split/merge: features this correlated count as linked
SAME_INDEX_INTACT = 0.9                      # split/merge: a feature this correlated with its own index is intact


def lifecycle(paired: PairedStats) -> dict[str, float | None]:
    """Alive-feature counts before and after, their relative change (None if none was alive before),
    and how many features died (alive only in the original) or revived (alive only after)."""
    alive_a, alive_b = paired.original.alive(), paired.adapted.alive()
    before, after = int(alive_a.sum()), int(alive_b.sum())
    return {'alive_original': before, 'alive_adapted': after,
            'alive_relative_change': (after - before) / before if before else None,
            'died': int((alive_a & ~alive_b).sum()), 'revived': int((~alive_a & alive_b).sum())}


def frequency_log2_ratio(paired: PairedStats) -> torch.Tensor:
    """Per feature: log2(adapted / original firing frequency); nan unless alive in both."""
    freq_a, freq_b = paired.original.frequency(), paired.adapted.frequency()
    both = paired.original.alive() & paired.adapted.alive()
    ratio = torch.full_like(freq_a, float('nan'))
    ratio[both] = (freq_b[both] / freq_a[both]).log2()
    return ratio


def weight_drift(original: JumpReLUSAE,
                 adapted: JumpReLUSAE) -> tuple[dict, dict[str, torch.Tensor]]:
    """Per-feature encoder/decoder cosines and threshold ratios. Returns (summary, per-feature arrays)."""
    encoder = cosine_similarity(original.w_enc.double(), adapted.w_enc.double(), dim=0)   # (d, n): columns
    decoder = cosine_similarity(original.w_dec.double(), adapted.w_dec.double(), dim=1)   # (n, d): rows
    ratio = adapted.threshold.double() / original.threshold.double()
    summary = {'encoder_cosine': {**quantiles(encoder, (0.01, 0.5)), 'min': encoder.min().item()},
               'decoder_cosine': {**quantiles(decoder, (0.01, 0.5)), 'min': decoder.min().item()},
               'threshold_ratio': quantiles(ratio, RATIO_QUANTILES)}
    return summary, {'encoder_cosine': encoder, 'decoder_cosine': decoder, 'threshold_ratio': ratio}


def stability(paired: PairedStats, original: JumpReLUSAE | None = None,
              adapted: JumpReLUSAE | None = None) -> tuple[dict, dict[str, torch.Tensor]]:
    """Returns (summary, per-feature arrays). Weight drift is added when both models are given."""
    correlation, defined = paired.correlation.result()
    corr = correlation[defined]
    jaccard = paired.jaccard()
    log2_ratio = frequency_log2_ratio(paired)
    summary = {
        'n_examples': paired.original.n,
        'active_set_jaccard': {'mean': jaccard.mean().item() if len(jaccard) else None,
                               **quantiles(jaccard, TAIL_QUANTILES)},
        'activation_correlation': {
            'n_features_defined': int(defined.sum()), **quantiles(corr, TAIL_QUANTILES),
            f'fraction_below_{LOW_CORRELATION:g}': (corr < LOW_CORRELATION).double().mean().item() if len(corr) else None},
        'firing_frequency_change': {
            'log2_ratio': quantiles(log2_ratio, RATIO_QUANTILES),
            f'count_changed_over_{FREQUENCY_CHANGE_FACTOR:g}x':
                int((log2_ratio.abs() > math.log2(FREQUENCY_CHANGE_FACTOR)).sum())},
        'lifecycle': lifecycle(paired),
    }
    alive_a, alive_b = paired.original.alive(), paired.adapted.alive()
    per_feature = {'activation_correlation': correlation, 'frequency_log2_ratio': log2_ratio,
                   'died': alive_a & ~alive_b, 'revived': ~alive_a & alive_b}
    if original is not None and adapted is not None:
        drift, drift_per_feature = weight_drift(original, adapted)
        summary.update(drift)
        per_feature.update(drift_per_feature)
    return summary, per_feature


def compact_stability(paired: PairedStats) -> dict[str, float | None]:
    """A few stability() numbers, cheap enough to log, and gate checkpoints on, every epoch."""
    correlation, defined = paired.correlation.result()
    jaccard = paired.jaccard()
    return {'activation_correlation_median': quantiles(correlation[defined], (0.5,))['p50'],
            'active_set_jaccard_mean': jaccard.mean().item() if len(jaccard) else None,
            'active_set_jaccard_p10': quantiles(jaccard, (0.1,))['p10'],
            **lifecycle(paired)}


class CrossCorrelation:
    """Pearson correlation between every alive original feature and every alive adapted feature.

    Restricted to alive features (a few thousand at most on our data), so the matrix stays small.
    """

    def __init__(self, alive_original: torch.Tensor, alive_adapted: torch.Tensor):
        self.index_a, self.index_b = alive_original.nonzero().flatten(), alive_adapted.nonzero().flatten()
        device = alive_original.device
        self.n = 0
        self.sum_a, self.sum_aa = (torch.zeros(len(self.index_a), dtype=torch.float64, device=device) for _ in range(2))
        self.sum_b, self.sum_bb = (torch.zeros(len(self.index_b), dtype=torch.float64, device=device) for _ in range(2))
        self.sum_ab = torch.zeros(len(self.index_a), len(self.index_b), dtype=torch.float64, device=device)

    def add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """Accumulate one batch of full-width original (a) and adapted (b) features."""
        a, b = a[:, self.index_a].double(), b[:, self.index_b].double()
        self.n += len(a)
        self.sum_a += a.sum(0)
        self.sum_aa += a.square().sum(0)
        self.sum_b += b.sum(0)
        self.sum_bb += b.square().sum(0)
        self.sum_ab += a.T @ b

    def result(self) -> torch.Tensor:
        """(alive original x alive adapted) correlations, rows/columns ordered as index_a/index_b.

        Pairs where either feature never varies get 0, not nan.
        """
        mean_a, mean_b = self.sum_a / self.n, self.sum_b / self.n
        std_a = (self.sum_aa / self.n - mean_a ** 2).clamp(min=0).sqrt()
        std_b = (self.sum_bb / self.n - mean_b ** 2).clamp(min=0).sqrt()
        covariance = self.sum_ab / self.n - mean_a[:, None] * mean_b[None, :]
        denominator = std_a[:, None] * std_b[None, :]
        return torch.where(denominator > 0, covariance / denominator.clamp(min=1e-300), torch.zeros_like(covariance))


@torch.no_grad()
def split_merge(pairs: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
                paired: PairedStats) -> tuple[dict, dict[str, torch.Tensor]]:
    """Features that split into several, or merged into one. Returns (summary, per-feature bool arrays).

    `pairs()` yields (original features, adapted features) batches over the same examples as `paired`
    (a finished PairedStats, which supplies the alive masks and same-index correlations).

      split   original feature i is no longer intact (corr with adapted i < SAME_INDEX_INTACT, or adapted
              i dead) and at least two adapted features correlate >= LINK_CORRELATION with it.
      merge   adapted feature j is no longer intact in the same sense, and at least two original features
              correlate >= LINK_CORRELATION with it.
    Requiring "no longer intact" skips pairs that were already correlated in the original SAE, so the
    original compared with itself has no splits or merges.
    """
    alive_a, alive_b = paired.original.alive(), paired.adapted.alive()
    split, merged = torch.zeros_like(alive_a), torch.zeros_like(alive_a)
    summary = {'split_count': 0, 'merged_count': 0,
               'link_correlation': LINK_CORRELATION, 'same_index_intact': SAME_INDEX_INTACT}
    if not alive_a.any() or not alive_b.any():
        return summary, {'split': split, 'merged': merged}
    cross = CrossCorrelation(alive_a, alive_b)
    for a, b in pairs():
        cross.add(a, b)
    linked = cross.result() >= LINK_CORRELATION
    same_index, _ = paired.correlation.result()
    intact = torch.nan_to_num(same_index, nan=0.0) >= SAME_INDEX_INTACT
    split[cross.index_a] = (linked.sum(dim=1) >= 2) & ~intact[cross.index_a]
    merged[cross.index_b] = (linked.sum(dim=0) >= 2) & ~intact[cross.index_b]
    summary.update(split_count=int(split.sum()), merged_count=int(merged.sum()))
    return summary, {'split': split, 'merged': merged}
