"""Streaming accumulators: per-feature sums over batches, so no (examples x features) matrix is ever kept."""

import torch

# Log10 histogram of nonzero activation magnitudes, for approximate quantiles (0.01-decade bins).
LOG10_RANGE = (-6.0, 6.0)
HISTOGRAM_BINS = 1200


def label(q):
    """0.5 -> 'p50', 0.01 -> 'p1'."""
    return f'p{q * 100:g}'


def quantiles(values, qs):
    """{'p50': ..., ...} for a tensor, ignoring nan. Values are None when nothing is left."""
    values = values.detach().double().flatten().cpu()
    values = values[~values.isnan()]
    if len(values) == 0:
        return {label(q): None for q in qs}
    result = torch.quantile(values, torch.tensor(qs, dtype=torch.float64))
    return {label(q): v.item() for q, v in zip(qs, result)}


def _zeros(n, device):
    return torch.zeros(n, dtype=torch.float64, device=device)


class FeatureStats:
    """One model's features on one split: firing counts, activation mass and a magnitude histogram."""

    def __init__(self, n_features, device):
        self.n = 0
        self.count, self.mass, self.max = _zeros(n_features, device), _zeros(n_features, device), _zeros(n_features, device)
        self.histogram = _zeros(HISTOGRAM_BINS, device)

    def add(self, features):
        active = features > 0
        values = features.double()
        self.n += len(features)
        self.count += active.sum(dim=0)
        self.mass += values.sum(dim=0)
        if len(features):
            self.max = torch.maximum(self.max, values.max(dim=0).values)
        nonzero = features[active].float()
        if len(nonzero):
            logs = nonzero.log10().clamp(*LOG10_RANGE)
            self.histogram += torch.histc(logs, bins=HISTOGRAM_BINS, min=LOG10_RANGE[0], max=LOG10_RANGE[1]).double()

    def frequency(self):
        """Per feature: share of examples on which it fires."""
        return self.count / max(self.n, 1)

    def alive(self):
        """Per feature: fired at least once."""
        return self.count > 0

    def mean_magnitude(self):
        """Per feature: mean value when active (nan for a dead feature)."""
        return torch.where(self.alive(), self.mass / self.count.clamp(min=1), torch.full_like(self.mass, float('nan')))

    def magnitude_quantiles(self, qs):
        """Approximate quantiles of every nonzero activation value, read off the log10 histogram."""
        total = self.histogram.sum()
        if total == 0:
            return {label(q): None for q in qs}
        cumulative = (self.histogram.cumsum(0) / total).cpu()
        width = (LOG10_RANGE[1] - LOG10_RANGE[0]) / HISTOGRAM_BINS
        result = {}
        for q in qs:
            index = min(int(torch.searchsorted(cumulative, torch.tensor(q, dtype=torch.float64))), HISTOGRAM_BINS - 1)
            result[label(q)] = 10 ** (LOG10_RANGE[0] + (index + 0.5) * width)
        return result


class RunningCorrelation:
    """Per-feature Pearson correlation between two feature matrices, accumulated batch by batch.

        corr = (E[ab] - E[a] E[b]) / sqrt(Var a * Var b),   where Var a = E[a^2] - E[a]^2

    Only five running sums per feature are kept, so neither full matrix is ever held in memory.
    """

    def __init__(self, n_features, device):
        self.n = 0
        self.sum_a, self.sum_b, self.sum_aa, self.sum_bb, self.sum_ab = (_zeros(n_features, device) for _ in range(5))

    def add(self, a, b):
        a, b = a.double(), b.double()
        self.n += len(a)
        self.sum_a += a.sum(dim=0)
        self.sum_b += b.sum(dim=0)
        self.sum_aa += (a * a).sum(dim=0)
        self.sum_bb += (b * b).sum(dim=0)
        self.sum_ab += (a * b).sum(dim=0)

    def result(self):
        """Returns (correlation, defined). Correlation is nan where either side never varies, e.g. a dead feature."""
        mean_a, mean_b = self.sum_a / self.n, self.sum_b / self.n
        var_a = self.sum_aa / self.n - mean_a ** 2
        var_b = self.sum_bb / self.n - mean_b ** 2
        covariance = self.sum_ab / self.n - mean_a * mean_b
        defined = (var_a > 0) & (var_b > 0)
        correlation = torch.full_like(var_a, float('nan'))
        correlation[defined] = (covariance[defined] / (var_a * var_b)[defined].sqrt()).clamp(-1, 1)
        return correlation, defined


class PairedStats:
    """The original (a) and adapted (b) SAE's features on the same examples, side by side."""

    def __init__(self, n_features, device):
        self.original = FeatureStats(n_features, device)
        self.adapted = FeatureStats(n_features, device)
        self.correlation = RunningCorrelation(n_features, device)
        self._jaccard = []

    def add(self, a, b):
        self.original.add(a)
        self.adapted.add(b)
        self.correlation.add(a, b)
        active_a, active_b = a > 0, b > 0
        either = (active_a | active_b).sum(dim=1)
        both = (active_a & active_b).sum(dim=1).double()
        # An example where neither model fires anything counts as perfect agreement.
        self._jaccard.append(torch.where(either > 0, both / either.clamp(min=1), torch.ones_like(both)).cpu())

    def jaccard(self):
        """Per example: |both fire| / |either fires|."""
        return torch.cat(self._jaccard) if self._jaccard else torch.empty(0, dtype=torch.float64)
