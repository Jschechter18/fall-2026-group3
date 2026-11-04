"""SAE health and stability diagnostics, beyond reconstruction error.

    stats      streaming accumulators (FeatureStats, PairedStats, RunningCorrelation) and quantiles()
    health     utilization, usage concentration and activation magnitudes of one model on one split
    geometry   weights only: decoder norms, near-duplicate features, encoder-decoder alignment, thresholds
    stability  adapted vs original: frequency change, Jaccard, correlation, died/revived, split/merge

Callers feed feature batches in, so a frozen encoder's cache can stand in for re-encoding.
"""

from .geometry import parameter_health
from .health import activation_health, compact_health
from .stability import compact_stability, split_merge, stability
from .stats import FeatureStats, PairedStats, RunningCorrelation, quantiles

__all__ = ['FeatureStats', 'PairedStats', 'RunningCorrelation', 'activation_health', 'compact_health',
           'compact_stability', 'parameter_health', 'quantiles', 'split_merge', 'stability']
