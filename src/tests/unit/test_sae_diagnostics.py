"""SAE health and stability diagnostics on tiny hand-built SAEs and feature matrices."""
import math

import pytest
import torch

from mas_sae.sae import diagnostics
from mas_sae.sae.diagnostics.health import entropy_and_effective_size, top_share
from mas_sae.sae.pretrained import JumpReLUSAE


def stats_of(features):
    stats = diagnostics.FeatureStats(features.shape[1], 'cpu')
    for batch in features.split(2):
        stats.add(batch)
    return stats


def paired_of(a, b):
    paired = diagnostics.PairedStats(a.shape[1], 'cpu')
    for x, y in zip(a.split(2), b.split(2)):
        paired.add(x, y)
    return paired


def test_utilization_counts_firing_frequencies():
    # Feature 0 fires on every example, feature 1 on half, feature 2 on one in four, feature 3 never.
    features = torch.tensor([[1., 2., 3., 0.], [1., 2., 0., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]])
    summary, per_feature = diagnostics.activation_health(stats_of(features))
    assert per_feature['frequency'].tolist() == [1.0, 0.5, 0.25, 0.0]
    utilization = summary['utilization']
    assert utilization['alive_count'] == 3 and utilization['alive_fraction'] == 0.75
    assert utilization['fraction_frequency_at_least_0.01'] == 0.75
    assert utilization['frequency_all']['max'] == 1.0
    assert summary['activations']['dense_over_50pct_count'] == 1
    assert summary['activations']['dense_over_90pct_count'] == 1
    assert per_feature['mean_magnitude'][:3].tolist() == [1.0, 2.0, 3.0]
    assert math.isnan(per_feature['mean_magnitude'][3])
    # Nonzero values are 1 (x4), 2 (x2) and 3 (x1): the median is 1, up to histogram resolution.
    assert summary['activations']['nonzero_magnitude']['p50'] == pytest.approx(1.0, rel=0.03)


def test_effective_size_is_n_for_uniform_usage_and_one_for_a_monopoly():
    assert entropy_and_effective_size(torch.ones(8, dtype=torch.float64))[1] == pytest.approx(8)
    assert entropy_and_effective_size(torch.tensor([5., 0., 0.], dtype=torch.float64))[1] == pytest.approx(1)
    mass = torch.tensor([97.] + [1.] * 99, dtype=torch.float64)
    assert top_share(mass, 0.01) == pytest.approx(97 / 196)


def test_geometry_finds_near_duplicates_and_alignment():
    w_dec = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    sae = JumpReLUSAE(w_dec.T.clone(), torch.zeros(3), w_dec, torch.zeros(3), torch.ones(3))
    summary, per_feature = diagnostics.parameter_health(sae, alive=torch.tensor([True, False, True]))
    assert per_feature['nearest_neighbour_abs_cosine'].tolist() == [1.0, 1.0, 0.0]
    assert summary['nearest_neighbour_abs_cosine']['count_over_0.95'] == 2
    # Among alive features only (0 and 2) there is no duplicate.
    assert summary['nearest_neighbour_abs_cosine_alive']['count_over_0.9'] == 0
    assert summary['mean_pairwise_abs_cosine'] == pytest.approx(2 / 6)
    assert torch.allclose(per_feature['encoder_decoder_alignment'], torch.ones(3))


def test_identical_models_are_perfectly_stable():
    features = torch.tensor([[1., 0., 2.], [0., 3., 1.], [2., 1., 0.], [0., 0., 4.]])
    paired = paired_of(features, features)
    summary, per_feature = diagnostics.stability(paired)
    assert summary['active_set_jaccard']['mean'] == 1.0
    assert summary['activation_correlation']['p1'] == pytest.approx(1.0)
    assert summary['lifecycle'] == {'alive_original': 3, 'alive_adapted': 3, 'alive_relative_change': 0.0,
                                    'died': 0, 'revived': 0}
    split_merge, _ = diagnostics.split_merge(lambda: zip(features.split(2), features.split(2)), paired)
    assert split_merge['split_count'] == 0 and split_merge['merged_count'] == 0
    compact = diagnostics.compact_stability(paired)
    assert compact['activation_correlation_median'] == pytest.approx(1.0) and compact['died'] == 0


def test_a_silenced_feature_counts_as_died_and_lowers_jaccard():
    original = torch.tensor([[1., 1.], [1., 0.], [0., 1.], [1., 1.]])
    adapted = original.clone()
    adapted[:, 1] = 0
    summary, per_feature = diagnostics.stability(paired_of(original, adapted))
    assert summary['lifecycle']['died'] == 1 and per_feature['died'].tolist() == [False, True]
    assert summary['active_set_jaccard']['mean'] == pytest.approx((0.5 + 1 + 0 + 0.5) / 4)


def test_split_and_merge_are_detected():
    torch.manual_seed(0)
    signal = torch.rand(40, 1) + 0.5
    noise = torch.rand(40, 1) + 0.5
    # Original: feature 0 = signal, feature 1 = noise, feature 2 dead.
    original = torch.cat([signal, noise, torch.zeros(40, 1)], dim=1)
    # Adapted 0 blends signal and noise (corr ~0.7 with each), so it is no longer intact and links to
    # original 0 and 1: a merge. Adapted 2 is a copy of the signal, so original 0 links to adapted 0
    # and 2: a split.
    adapted = torch.cat([signal + noise, noise, signal], dim=1)
    paired = paired_of(original, adapted)
    summary, arrays = diagnostics.split_merge(lambda: zip(original.split(8), adapted.split(8)), paired)
    assert arrays['split'][0] and summary['split_count'] >= 1
    assert arrays['merged'][0] and summary['merged_count'] >= 1


def test_weight_drift_is_reported_when_models_are_given():
    sae = JumpReLUSAE(torch.eye(3), torch.zeros(3), torch.eye(3), torch.zeros(3), torch.ones(3))
    moved = JumpReLUSAE(torch.eye(3), torch.zeros(3), torch.eye(3), torch.zeros(3), torch.ones(3) * 2)
    features = torch.ones(4, 3)
    summary, per_feature = diagnostics.stability(paired_of(features, features), sae, moved)
    assert summary['threshold_ratio']['p50'] == pytest.approx(2.0)
    assert summary['decoder_cosine']['min'] == pytest.approx(1.0)
