"""Matched edits reuse selections and apply equal relative activation changes."""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from mas_sae.probe import feature_selection as fs, matched_edits as me


def test_scores_equal_explicit_error_preserving_norm_matched_edits():
    rng = np.random.default_rng(4)
    X, delta, w = rng.normal(size=(7, 4)), rng.normal(size=(7, 4)), rng.normal(size=4)
    shift, norm, scale, valid = me.direction_scores(delta, X, w)
    for dose in (0.01, 0.1):
        edit = delta * (dose * scale[:, None])
        assert valid.all()
        assert np.allclose(np.linalg.norm(edit, axis=1), dose * np.linalg.norm(X, axis=1))
        assert np.allclose(expit((X + edit) @ w + 0.3), expit(X @ w + 0.3 + dose * shift))
    assert np.allclose(norm, np.linalg.norm(delta, axis=1))


def test_zero_directions_are_flagged_instead_of_fabricating_a_matched_edit():
    shift, _, scale, valid = me.direction_scores(np.array([[0., 0.], [1., 0.]]),
                                                np.array([[3., 4.], [0., 0.]]), np.ones(2))
    assert not valid.any()
    assert np.isnan(shift).all()
    assert (scale == 0).all()


def test_changed_inputs_refuse_reusing_selection():
    cfg = fs.FeatureSelectionConfig('.', 'acts', 'results', 'labels', 'manifest')
    with pytest.raises(ValueError, match='labels_sha256'):
        me.validate_source({'provenance': {'labels_sha256': 'old'}}, cfg, {'labels_sha256': 'new'})


def test_rerun_reuses_selection_and_persists_readout_norms_and_predictions(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    selected = pd.DataFrame({'target_name': np.where(X[:, 0] > 0, 'adoption', 'retention'),
                             'partition': ['train'] * 40 + ['validation'] * 20,
                             'ep_question_id': [f'q{i}' for i in range(60)]})
    np.save(tmp_path / 'center.npy', X[:40].mean(0))
    cfg = fs.FeatureSelectionConfig(tmp_path, 'acts', 'results', 'labels', 'manifest',
              target_labels=['adoption', 'retention'], label_column='behavior_label',
              saes={'Raw': {'kind': 'activations', 'path': 'center.npy'}},
              output_dir=tmp_path / 'out', n_random_draws=2, n_bootstrap=10, n_permutations=19)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'feature_selection_summary.json').write_text(json.dumps({
        'config': asdict(cfg), 'provenance': {},
        'readout': {'n_fit': 40, 'n_eval': 20, 'chosen_C': 0.1},
        'saes': {'Raw': {'targeted_features': [0], 'random_control_draws': [[1], [2]]}}}, default=str))
    monkeypatch.setattr(fs, 'provenance', lambda *args: {})
    monkeypatch.setattr(me.ls, 'load_inputs', lambda *args: (selected, {}, me.bd.Columns()))
    monkeypatch.setattr(me.bd, 'load_layer_matrix', lambda *args: X)
    # Family-clustered inference: two questions per family, without crossing partitions.
    monkeypatch.setattr(me.bd, 'episode_groups', lambda *args: (np.arange(60) // 2, 'family'))
    def forbidden(*args, **kwargs):
        pytest.fail('Rerun must not train feature probes, reselect features, or retune readout C')
    monkeypatch.setattr(fs, 'rank_features', forbidden)
    monkeypatch.setattr(me.offline, 'fit_readout', forbidden)
    out = me.run(cfg, source, doses=[0.01, 0.05])
    summary = json.loads((out / 'matched_edit_summary.json').read_text())
    assert summary['readout']['chosen_C'] == 0.1
    assert (out / 'readout.npz').exists()
    assert (out / 'predictions_raw_controls.npz').exists()
    norms = pd.read_csv(out / 'edit_norms.csv')
    assert norms.valid.all()
    assert np.allclose(norms['applied_norm_at_0.05'], 0.05 * norms.activation_norm)
    saved = np.load(out / 'predictions_Raw.npz')
    assert saved['random'].shape == (2, 20, 2)
    assert summary['targeted_vs_random']['Raw'][0]['n_valid'] == 20
    assert 'ci95_pp_questions_and_draws' in summary['targeted_vs_random']['Raw'][0]


def test_invalid_doses_fail_before_loading_source(tmp_path):
    for doses in ([], [0], [-0.1], [float('nan')]):
        with pytest.raises(ValueError, match='Doses'):
            me.run(None, tmp_path, doses)
