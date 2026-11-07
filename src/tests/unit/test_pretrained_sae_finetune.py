"""Pretrained SAE fine-tuning: decoder-only mode keeps features and reuses disk caches; full mode trains the encoder."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mas_sae.experiments.artifacts import sha256_file
from mas_sae.sae.finetuning import cache, data, evaluation, loop, selection
from mas_sae.sae.pretrained import JumpReLUSAE, load_gemma_scope_sae
from repo_paths import REPO_ROOT

spec = importlib.util.spec_from_file_location('pretrained_sae_finetune',
    REPO_ROOT / 'scripts/sae/finetune_pretrained_sae.py')
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


# The script is loaded by path only for run(); its helpers are imported from mas_sae.sae.finetuning.
# The full-mode options at their defaults: decoder-only behaviour is unchanged. Stability gates are off,
# so these tiny problems may drift freely; test_stability_gate_* covers the gates.
NO_STABILITY_GATES = dict(min_activation_correlation=0.0, min_jaccard=0.0, max_alive_change=float('inf'))
DECODER_ONLY = dict(train='decoder', lr_enc=None, l0_coef=1.0, l0_tolerance=0.1, bandwidth=None,
                    freeze_threshold=False, **NO_STABILITY_GATES)


def model():
    return JumpReLUSAE(torch.eye(3), torch.zeros(3), torch.zeros(3, 3),
                      torch.zeros(3), torch.ones(3) * 0.01)


def test_cache_hits_without_reencoding_and_ignores_decoder_changes(tmp_path, monkeypatch):
    sae = model()
    X = torch.ones(5, 3)
    Z, path = cache.cached_features(sae, X, {'split': 'train', 'activation_sha256': 'a'},
                                 tmp_path, 2, 'cpu')
    assert isinstance(Z, np.memmap)
    assert np.array_equal(Z, np.ones((5, 3)))
    with torch.no_grad():
        sae.w_dec.add_(1)
    def forbidden(*args):
        pytest.fail('Cache hit must not invoke the encoder')
    monkeypatch.setattr(sae, 'encoder', forbidden)
    reused, hit = cache.cached_features(sae, X, {'split': 'train', 'activation_sha256': 'a'},
                                     tmp_path, 3, 'cpu')
    assert hit == path and np.array_equal(reused, Z)


def test_input_or_encoder_changes_make_a_new_cache(tmp_path):
    sae, X = model(), torch.ones(3, 3)
    _, first = cache.cached_features(sae, X, {'split': 'train', 'activation_sha256': 'a'}, tmp_path, 2, 'cpu')
    _, changed_input = cache.cached_features(sae, X * 2, {'split': 'train', 'activation_sha256': 'b'}, tmp_path, 2, 'cpu')
    with torch.no_grad():
        sae.threshold.fill_(2)
    Z, changed_encoder = cache.cached_features(sae, X, {'split': 'train', 'activation_sha256': 'a'}, tmp_path, 2, 'cpu')
    assert len({first, changed_input, changed_encoder}) == 3
    assert not np.any(Z)


def test_failed_cache_build_is_not_reused(tmp_path, monkeypatch):
    sae = model()
    monkeypatch.setattr(sae, 'encoder', lambda x: x * float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        cache.cached_features(sae, torch.ones(3, 3), {'split': 'train'}, tmp_path, 2, 'cpu')
    assert not list(tmp_path.iterdir())


def test_end_to_end_improves_reconstruction_preserves_features_and_never_reads_test(tmp_path, monkeypatch):
    src, acts = tmp_path / 'source', tmp_path / 'acts'
    src.mkdir()
    acts.mkdir()
    sae = model()
    config = {'architecture': 'jump_relu', 'hf_hook_point_in': 'model.layers.25.output'}
    loop.save_model(sae, config, src)
    original_hash = sha256_file(src / 'params.safetensors')
    generator = torch.Generator().manual_seed(3)
    X = torch.rand(80, 3, generator=generator) * 3
    torch.save(X[:60], acts / 'train_attempt2.pt')
    torch.save(X[60:], acts / 'validation_attempt2.pt')
    # A malformed sealed file would fail if opened.
    (acts / 'test_attempt2.pt').write_text('must not be read')
    args = argparse.Namespace(sae_dir=src, activations_dir=acts, layer=25,
        activation_file='{split}_attempt2.pt', cache_dir=tmp_path / 'cache',
        output_dir=tmp_path / 'out', probe_config=None, epochs=8, batch_size=15,
        lr=0.05, patience=3, seed=42, device='cpu', **DECODER_ONLY)
    out = ft.run(args)
    summary = json.loads((out / 'summary.json').read_text())
    adapted, _ = load_gemma_scope_sae(out / 'best', expected_layer=25)
    assert summary['adapted']['validation']['mse'] < summary['original']['validation']['mse']
    assert summary['best_epoch'] > 0
    for name in cache.ENCODER_KEYS:
        assert torch.equal(getattr(sae, name), getattr(adapted, name))
    assert torch.equal(sae.encoder(X), adapted.encoder(X))
    assert not torch.equal(sae.w_dec, adapted.w_dec)
    assert not torch.equal(sae.b_dec, adapted.b_dec)
    assert sha256_file(src / 'params.safetensors') == original_hash
    assert json.loads((out / 'manifest.json').read_text())['status'] == 'completed'
    assert len(list(args.cache_dir.glob('*/manifest.json'))) == 2
    # Frozen encoder: features are identical to the original's, so stability is perfect.
    stability = summary['stability']
    assert stability['active_set_jaccard']['mean'] == 1.0
    assert stability['lifecycle']['died'] == stability['lifecycle']['revived'] == 0
    assert stability['split_merge']['split_count'] == stability['split_merge']['merged_count'] == 0
    history = json.loads((out / 'history.json').read_text())
    assert all(entry['eligible'] and not entry['ineligible_reasons'] for entry in history)
    assert 'health' in json.loads((out / 'baseline.json').read_text())


def test_cohort_variance_uses_its_own_mean():
    X = torch.tensor([[1., 2., 3.], [2., 3., 4.], [5., 6., 7.]])
    mask = np.array([True, True, False])
    sae = model()
    split = data.Split('validation', X, Path('unused'), 'unused', mean=X.double().mean(0), probe_cohort=mask,
                     probe_cohort_mean=X[mask].double().mean(0), cached_features=sae.encoder(X).numpy())
    result = evaluation.evaluate(sae, split, 2, 'cpu')
    expected = 1 - X[mask].square().sum().item() / (X[mask] - X[mask].mean(0)).square().sum().item()
    assert result['probe_cohort']['variance_explained'] == pytest.approx(expected)


def test_ste_forward_matches_encoder_and_threshold_gradient_is_local():
    sae = JumpReLUSAE(torch.eye(3), torch.zeros(3), torch.eye(3), torch.zeros(3), torch.ones(3))
    # Per feature: well below, just above (inside the window) and well above the threshold of 1.
    x = torch.tensor([[0.2, 1.01, 3.0]], requires_grad=True)
    threshold = torch.ones(3, requires_grad=True)
    z, l0 = sae.encoder_ste(x, bandwidth=0.1, threshold=threshold)
    assert torch.equal(z, sae.encoder(x)) and l0.item() == 2
    (z.sum() + l0.sum()).backward()
    assert threshold.grad[0] == 0 and threshold.grad[2] == 0 and threshold.grad[1] < 0
    # Gated-off features pass no gradient back to the input; open ones pass it through.
    assert x.grad.tolist() == [[0.0, 1.0, 1.0]]


def test_l0_tolerance_rejects_denser_checkpoints():
    assert selection.within_l0(110, 100, 0.1) and selection.within_l0(90, 100, 0.1)
    assert not selection.within_l0(111, 100, 0.1) and not selection.within_l0(5, 0, 0.1)


def test_full_mode_trains_encoder_into_a_separate_checkpoint(tmp_path):
    src, acts = tmp_path / 'source', tmp_path / 'acts'
    src.mkdir()
    acts.mkdir()
    sae = JumpReLUSAE(torch.eye(3) * 0.5, torch.zeros(3), torch.zeros(3, 3), torch.zeros(3),
                      torch.ones(3) * 0.01)
    loop.save_model(sae, {'architecture': 'jump_relu', 'hf_hook_point_in': 'model.layers.25.output'}, src)
    original_hash = sha256_file(src / 'params.safetensors')
    X = torch.rand(80, 3, generator=torch.Generator().manual_seed(3)) * 3 + 0.1
    torch.save(X[:60], acts / 'train_attempt2.pt')
    torch.save(X[60:], acts / 'validation_attempt2.pt')
    (acts / 'test_attempt2.pt').write_text('must not be read')
    args = argparse.Namespace(sae_dir=src, activations_dir=acts, layer=25,
        activation_file='{split}_attempt2.pt', cache_dir=tmp_path / 'cache',
        output_dir=tmp_path / 'out', probe_config=None, epochs=8, batch_size=15,
        lr=0.05, patience=3, seed=42, device='cpu',
        **{**DECODER_ONLY, 'train': 'full', 'lr_enc': 0.01})
    out = ft.run(args)
    summary = json.loads((out / 'summary.json').read_text())
    adapted, _ = load_gemma_scope_sae(out / 'best', expected_layer=25)
    assert summary['train_mode'] == 'full' and summary['best_epoch'] > 0
    assert summary['adapted']['validation']['mse'] < summary['original']['validation']['mse']
    assert selection.within_l0(summary['adapted']['validation']['l0'], summary['l0_target'], 0.1)
    assert not summary['encoder_unchanged'] and not torch.equal(sae.w_enc, adapted.w_enc)
    assert summary['stability']['n_examples'] == 20
    assert set(np.load(out / 'feature_health.npz')) >= {'activation_correlation', 'encoder_cosine',
                                                         'original_frequency', 'adapted_frequency'}
    assert summary['health']['adapted']['validation']['utilization']['alive_count'] >= 0
    assert sha256_file(src / 'params.safetensors') == original_hash
    assert not args.cache_dir.exists()  # full mode never writes an encoder cache
    assert json.loads((out / 'manifest.json').read_text())['status'] == 'completed'


def test_stability_gate_rejects_drifted_epochs_and_says_why():
    args = argparse.Namespace(l0_tolerance=0.1, min_activation_correlation=0.95, min_jaccard=0.85,
                              max_alive_change=0.05)
    steady = {'activation_correlation_median': 0.99, 'active_set_jaccard_mean': 0.95, 'alive_relative_change': 0.0}
    assert selection.eligibility(100, steady, 100, args) == (True, [])
    drifted = {'activation_correlation_median': 0.9, 'active_set_jaccard_mean': 0.8, 'alive_relative_change': -0.1}
    eligible, reasons = selection.eligibility(120, drifted, 100, args)
    assert not eligible and len(reasons) == 4
    assert any('correlation' in r for r in reasons) and any('Jaccard' in r for r in reasons)
    disabled = argparse.Namespace(l0_tolerance=0.1, **NO_STABILITY_GATES)
    assert selection.eligibility(100, drifted, 100, disabled) == (True, [])
