"""Refinement recovery and artifact contracts on small CPU activations."""
import json
from pathlib import Path

import pytest
import torch

from mas_sae.sae import refinement
from mas_sae.sae.callbacks.checkpointing import load_sae_checkpoint
from test_sae_resampling import setup


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    runner, x = setup()
    source = tmp_path / 'source'
    (source / 'checkpoints').mkdir(parents=True)
    config = dict(input_dim=3, hidden_dim=8, latent_dim=6, sparsity_mode='topk', top_k=1,
                  evaluate_test=False, activation_run_name='synthetic', layer=25,
                  lr=.001, sparsity_coefficient=0, inactivity_window_examples=10,
                  aux_coefficient=.0625, aux_k=2, aux_dead_window_examples=10,
                  batch_size=16, train_shuffle=True)
    (source / 'config.json').write_text(json.dumps(config))
    torch.save(dict(model=runner.model.state_dict(), optimizer=runner.optimizer.state_dict(),
                    training_examples_seen=100, last_fired_example=runner.last_fired_example),
               source / 'checkpoints/last_checkpoint.pt')
    location = tmp_path / 'data/activations/synthetic/layer_25'
    location.mkdir(parents=True)
    for split in ('train', 'validation'):
        torch.save(x, location / f'{split}.pt')
    return dict(source_run=str(source), output_dir=str(tmp_path / 'output'), additional_epochs=3,
                resampling_budget=2, seeds=[7], branches=[
                    dict(name='warmup_control', strategy='none', warmup_steps=5),
                    dict(name='residual_warmup', strategy='residual', warmup_steps=5)])


def test_resume_matches_uninterrupted_and_never_replaces_again(experiment, tmp_path, monkeypatch):
    settings = experiment
    refinement.run_refinement(settings, tmp_path)
    original_output = Path(settings['output_dir'])
    settings = dict(settings, output_dir=str(tmp_path / 'interrupted'))
    original_save = refinement.save_checkpoint

    def interrupt(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if path.name == 'last_checkpoint.pt' and args[2] == 1 and path.parent.name == 'residual_warmup':
            raise RuntimeError('simulated interruption after commit')

    monkeypatch.setattr(refinement, 'save_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        refinement.run_refinement(settings, tmp_path)
    monkeypatch.setattr(refinement, 'save_checkpoint', original_save)
    def forbidden(*args, **kwargs):
        raise AssertionError('Committed branches must not resample')
    monkeypatch.setattr(refinement, 'resample_features', forbidden)
    refinement.run_refinement(settings, tmp_path, resume=True)
    for branch in ('warmup_control', 'residual_warmup'):
        relative = Path('seed_7') / branch
        a = torch.load(original_output / relative / 'final_checkpoint.pt', weights_only=True)
        b = torch.load(Path(settings['output_dir']) / relative / 'final_checkpoint.pt', weights_only=True)
        assert a['history'] == b['history']
        assert a['warmup_history'] == b['warmup_history']
        assert torch.equal(a['loader_rng'], b['loader_rng'])
        for key, value in a['model_state_dict'].items():
            assert torch.equal(value, b['model_state_dict'][key])
        assert load_sae_checkpoint(Path(settings['output_dir']) / relative / 'final_checkpoint.pt').top_k == 1
    report = json.loads((Path(settings['output_dir']) / 'comparison.json').read_text())
    assert report['selected_checkpoint'] is None
    assert report['branches'][1]['matched_control'] == 'warmup_control'
    assert not (tmp_path / 'data/activations/synthetic/layer_25/test.pt').exists()


@pytest.mark.parametrize('change', ['config', 'source', 'data'])
def test_resume_rejects_changed_inputs(experiment, tmp_path, change):
    refinement.run_refinement(experiment, tmp_path)
    if change == 'config':
        experiment['positive_fraction'] = .02
    elif change == 'source':
        path = Path(experiment['source_run']) / 'checkpoints/last_checkpoint.pt'
        state = torch.load(path, weights_only=True)
        state['training_examples_seen'] += 1
        torch.save(state, path)
    else:
        torch.save(torch.rand(32, 3), tmp_path / 'data/activations/synthetic/layer_25/train.pt')
    with pytest.raises(ValueError, match='changed'):
        refinement.run_refinement(experiment, tmp_path, resume=True)


def test_dry_run_and_explicit_selection(experiment, tmp_path):
    output = Path(experiment['output_dir'])
    refinement.run_refinement(experiment, tmp_path, dry_run=True)
    assert not output.exists()
    refinement.run_refinement(experiment, tmp_path)
    checkpoint = output / 'seed_7/residual_warmup/final_checkpoint.pt'
    selection = refinement.select_checkpoint(output, checkpoint)
    assert selection['branch'] == 'residual_warmup'
    assert selection['requires_new_feature_analysis']
    assert selection['checkpoint_sha256'] == refinement.sha256_file(checkpoint)
    with pytest.raises(ValueError, match='listed'):
        refinement.select_checkpoint(output, Path(experiment['source_run']) / 'checkpoints/last_checkpoint.pt')
    refinement.run_refinement(experiment, tmp_path, resume=True)
    assert json.loads((output / 'selection.json').read_text()) == selection
    assert refinement.sha256_file(checkpoint) == selection['checkpoint_sha256']


def test_reject_invalid_warmup_without_writes(experiment, tmp_path):
    experiment['branches'][0]['warmup_steps'] = 100
    with pytest.raises(ValueError, match='Warmup cannot exceed'):
        refinement.run_refinement(experiment, tmp_path, dry_run=True)
    assert not Path(experiment['output_dir']).exists()
