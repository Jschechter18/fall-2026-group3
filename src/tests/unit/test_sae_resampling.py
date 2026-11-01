import copy
import pytest
import torch
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.resampling import resample_features


def setup():
    torch.manual_seed(12)
    model = SparseAutoencoder(3, 8, 6, 'topk', 1)
    optimizer = torch.optim.Adam(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    with torch.no_grad():
        model.encoder_layer[0].bias[:4] = -100
    runner = ModelRunner(model, 0, optimizer, aux_coefficient=.0625, aux_k=2, aux_dead_window_examples=10)
    runner.training_examples_seen = 100
    runner.last_fired_example = torch.tensor([0, 0, 0, 0, 100, 100])
    return runner, torch.rand(32, 3)


def test_resampling_selectivity_normalization_and_optimizer_reset():
    runner, x = setup()
    before = copy.deepcopy(runner.model.state_dict())
    result = resample_features(runner, x, budget=2)
    ids = result['feature_ids']
    assert len(ids) == 2
    assert set(ids) <= set(range(4))
    keep = [i for i in range(6) if i not in ids]
    encoder, decoder = runner.model.encoder_layer[0], runner.model.decoder_layer[0]
    assert torch.equal(encoder.weight[keep], before['encoder_layer.0.weight'][keep])
    assert torch.equal(decoder.weight[:, keep], before['decoder_layer.0.weight'][:, keep])
    assert torch.equal(decoder.bias, before['decoder_layer.0.bias'])
    assert torch.allclose(decoder.weight[:, ids].norm(dim=0), torch.ones(2))
    assert (runner.last_fired_example[ids] == 100).all()
    for parameter, axis in [(encoder.weight, 0), (encoder.bias, 0), (decoder.weight, 1)]:
        moment = runner.optimizer.state[parameter]['exp_avg']
        assert (moment.index_select(axis, torch.tensor(ids)) == 0).all()
        assert (moment.index_select(axis, torch.tensor(keep)) != 0).all()
    assert (runner.model.encode_pre_topk(x)[:, ids] > 0).any(0).all()


def test_resampling_repeatable_and_no_candidates_is_noop():
    a, x = setup()
    b, _ = setup()
    assert resample_features(a, x, 2, 7) == resample_features(b, x, 2, 7)
    for left, right in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(left, right)
    a.last_fired_example.fill_(100)
    before = copy.deepcopy(a.model.state_dict())
    assert resample_features(a, x, 2)['feature_ids'] == []
    assert all(torch.equal(before[k], v) for k, v in a.model.state_dict().items())


def test_paired_workflow_smoke(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    import yaml
    script = Path(__file__).resolve().parents[3] / 'scripts/sae/experiment_sae_resampling.py'
    spec = importlib.util.spec_from_file_location('resampling_experiment', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner, x = setup()
    source = tmp_path / 'source'
    (source / 'checkpoints').mkdir(parents=True)
    config = dict(input_dim=3, hidden_dim=8, latent_dim=6, sparsity_mode='topk', top_k=1,
                  evaluate_test=False, activation_run_name='synthetic', layer=25,
                  lr=.001, sparsity_coefficient=0, inactivity_window_examples=10,
                  aux_coefficient=.0625, aux_k=2, aux_dead_window_examples=10,
                  batch_size=16, train_shuffle=True)
    (source / 'config.json').write_text(json.dumps(config))
    checkpoint = source / 'checkpoints/last_checkpoint.pt'
    torch.save(dict(model=runner.model.state_dict(), optimizer=runner.optimizer.state_dict(),
                    training_examples_seen=100, last_fired_example=runner.last_fired_example), checkpoint)
    original = checkpoint.read_bytes()
    location = tmp_path / 'data/activations/synthetic/layer_25'
    location.mkdir(parents=True)
    for split in ['train', 'validation']:
        torch.save(x, location / f'{split}.pt')
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'scripts/sae/experiment.py'))
    settings = tmp_path / 'experiment.yaml'
    settings.write_text(yaml.safe_dump(dict(source_trial=str(source), output_dir=str(tmp_path / 'output'),
                                          additional_epochs=1, resampling_budget=2, seeds=[7], branches=[
        {'name': 'control', 'strategy': 'none'},
        {'name': 'warmup_control', 'strategy': 'none', 'warmup_steps': 2},
        {'name': 'residual', 'strategy': 'residual'},
        {'name': 'residual_warmup', 'strategy': 'residual', 'warmup_steps': 2},
        {'name': 'bricken_warmup', 'strategy': 'bricken', 'warmup_steps': 2},
    ])))
    monkeypatch.setattr(sys, 'argv', ['experiment', '--config', str(settings)])
    training_batches = []
    original_common = module.ModelRunner._common
    def record_order(self, batch, training_mode=False):
        if training_mode:
            training_batches.append(batch.clone())
        return original_common(self, batch, training_mode)
    monkeypatch.setattr(module.ModelRunner, '_common', record_order)
    module.main()
    # Two batches per epoch; all five branches must see identical examples/order.
    assert len(training_batches) == 10
    for offset in [2, 4, 6, 8]:
        assert torch.equal(training_batches[0], training_batches[offset])
        assert torch.equal(training_batches[1], training_batches[offset + 1])
    assert checkpoint.read_bytes() == original
    for branch in ['control', 'warmup_control', 'residual', 'residual_warmup', 'bricken_warmup']:
        directory = tmp_path / 'output/seed_7' / branch
        assert (directory / 'final_checkpoint.pt').exists()
        assert len(json.loads((directory / 'history.json').read_text())) == 1
    directory = tmp_path / 'output/seed_7'
    events = [json.loads((directory / name / 'resampling.json').read_text())
              for name in ['residual', 'residual_warmup', 'bricken_warmup']]
    assert all(event['feature_ids'] == events[0]['feature_ids'] for event in events)
    assert events[0]['source_indices'] == events[1]['source_indices']
    for name in ['warmup_control', 'residual_warmup', 'bricken_warmup']:
        rates = json.loads((directory / name / 'learning_rates.json').read_text())['rates_per_update']
        assert rates[0][0] == pytest.approx(.0001)
        assert rates[1][0] == pytest.approx(.001)
    assert (directory / 'control/post_intervention_metrics.json').exists()
    assert (directory / 'bricken_warmup/feature_counts_epoch_001.json').exists()
    from mas_sae.sae.callbacks.checkpointing import load_sae_checkpoint
    restored = load_sae_checkpoint(directory / 'bricken_warmup/final_checkpoint.pt')
    assert restored.top_k == 1
    initial_control = torch.load(directory / 'control/initial_checkpoint.pt', weights_only=True)
    initial_treatment = torch.load(directory / 'bricken_warmup/initial_checkpoint.pt', weights_only=True)
    assert all(torch.equal(value, initial_treatment['model_state_dict'][key])
               for key, value in initial_control['model_state_dict'].items())


def test_reference_initialization_uses_inputs_zero_bias_and_alive_scale():
    from mas_sae.sae.resampling import scan_features
    runner, x = setup()
    positive, main, errors = scan_features(runner.model, x)
    alive_norm = runner.model.encoder_layer[0].weight[4:].norm(dim=-1).mean().item()
    result = resample_features(runner, x, 2, 7, strategy='bricken',
                               feature_ids=torch.tensor([0, 1]), scan=(positive, main, errors))
    ids = result['feature_ids']
    expected = torch.nn.functional.normalize(x[result['source_indices']], dim=-1)
    encoder, decoder = runner.model.encoder_layer[0], runner.model.decoder_layer[0]
    assert torch.allclose(decoder.weight[:, ids].T, expected)
    assert torch.allclose(encoder.weight[ids], expected * alive_norm * .2)
    assert (encoder.bias[ids] == 0).all()
    assert result['sampling_weight'] == 'reconstruction_mse_squared'


def test_warmup_endpoints_and_constant_schedule():
    from mas_sae.sae.resampling import LinearWarmup
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    schedule = LinearWarmup(optimizer, steps=4, start_factor=.1)
    for _ in range(6):
        schedule()
    assert [r[0] for r in schedule.history] == pytest.approx([.001, .004, .007, .01, .01, .01])
    constant = LinearWarmup(optimizer)
    constant()
    assert constant.history == [[.01]]
    with pytest.raises(ValueError):
        LinearWarmup(optimizer, steps=1)


def test_invalid_selection_does_not_mutate_model():
    runner, x = setup()
    before = copy.deepcopy(runner.model.state_dict())
    for ids in [torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([6]), torch.tensor([4])]:
        with pytest.raises(ValueError):
            resample_features(runner, x, budget=2, feature_ids=ids)
    assert all(torch.equal(before[k], v) for k, v in runner.model.state_dict().items())
