import copy
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
    runner.last_fired_example = torch.zeros(6, dtype=torch.long)
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
    script = Path(__file__).resolve().parents[3] / 'scripts/experiment_sae_resampling.py'
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
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *args):
            self.activations = x
        def __len__(self):
            return len(x)
        def __getitem__(self, index):
            return x[index]
    monkeypatch.setattr(module, 'ActivationDataset', Dataset)
    settings = tmp_path / 'experiment.yaml'
    settings.write_text(yaml.safe_dump(dict(source_trial=str(source), output_dir=str(tmp_path / 'output'),
                                          additional_epochs=1, resampling_budget=2, seeds=[7])))
    monkeypatch.setattr(sys, 'argv', ['experiment', '--config', str(settings)])
    module.main()
    assert checkpoint.read_bytes() == original
    for branch in ['control', 'resampling']:
        directory = tmp_path / 'output/seed_7' / branch
        assert (directory / 'final_checkpoint.pt').exists()
        assert len(json.loads((directory / 'history.json').read_text())) == 1
    assert len(json.loads((tmp_path / 'output/seed_7/resampling/resampling.json').read_text())['feature_ids']) == 2
