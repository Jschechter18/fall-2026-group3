"""Verify explicit weight regularization and its validation boundary."""
import pytest
import torch
import yaml
from pathlib import Path

from mas_sae.sae.hyperparamters import Hyperparameters, validate_hyperparameters
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder
from mas_sae.sae.training import _saved_hyperparameters
from mas_sae.sae.tuning import expand_search_space
from dataclasses import asdict


@pytest.mark.parametrize('mode', ['l1', 'topk'])
def test_penalty_gradients_and_validation_exclusion(mode):
    torch.manual_seed(4)
    model = SparseAutoencoder(4, 8, 6, mode, 2)
    optimizer = torch.optim.Adam(model.parameters())
    runner = ModelRunner(model, 0.1, optimizer, weight_l2_coefficient=0.01)
    batch = torch.randn(3, 4)
    output = runner._common(batch, training_mode=True)
    penalty = output['weighted_weight_l2_loss']
    encoder = model.encoder_layer[0]
    expected = 0.01 * (encoder.weight.square().sum()
                       + model.decoder_layer[0].weight.square().sum())
    torch.testing.assert_close(penalty, expected)
    grad, bias_grad = torch.autograd.grad(
        penalty, (encoder.weight, encoder.bias), allow_unused=True)
    torch.testing.assert_close(grad, 0.02 * encoder.weight)
    assert bias_grad is None
    validation = runner._common(batch, training_mode=False)
    assert validation['weighted_weight_l2_loss'].item() == 0
    torch.testing.assert_close(validation['loss'],
                               validation['rec_loss'] + validation['weighted_sparsity_loss'])


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True])
def test_invalid_coefficient(value):
    with pytest.raises(ValueError, match='weight_l2_coefficient'):
        validate_hyperparameters(Hyperparameters(weight_l2_coefficient=value))


def test_legacy_config_and_sweep():
    legacy = asdict(Hyperparameters())
    legacy.pop('weight_l2_coefficient')
    assert _saved_hyperparameters(legacy).weight_l2_coefficient == 0
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / 'configs/sae_hp_tuning/layer_25_l1_weight_l2.yaml').read_text())
    candidates = expand_search_space(config)
    assert len(candidates) == 6
    assert candidates[0][1].weight_l2_coefficient == 0
    assert all(hp.sparsity_mode == 'l1' for _, hp in candidates)
