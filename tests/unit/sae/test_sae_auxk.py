import copy
import json
import math
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader

from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


def make_runner(coefficient=0.25, window=2, aux_k=2):
    model = SparseAutoencoder(2, 2, 3, "topk", 1)
    with torch.no_grad():
        model.encoder_layer[0].weight.zero_()
        model.encoder_layer[0].bias.copy_(torch.tensor([2., 1., 0.]))
        model.decoder_layer[0].weight.copy_(torch.tensor([[1., 0., 1.], [0., 1., 0.]]))
        model.decoder_layer[0].bias.zero_()
    return ModelRunner(
        model, 0.0, torch.optim.SGD(model.parameters(), lr=0.0),
        aux_coefficient=coefficient, aux_k=aux_k,
        aux_dead_window_examples=window,
    )


def test_auxk_warmup_and_auxiliary_firing_does_not_reset_counters():
    runner = make_runner()
    batch = torch.tensor([[2., 3.]])
    first = runner._common(batch, training_mode=True)
    assert first["aux_loss"].item() == 0
    second = runner._common(batch, training_mode=True)
    assert second["aux_loss"].item() == pytest.approx(2.0)
    assert second["aux_eligible_feature_fraction"] == pytest.approx(2 / 3)
    assert second["loss"].item() == pytest.approx(5.0)
    assert runner.training_examples_seen == 2
    assert runner.last_fired_example.tolist() == [2, 0, 0]
    runner._common(batch, training_mode=True)
    assert runner.last_fired_example.tolist() == [3, 0, 0]


def test_auxk_gradients_reach_eligible_weights_without_main_residual_or_bias():
    runner = make_runner(window=1)
    outputs = runner._common(torch.tensor([[2., 3.]]), training_mode=True)
    outputs["reconstructed"].retain_grad()
    outputs["aux_loss"].backward()
    encoder_grad = runner.model.encoder_layer[0].weight.grad
    decoder_grad = runner.model.decoder_layer[0].weight.grad
    assert encoder_grad[1].abs().sum() > 0
    assert decoder_grad[:, 1].abs().sum() > 0
    assert encoder_grad[0].abs().sum() == 0
    assert decoder_grad[:, 0].abs().sum() == 0
    assert runner.model.decoder_layer[0].bias.grad is None
    assert outputs["reconstructed"].grad is None


def test_auxk_handles_no_eligible_features_and_caps_selection():
    runner = make_runner()
    runner.training_examples_seen = 10
    runner.last_fired_example = torch.tensor([10, 10, 10])
    x = torch.tensor([[2., 3.]])
    _, reconstruction, pre_topk = runner.model.forward_with_pre_topk(x)
    loss, fraction, _ = runner._auxiliary_loss(x, reconstruction, pre_topk)
    assert loss.item() == 0
    assert fraction == 0
    runner.last_fired_example[1] = 0  # Only one eligible feature, aux_k is two.
    loss, fraction, _ = runner._auxiliary_loss(x, reconstruction, pre_topk)
    assert loss.item() == pytest.approx(2.0)
    assert fraction == pytest.approx(1 / 3)


def test_main_path_revival_removes_feature_from_auxk_eligibility():
    runner = make_runner(window=2)
    batch = torch.tensor([[2., 3.]])
    runner.train_epoch(DataLoader(batch.repeat(2, 1), batch_size=1))
    assert runner.last_fired_example.tolist() == [2, 0, 0]
    # Feature 1 now wins the main TopK competition in the next epoch.
    with torch.no_grad():
        runner.model.encoder_layer[0].bias[1] = 3.0
    outputs = runner._common(batch, training_mode=True)
    assert outputs["sparse_features"].tolist() == [[0., 3., 0.]]
    assert runner.last_fired_example.tolist() == [2, 3, 0]
    assert outputs["aux_eligible_feature_fraction"] == pytest.approx(1 / 3)
    runner.optimizer.zero_grad(set_to_none=True)
    outputs["aux_loss"].backward()
    # Only zero-valued feature 2 remains eligible; revived feature 1 is excluded.
    assert runner.model.encoder_layer[0].bias.grad[1].item() == 0
    assert runner.model.decoder_layer[0].weight.grad[:, 1].abs().sum().item() == 0


def test_zero_eligible_activations_do_not_select_active_features(monkeypatch):
    runner = make_runner(window=1)
    runner.training_examples_seen = 10
    runner.last_fired_example = torch.tensor([10, 0, 0])
    # An active feature has a large activation; all eligible values are zero.
    pre_topk = torch.tensor([[100., 0., 0.]], requires_grad=True)
    decode = Mock(wraps=runner.model.decode_without_bias)
    monkeypatch.setattr(runner.model, "decode_without_bias", decode)
    loss, fraction, _ = runner._auxiliary_loss(
        torch.ones(1, 2), torch.zeros(1, 2), pre_topk,
    )
    assert torch.equal(decode.call_args.args[0], torch.zeros_like(pre_topk))
    assert loss.item() == pytest.approx(1.0)
    assert fraction == pytest.approx(2 / 3)
    loss.backward()
    assert torch.isfinite(pre_topk.grad).all()
    assert pre_topk.grad[0, 0].item() == 0


def test_auxk_normalization_and_bias_free_prediction():
    runner = make_runner(window=1)
    runner.model.input_scale.fill_(2.0)
    with torch.no_grad():
        runner.model.decoder_layer[0].bias.fill_(1.0)
    # Main normalized prediction is [3, 1], residual [0, 3], auxiliary [0, 1].
    outputs = runner._common(torch.tensor([[6., 8.]]), training_mode=True)
    assert outputs["aux_loss"].item() == pytest.approx(2.0)


def test_disabled_auxk_matches_original_optimizer_update():
    runner = make_runner(coefficient=0)
    reference = copy.deepcopy(runner.model)
    runner.optimizer = torch.optim.SGD(runner.model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    batch = torch.tensor([[2., 3.]])
    _, reconstructed = reference(batch)
    loss = torch.nn.functional.mse_loss(reconstructed, batch)
    loss.backward()
    reference_optimizer.step()
    reference.normalize_decoder_weights()
    metrics = runner.train_epoch(DataLoader(batch, batch_size=1))
    for key, value in reference.state_dict().items():
        assert torch.equal(value, runner.model.state_dict()[key])
    assert metrics["aux_loss"] == 0
    assert metrics["loss"] == metrics["rec_loss"]


def test_auxk_training_metrics_and_evaluation_are_separate():
    runner = make_runner(window=2)
    loader = DataLoader(torch.tensor([[2., 3.]] * 3), batch_size=2)
    metrics = runner.train_epoch(loader)
    assert metrics["aux_loss"] == pytest.approx(2.0)
    assert metrics["weighted_aux_loss"] == pytest.approx(0.5)
    assert metrics["aux_eligible_feature_fraction"] == pytest.approx(2 / 3)
    assert metrics["mean_active_features"] == 1
    previous_fired = runner.last_fired_example.clone()
    for evaluate in (runner.val_epoch, runner.test):
        metrics = evaluate(loader)
        assert metrics["loss"] == metrics["rec_loss"]
        assert runner.training_examples_seen == 3
        assert torch.equal(runner.last_fired_example, previous_fired)


@pytest.mark.parametrize("settings", [
    {"aux_coefficient": -1}, {"aux_coefficient": float("nan")},
    {"aux_coefficient": float("inf")}, {"aux_coefficient": True},
    {"aux_k": 0}, {"aux_k": True}, {"aux_dead_window_examples": 0},
    {"aux_dead_window_examples": 1.5}, {"aux_coefficient": 1, "aux_k": 4},
])
def test_invalid_auxk_settings(settings):
    model = SparseAutoencoder(2, 2, 3, "topk", 1)
    with pytest.raises(ValueError):
        ModelRunner(model, 0.0, torch.optim.SGD(model.parameters(), lr=0), **settings)


def test_enabled_auxk_rejects_l1():
    model = SparseAutoencoder(2, 2, 3, "l1")
    with pytest.raises(ValueError, match="topk"):
        ModelRunner(model, 0.0, torch.optim.SGD(model.parameters(), lr=0),
                    aux_coefficient=0.1, aux_k=2)


def test_auxiliary_metrics_weight_uneven_batches_by_examples():
    runner = make_runner(window=3)
    loader = DataLoader(torch.tensor([[2., 3.]] * 3), batch_size=2)
    metrics = runner.train_epoch(loader)
    # First two examples are warmup; only the final example uses AuxK.
    assert metrics["aux_loss"] == pytest.approx(2 / 3)
    assert metrics["weighted_aux_loss"] == pytest.approx(0.5 / 3)
    assert metrics["aux_eligible_feature_fraction"] == pytest.approx(2 / 9)
    assert metrics["loss"] == pytest.approx(metrics["rec_loss"] + 0.5 / 3)
    assert metrics["mean_active_features"] == 1
    assert metrics["inactive_feature_fraction"] == pytest.approx(2 / 3)
    assert runner.last_feature_frequencies.tolist() == [1., 0., 0.]


@pytest.mark.parametrize("method", ["val_epoch", "test"])
def test_evaluation_never_calls_auxiliary_path(method, monkeypatch):
    runner = make_runner(window=1)
    loader = DataLoader(torch.tensor([[2., 3.]] * 3), batch_size=2)
    runner.train_epoch(loader)
    counters = runner.last_fired_example.clone()
    state = copy.deepcopy(runner.model.state_dict())
    forbidden = Mock(side_effect=AssertionError("Evaluation must not compute AuxK"))
    monkeypatch.setattr(runner.model, "forward_with_pre_topk", forbidden)
    monkeypatch.setattr(runner, "_auxiliary_loss", forbidden)
    for coefficient in (0.25, 1000.0):
        runner.aux_coefficient = coefficient
        metrics = getattr(runner, method)(loader)
        assert metrics["loss"] == metrics["rec_loss"] == pytest.approx(4.5)
        assert not any(key.startswith("aux_") or key == "weighted_aux_loss" for key in metrics)
        assert runner.training_examples_seen == 3
        assert torch.equal(runner.last_fired_example, counters)
        for key, value in state.items():
            assert torch.equal(runner.model.state_dict()[key], value)
    forbidden.assert_not_called()


def test_auxiliary_diagnostics_aggregate_squared_sums_over_uneven_batches():
    runner = make_runner(window=1)
    loader = DataLoader(torch.tensor([[2., 3.], [2., 5.], [2., 7.]]), batch_size=2)
    metrics = runner.train_epoch(loader)
    auxiliary = metrics["auxiliary"]
    assert auxiliary["measured_examples"] == 3
    assert auxiliary["eligible_feature_fraction"] == pytest.approx(2 / 3)
    assert auxiliary["mean_positive_features"] == 1
    assert auxiliary["eligible_positive_activation_fraction"] == pytest.approx(0.5)
    assert auxiliary["prediction_rms"] == pytest.approx(math.sqrt(3 / 6))
    assert auxiliary["residual_rms"] == pytest.approx(math.sqrt(83 / 6))
    assert auxiliary["residual_error_reduction"] == pytest.approx(1 - 56 / 83)
    assert json.loads(json.dumps(auxiliary, allow_nan=False)) == auxiliary
    assert "auxiliary" not in runner.val_epoch(loader)


def test_auxiliary_diagnostics_exclude_warmup_examples():
    runner = make_runner(window=3)
    loader = DataLoader(torch.tensor([[2., 3.]] * 3), batch_size=2)
    auxiliary = runner.train_epoch(loader)["auxiliary"]
    assert auxiliary["measured_examples"] == 1
    assert auxiliary["mean_positive_features"] == 1
    assert auxiliary["residual_rms"] == pytest.approx(math.sqrt(9 / 2))
    assert auxiliary["residual_error_reduction"] == pytest.approx(1 - 4 / 9)


def test_auxiliary_diagnostics_unavailable_during_warmup_and_absent_when_disabled():
    loader = DataLoader(torch.tensor([[2., 3.]]), batch_size=1)
    auxiliary = make_runner(window=10).train_epoch(loader)["auxiliary"]
    assert auxiliary["measured_examples"] == 0
    assert auxiliary["eligible_feature_fraction"] == 0
    for key in ("mean_positive_features", "prediction_rms", "residual_rms", "residual_error_reduction", "eligible_positive_activation_fraction"):
        assert auxiliary[key] is None
    assert "auxiliary" not in make_runner(coefficient=0).train_epoch(loader)


def test_eligible_positive_fraction_counts_before_selection_and_excludes_active_features():
    runner = make_runner(window=1, aux_k=1)
    with torch.no_grad():
        runner.model.encoder_layer[0].bias[2] = 0.5
    loader = DataLoader(torch.tensor([[2., 3.]] * 3), batch_size=2)
    auxiliary = runner.train_epoch(loader)["auxiliary"]
    assert auxiliary["eligible_positive_activation_fraction"] == 1.0
    assert auxiliary["mean_positive_features"] == 1.0


@pytest.mark.parametrize("residual, expected_reduction", [(0., None), (-1., -3.), (1., 1.)])
def test_auxiliary_error_reduction_handles_zero_negative_and_perfect_cases(residual, expected_reduction):
    runner = make_runner(window=1)
    loader = DataLoader(torch.tensor([[2., residual]]), batch_size=1)
    auxiliary = runner.train_epoch(loader)["auxiliary"]
    if expected_reduction is None:
        assert auxiliary["residual_error_reduction"] is None
    else:
        assert auxiliary["residual_error_reduction"] == pytest.approx(expected_reduction)
    json.dumps(auxiliary, allow_nan=False)
