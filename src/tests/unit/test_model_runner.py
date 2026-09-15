from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from mas_sae.sae import model_runner as model_runner_module
from mas_sae.sae.model_runner import ModelRunner


class TrackingModel(nn.Module):
    """Small deterministic model that records the context of each forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.training_states: list[bool] = []
        self.grad_states: list[bool] = []

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.training_states.append(self.training)
        self.grad_states.append(torch.is_grad_enabled())
        scaled_activations = activations * self.scale
        return torch.relu(scaled_activations), scaled_activations


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters: object, lr: float) -> None:
        super().__init__(parameters, lr=lr)
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args: object, **kwargs: object) -> None:
        self.zero_grad_calls += 1
        super().zero_grad(*args, **kwargs)

    def step(self, *args: object, **kwargs: object) -> object:
        self.step_calls += 1
        return super().step(*args, **kwargs)


@pytest.fixture(autouse=True)
def disable_progress_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_runner_module,
        "tqdm",
        lambda iterable, **_: iterable,
    )


def test_loss_fn_combines_reconstruction_and_weighted_sparsity_losses() -> None:
    runner = ModelRunner(
        model=Mock(),
        sparsity_coefficient=0.25,
        optimizer=Mock(),
    )
    batch = torch.tensor([[1.0, -1.0], [3.0, 1.0]])
    reconstructed = torch.tensor([[0.0, -2.0], [1.0, 1.0]])
    sparse_features = torch.tensor([[-2.0, 0.0], [1.0, 5.0]])

    loss = runner._loss_fn(reconstructed, batch, sparse_features)

    expected_loss = F.mse_loss(reconstructed, batch)
    expected_loss += 0.25 * sparse_features.abs().mean()
    assert torch.equal(loss, expected_loss)


def test_common_passes_batch_to_model_and_returns_outputs() -> None:
    batch = torch.randn(3, 4)
    sparse_features = torch.randn(3, 6)
    reconstructed = torch.randn(3, 4)
    model = Mock(return_value=(sparse_features, reconstructed))
    runner = ModelRunner(model=model, sparsity_coefficient=0.1, optimizer=Mock())

    loss, returned_reconstruction, returned_features = runner._common(batch)

    model.assert_called_once_with(batch)
    assert returned_reconstruction is reconstructed
    assert returned_features is sparse_features
    expected_loss = F.mse_loss(reconstructed, batch)
    expected_loss += 0.1 * sparse_features.abs().mean()
    assert torch.equal(loss, expected_loss)


def test_train_epoch_enables_training_and_updates_model_for_every_batch() -> None:
    model = TrackingModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    runner = ModelRunner(model, sparsity_coefficient=0.05, optimizer=optimizer)
    dataloader = DataLoader(
        torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
        batch_size=2,
    )
    initial_scale = model.scale.detach().clone()

    loss = runner.train_epoch(dataloader)

    assert model.training is True
    assert model.training_states == [True, True]
    assert model.grad_states == [True, True]
    assert optimizer.zero_grad_calls == len(dataloader)
    assert optimizer.step_calls == len(dataloader)
    assert not torch.equal(model.scale.detach(), initial_scale)
    assert isinstance(loss, float)
    assert loss >= 0


@pytest.mark.parametrize("runner_method", ["val_epoch", "test"])
def test_evaluation_pipeline_disables_gradients_and_does_not_update_model(
    runner_method: str,
) -> None:
    model = TrackingModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    sparsity_coefficient = 0.2
    runner = ModelRunner(model, sparsity_coefficient, optimizer)
    activations = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    dataloader = DataLoader(activations, batch_size=2)
    initial_scale = model.scale.detach().clone()
    expected_batch_losses = []
    for batch in dataloader:
        reconstructed = batch * initial_scale
        sparse_features = torch.relu(reconstructed)
        expected_batch_losses.append(
            F.mse_loss(reconstructed, batch).item()
            + sparsity_coefficient * sparse_features.abs().mean().item()
        )

    loss = getattr(runner, runner_method)(dataloader)

    assert model.training is False
    assert model.training_states == [False, False]
    assert model.grad_states == [False, False]
    assert optimizer.zero_grad_calls == 0
    assert optimizer.step_calls == 0
    assert torch.equal(model.scale.detach(), initial_scale)
    assert loss == pytest.approx(sum(expected_batch_losses) / len(dataloader))
