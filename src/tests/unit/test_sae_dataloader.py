from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mas_sae.sae import dataloader as dataloader_module
from mas_sae.sae.dataloader import ActivationDataset, create_sae_dataloader


@pytest.fixture
def activations() -> torch.Tensor:
    return torch.arange(15, dtype=torch.float32).reshape(5, 3)


@pytest.fixture
def mocked_activation_store(
    monkeypatch: pytest.MonkeyPatch,
    activations: torch.Tensor,
) -> tuple[Mock, Mock]:
    store = Mock()
    store.load_activations.return_value = activations
    store_constructor = Mock(return_value=store)
    monkeypatch.setattr(dataloader_module, "ActivationStore", store_constructor)
    return store_constructor, store


def test_activation_dataset_loads_from_configured_store(
    mocked_activation_store: tuple[Mock, Mock],
    activations: torch.Tensor,
) -> None:
    store_constructor, store = mocked_activation_store

    dataset = ActivationDataset(location="s3://test-bucket/activations")

    store_constructor.assert_called_once_with("s3://test-bucket/activations")
    store.load_activations.assert_called_once_with()
    assert dataset.activations is activations


def test_activation_dataset_length_is_number_of_activation_vectors(
    mocked_activation_store: tuple[Mock, Mock],
    activations: torch.Tensor,
) -> None:
    dataset = ActivationDataset(location="test-location")

    assert len(dataset) == activations.shape[0]


def test_activation_dataset_returns_activation_at_requested_index(
    mocked_activation_store: tuple[Mock, Mock],
    activations: torch.Tensor,
) -> None:
    dataset = ActivationDataset(location="test-location")

    assert torch.equal(dataset[2], activations[2])


def test_create_sae_dataloader_uses_requested_location(
    mocked_activation_store: tuple[Mock, Mock],
) -> None:
    store_constructor, _ = mocked_activation_store

    create_sae_dataloader(
        batch_size=2,
        num_workers=0,
        location="s3://test-bucket/activations",
    )

    store_constructor.assert_called_once_with("s3://test-bucket/activations")


def test_create_sae_dataloader_configures_batching_and_preserves_order(
    mocked_activation_store: tuple[Mock, Mock],
    activations: torch.Tensor,
) -> None:
    loader = create_sae_dataloader(
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    assert isinstance(loader, DataLoader)
    assert isinstance(loader.dataset, ActivationDataset)
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.batch_size == 2
    assert loader.num_workers == 0

    batches = list(loader)
    assert [batch.shape for batch in batches] == [(2, 3), (2, 3), (1, 3)]
    assert torch.equal(torch.cat(batches), activations)


def test_create_sae_dataloader_uses_random_sampler_when_shuffling(
    mocked_activation_store: tuple[Mock, Mock],
) -> None:
    loader = create_sae_dataloader(
        batch_size=2,
        num_workers=0,
        shuffle=True,
    )

    assert isinstance(loader.sampler, RandomSampler)
