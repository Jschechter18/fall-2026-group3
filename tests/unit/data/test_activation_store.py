import pytest
import torch

from mas_sae.data.activation_store import ActivationStore


def test_local_store_round_trip(tmp_path):
    store = ActivationStore(str(tmp_path / "activations"))

    expected = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(4, 3)

    store.save_activations("discovery", expected)

    actual = store.load_activations("discovery")

    assert torch.equal(actual, expected)


def test_save_creates_store_directory_and_split_file(tmp_path):
    location = tmp_path / "nested" / "activations"
    store = ActivationStore(str(location))

    activations = torch.ones(2, 4)

    store.save_activations("validation", activations)

    assert location.is_dir()
    assert (location / "validation.pt").is_file()


def test_splits_are_stored_independently(tmp_path):
    store = ActivationStore(str(tmp_path / "activations"))

    discovery = torch.zeros(2, 3)
    validation = torch.ones(3, 3)

    store.save_activations("discovery", discovery)
    store.save_activations("validation", validation)

    assert torch.equal(
        store.load_activations("discovery"),
        discovery,
    )
    assert torch.equal(
        store.load_activations("validation"),
        validation,
    )


def test_missing_split_raises_file_not_found(tmp_path):
    store = ActivationStore(str(tmp_path / "activations"))

    with pytest.raises(FileNotFoundError):
        store.load_activations("discovery")


def test_save_rejects_non_matrix_activations(tmp_path):
    store = ActivationStore(str(tmp_path / "activations"))

    with pytest.raises(
        ValueError,
        match="2-dimensional",
    ):
        store.save_activations(
            "discovery",
            torch.ones(2, 3, 4),
        )


def test_s3_uri_is_rejected():
    with pytest.raises(
        ValueError,
        match="local-only",
    ):
        ActivationStore(
            "s3://bucket/activations"
        )


def test_accepts_path_object(tmp_path):
    location = tmp_path / "activations"

    store = ActivationStore(location)

    expected = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(4, 3)

    store.save_activations(
        "discovery",
        expected,
    )

    actual = store.load_activations(
        "discovery"
    )

    assert torch.equal(
        actual,
        expected,
    )
