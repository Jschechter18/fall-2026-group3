import pytest
import torch

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


INPUT_DIM = 8
HIDDEN_DIM = 6
LATENT_DIM = 12


@pytest.fixture
def model() -> SparseAutoencoder:
    torch.manual_seed(0)
    return SparseAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )


@pytest.mark.parametrize("input_shape", [(4, INPUT_DIM), (2, 5, INPUT_DIM)])
def test_encoder_preserves_leading_dimensions_and_uses_latent_dimension(
    model: SparseAutoencoder,
    input_shape: tuple[int, ...],
) -> None:
    activations = torch.randn(input_shape)

    sparse_features = model.encoder(activations)

    assert sparse_features.shape == (*input_shape[:-1], LATENT_DIM)


def test_encoder_features_are_nonnegative(model: SparseAutoencoder) -> None:
    activations = torch.randn(4, INPUT_DIM)

    sparse_features = model.encoder(activations)

    assert torch.all(sparse_features >= 0)


def test_decoder_projects_features_back_to_input_dimension(
    model: SparseAutoencoder,
) -> None:
    sparse_features = torch.randn(4, LATENT_DIM)

    reconstructed_activations = model.decoder(sparse_features)

    assert reconstructed_activations.shape == (4, INPUT_DIM)


def test_forward_matches_separate_encode_and_decode_calls(
    model: SparseAutoencoder,
) -> None:
    activations = torch.randn(4, INPUT_DIM)

    sparse_features, reconstructed_activations = model(activations)
    expected_features = model.encoder(activations)
    expected_reconstruction = model.decoder(expected_features)

    assert torch.equal(sparse_features, expected_features)
    assert torch.equal(reconstructed_activations, expected_reconstruction)


def test_forward_supports_backpropagation(model: SparseAutoencoder) -> None:
    activations = torch.randn(4, INPUT_DIM)
    sparse_features, reconstructed_activations = model(activations)
    loss = torch.nn.functional.mse_loss(reconstructed_activations, activations)
    loss = loss + sparse_features.abs().mean()

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
