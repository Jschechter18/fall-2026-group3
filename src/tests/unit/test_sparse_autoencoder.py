import pytest
import torch

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder


INPUT_DIM = 8
HIDDEN_DIM = 6
LATENT_DIM = 12


@pytest.mark.parametrize("shape", [(3, INPUT_DIM), (2, 3, INPUT_DIM)])
def test_topk_selects_largest_positive_features_per_example(shape):
    model = SparseAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, "topk", 3)
    x = torch.randn(shape)
    raw = model.encoder_layer(x / model.input_scale)
    expected = raw.clone()
    cutoff_indices = raw.topk(LATENT_DIM - 3, largest=False, dim=-1).indices
    expected.scatter_(-1, cutoff_indices, 0)
    actual = model.encoder(x)
    assert actual.shape == raw.shape
    assert torch.equal(actual, expected)
    assert ((actual > 0).sum(dim=-1) <= 3).all()
    model(x)[1].square().mean().backward()
    assert model.encoder_layer[0].weight.grad is not None
    assert torch.isfinite(model.encoder_layer[0].weight.grad).all()


@pytest.mark.parametrize("k", [None, 0, -1, LATENT_DIM + 1, 2.5, True])
def test_topk_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="top_k"):
        SparseAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, "topk", k)


def test_invalid_sparsity_mode():
    with pytest.raises(ValueError, match="sparsity_mode"):
        SparseAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, "invalid")


def test_topk_full_dictionary_matches_l1():
    model = SparseAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, "topk", LATENT_DIM)
    x = torch.randn(3, INPUT_DIM)
    assert torch.equal(model.encoder(x), model.encoder_layer(x / model.input_scale))


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


def test_input_scale_preserves_features_and_restores_output_units(model: SparseAutoencoder) -> None:
    activations = torch.randn(4, INPUT_DIM)
    features, reconstruction = model(activations)
    model.input_scale.fill_(3.0)

    scaled_features, scaled_reconstruction = model(activations * 3.0)

    assert torch.allclose(scaled_features, features, atol=1e-6)
    assert torch.allclose(scaled_reconstruction, reconstruction * 3.0, atol=1e-6)


def test_checkpoint_restores_scale_and_outputs(model: SparseAutoencoder, tmp_path) -> None:
    model.input_scale.fill_(7.0)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = SparseAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    restored.load_state_dict(torch.load(path, weights_only=True))
    activations = torch.randn(4, INPUT_DIM)

    assert restored.input_scale.item() == 7.0
    assert "input_scale" not in dict(restored.named_parameters())
    for expected, actual in zip(model(activations), restored(activations)):
        assert torch.equal(actual, expected)


def test_decoder_normalization_constrains_columns_and_preserves_bias(model: SparseAutoencoder) -> None:
    weight = model.decoder_layer[0].weight
    assert torch.allclose(weight.norm(dim=0), torch.ones(LATENT_DIM), atol=1e-6)
    bias = model.decoder_layer[0].bias.detach().clone()
    with torch.no_grad():
        weight.mul_(torch.arange(1, LATENT_DIM + 1))

    model.normalize_decoder_weights()

    assert model.decoder_layer[0].weight is weight
    assert torch.allclose(weight.norm(dim=0), torch.ones(LATENT_DIM), atol=1e-6)
    assert torch.equal(model.decoder_layer[0].bias, bias)


@pytest.mark.parametrize("mode", ["l1", "topk"])
@pytest.mark.parametrize("shape", [(3, INPUT_DIM), (2, 3, INPUT_DIM)])
def test_training_forward_matches_inference_with_one_encoder_call(mode, shape):
    model = SparseAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, mode, 3)
    model.input_scale.fill_(3.0)
    x = torch.randn(shape)
    expected = model(x)
    calls = []
    handle = model.encoder_layer.register_forward_hook(lambda *args: calls.append(1))
    try:
        features, reconstruction, pre_topk = model.forward_with_pre_topk(x)
    finally:
        handle.remove()
    assert len(calls) == 1
    assert torch.equal(features, expected[0])
    assert torch.equal(reconstruction, expected[1])
    assert torch.equal(pre_topk, model.encode_pre_topk(x))
    pre_topk.sum().backward()
    assert model.encoder_layer[0].weight.grad is not None


def test_select_topk_values_and_gradients():
    features = torch.tensor([[1., 4., 2., 3.]], requires_grad=True)
    selected = SparseAutoencoder.select_topk(features, 2)
    assert selected.tolist() == [[0., 4., 0., 3.]]
    selected.sum().backward()
    assert features.grad.tolist() == [[0., 1., 0., 1.]]
    assert torch.equal(SparseAutoencoder.select_topk(features, 0), torch.zeros_like(features))
    assert torch.equal(SparseAutoencoder.select_topk(features, 4), features)


@pytest.mark.parametrize("k", [-1, 5, True, 1.5])
def test_select_topk_rejects_invalid_counts(k):
    with pytest.raises(ValueError, match="k must"):
        SparseAutoencoder.select_topk(torch.ones(2, 4), k)


def test_bias_free_decoder_uses_normalized_units_and_shared_weights(model):
    model.input_scale.fill_(7.0)
    with torch.no_grad():
        model.decoder_layer[0].bias.fill_(2.0)
    features = torch.randn(2, 3, LATENT_DIM, requires_grad=True)
    decoded = model.decode_without_bias(features)
    assert torch.allclose(model.decoder(features), (decoded + 2.0) * 7.0, atol=1e-5)
    decoded.square().sum().backward()
    assert features.grad is not None
    assert model.decoder_layer[0].weight.grad is not None
    assert model.decoder_layer[0].bias.grad is None
