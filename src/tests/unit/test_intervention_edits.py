"""The intervention edits: size exactly r, signs, raw-neuron signs, "both", reconstruction, diagnostics."""

import pytest
import torch

from mas_sae.intervention import edits
from mas_sae.sae.pretrained import JumpReLUSAE
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder

D, WIDTH = 8, 20


def _jump_relu(seed=0):
    g = torch.Generator().manual_seed(seed)
    return JumpReLUSAE(w_enc=torch.randn(D, WIDTH, generator=g), b_enc=torch.zeros(WIDTH),
                       w_dec=torch.randn(WIDTH, D, generator=g), b_dec=torch.randn(D, generator=g),
                       threshold=torch.full((WIDTH,), 0.5))


def _x(n=5, seed=1):
    return torch.randn(n, D, generator=torch.Generator().manual_seed(seed)) * 3.0


def _sets(sae=None):
    sae = sae or _jump_relu()
    toward = edits.set_direction(edits.sae_feature_directions(sae, [0, 1, 2]))
    away = edits.set_direction(edits.sae_feature_directions(sae, [5, 6, 7]))
    return toward, away


def test_sae_feature_direction_is_the_unit_decoder_row():
    sae = _jump_relu()
    d = edits.sae_feature_directions(sae, [3, 11])
    assert torch.allclose(d, sae.w_dec[[3, 11]] / sae.w_dec[[3, 11]].norm(dim=1, keepdim=True), atol=1e-6)
    assert torch.allclose(d.norm(dim=1), torch.ones(2))


def test_custom_sae_direction_includes_input_scale_and_ignores_bias():
    sae = SparseAutoencoder(input_dim=D, hidden_dim=WIDTH, latent_dim=WIDTH)
    sae.input_scale.fill_(4.0)
    torch.nn.init.normal_(sae.decoder_layer[0].bias)
    column = sae.decoder_layer[0].weight[:, 7]                  # Linear stores (out, in): column 7
    assert torch.allclose(edits.sae_feature_directions(sae, [7])[0], column / column.norm(), atol=1e-6)


def test_raw_neuron_direction_is_the_signed_basis_vector():
    d = edits.raw_feature_directions([2, 5], signs=[1, -1], width=D)
    expected = torch.zeros(2, D)
    expected[0, 2], expected[1, 5] = 1.0, -1.0
    assert torch.equal(d, expected)
    u = edits.set_direction(d)
    assert torch.allclose(u[[2, 5]], torch.tensor([1.0, -1.0]) / 2 ** 0.5)
    with pytest.raises(ValueError, match="sign"):
        edits.raw_feature_directions([2, 5], signs=[1, 0], width=D)


def test_set_direction_is_the_normalized_sum():
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(edits.set_direction(d), torch.tensor([1.0, 1.0]) / 2 ** 0.5)


@pytest.mark.parametrize("arm", edits.ARMS)
@pytest.mark.parametrize("size", [0.02, 0.1])
def test_every_arm_has_size_exactly_r(arm, size):
    x = _x()
    toward, away = _sets()
    delta = edits.edit_delta(arm, x, toward, away, size)
    assert torch.allclose(delta.norm(dim=1) / x.norm(dim=1), torch.full((len(x),), size), atol=1e-6)


def test_arm_signs():
    x = _x()
    toward, away = _sets()
    up = edits.edit_delta("increase", x, toward, away, 0.05)
    down = edits.edit_delta("decrease", x, toward, away, 0.05)
    assert (up @ toward > 0).all() and torch.allclose(up / up.norm(dim=1, keepdim=True), toward.expand_as(up))
    assert (down @ away < 0).all() and torch.allclose(down / down.norm(dim=1, keepdim=True), -away.expand_as(down))


def test_increase_and_decrease_of_the_same_set_are_mirror_images():
    x = _x()
    toward, _ = _sets()
    assert torch.allclose(edits.edit_delta("increase", x, toward, toward, 0.05),
                          -edits.edit_delta("decrease", x, toward, toward, 0.05))


def test_both_is_one_combined_edit_of_size_r_not_2r():
    x = _x()
    toward, away = _sets()
    delta = edits.edit_delta("both", x, toward, away, 0.05)
    expected = 0.05 * x.norm(dim=1, keepdim=True) * edits.normalize(toward - away)
    assert torch.allclose(delta, expected)
    # it moves toward one set and away from the other
    assert (delta @ toward > 0).all() and (delta @ away < 0).all()


def test_decrease_is_never_a_no_op_even_when_no_feature_fires():
    sae = _jump_relu()
    x = torch.zeros(1, D) - 50.0 * edits.set_direction(edits.sae_feature_directions(sae, [5, 6, 7]))
    assert x.norm() > 0
    _, away = _sets(sae)
    assert edits.edit_delta("decrease", x, away, away, 0.05).norm() > 0


def test_unknown_arm_is_refused():
    toward, away = _sets()
    with pytest.raises(ValueError, match="Unknown arm"):
        edits.edit_delta("ablate", _x(), toward, away, 0.05)


def test_reconstruction_splice_is_decode_of_encode():
    sae = _jump_relu()
    x = _x()
    assert torch.allclose(edits.reconstruction_splice(sae, x), sae.decoder(sae.encoder(x)))


def test_diagnostics_for_an_sae_edit():
    sae = _jump_relu()
    x = _x(n=1)[0]
    toward, away = _sets(sae)
    delta = edits.edit_delta("both", x, toward, away, 0.05)
    z = sae.encoder(x[None])[0]
    edited = [0, 1, 2, 5, 6, 7]
    diag = edits.diagnose(x, delta, edited, z, sae.decoder(z[None])[0])
    assert diag.applied_relative_size == pytest.approx(0.05, abs=1e-6)
    assert diag.edited_features_active == int((z[edited] > 0).sum())
    assert diag.reconstruction_error == pytest.approx(
        float((x - sae.decoder(z[None])[0]).norm() / x.norm()), rel=1e-5)


def test_diagnostics_for_raw_activations_have_no_firing_or_reconstruction():
    x = _x(n=1)[0]
    u = edits.set_direction(edits.raw_feature_directions([1, 2], [1, 1], D))
    diag = edits.diagnose(x, edits.edit_delta("increase", x, u, u, 0.02), [1, 2])
    assert diag.applied_relative_size == pytest.approx(0.02, abs=1e-6)
    assert diag.edited_features_active is None and diag.reconstruction_error is None


def test_set_cosine():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([3.0, 3.0])
    assert edits.set_cosine(a, b) == pytest.approx(2 ** -0.5)
    assert edits.set_cosine(a, -a) == pytest.approx(-1.0)
