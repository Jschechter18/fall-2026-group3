import pytest
import torch
from torch import nn

from mas_sae.activations.capture import MultiSiteCapture


class TinyMultiLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


def test_multi_site_capture_collects_all_sites():
    model = TinyMultiLayerModel()

    x = torch.randn(1, 3, 4)

    with MultiSiteCapture(
        model,
        ["layers.0", "layers.1"],
    ) as capture:
        model(x)

    activations = capture.activations

    assert set(activations) == {
        "layers.0",
        "layers.1",
    }

    assert activations["layers.0"].shape == (1, 4)
    assert activations["layers.1"].shape == (1, 4)


def test_multi_site_capture_rejects_empty_sites():
    model = TinyMultiLayerModel()

    with pytest.raises(ValueError):
        MultiSiteCapture(model, [])
