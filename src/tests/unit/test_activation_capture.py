import pytest
import torch
from torch import nn

from mas_sae.activations.capture import ActivationCapture


class ToyBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = ToyBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TupleBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x + 2, x.mean()


class TupleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = TupleBlock()

    def forward(self, x: torch.Tensor):
        return self.block(x)


def test_capture_uses_last_sequence_token():
    model = ToyModel()

    x = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 3, 4)

    with ActivationCapture(model, "block") as capture:
        model(x)

    expected = (x + 1)[:, -1, :]

    assert capture.activation is not None
    assert capture.activation.shape == (2, 4)
    assert torch.equal(capture.activation, expected)


def test_capture_handles_tuple_output():
    model = TupleModel()

    x = torch.ones(1, 3, 4)

    with ActivationCapture(model, "block") as capture:
        model(x)

    assert capture.activation is not None
    assert torch.equal(
        capture.activation,
        (x + 2)[:, -1, :],
    )


def test_capture_keeps_only_first_forward_call():
    model = ToyModel()

    first = torch.zeros(1, 3, 4)
    second = torch.ones(1, 3, 4) * 10

    with ActivationCapture(model, "block") as capture:
        model(first)
        model(second)

    assert capture.activation is not None
    assert torch.equal(
        capture.activation,
        (first + 1)[:, -1, :],
    )


def test_capture_is_moved_to_cpu_float32():
    model = ToyModel()

    x = torch.ones(
        1,
        2,
        4,
        dtype=torch.float64,
    )

    with ActivationCapture(model, "block") as capture:
        model(x)

    assert capture.activation is not None
    assert capture.activation.device.type == "cpu"
    assert capture.activation.dtype == torch.float32


def test_unknown_module_raises_value_error():
    model = ToyModel()

    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        with ActivationCapture(model, "missing"):
            pass
