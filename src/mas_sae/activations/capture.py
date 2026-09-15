from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ActivationCapture:
    """Capture one activation vector from a named model module."""

    def __init__(
        self,
        model: nn.Module,
        module_name: str,
    ) -> None:
        self.model = model
        self.module_name = module_name
        self.activation: torch.Tensor | None = None
        self._handle: RemovableHandle | None = None

    @staticmethod
    def _extract_hidden_state(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        if (
            isinstance(output, (tuple, list))
            and output
            and isinstance(output[0], torch.Tensor)
        ):
            return output[0]

        raise TypeError(
            "Hooked module output does not contain a tensor hidden state."
        )

    def _hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        _ = module, inputs

        # generate() calls decoder layers repeatedly.
        # Keep only the first call: the prompt/prefill pass.
        if self.activation is not None:
            return

        hidden = self._extract_hidden_state(output)

        if hidden.ndim != 3:
            raise ValueError(
                "Expected hidden states with shape "
                "(batch, sequence, hidden_dim)."
            )

        # Keep one activation per example:
        # the final prompt-token representation.
        self.activation = (
            hidden[:, -1, :]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )

    def __enter__(self) -> "ActivationCapture":
        modules = dict(self.model.named_modules())

        if self.module_name not in modules:
            raise ValueError(
                f"Module {self.module_name!r} was not found in the model."
            )

        self.activation = None

        self._handle = modules[
            self.module_name
        ].register_forward_hook(self._hook)

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class MultiSiteCapture:
    """Capture several activation sites during one model forward pass."""

    def __init__(
        self,
        model: nn.Module,
        module_names: list[str],
    ) -> None:
        if not module_names:
            raise ValueError(
                "module_names must contain at least one module."
            )

        self._captures = [
            ActivationCapture(model, module_name)
            for module_name in module_names
        ]

        self._stack: ExitStack | None = None

    def __enter__(self) -> "MultiSiteCapture":
        self._stack = ExitStack()

        for capture in self._captures:
            self._stack.enter_context(capture)

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    @property
    def activations(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}

        for capture in self._captures:
            if capture.activation is None:
                raise RuntimeError(
                    f"No activation was captured for "
                    f"{capture.module_name!r}."
                )

            result[capture.module_name] = capture.activation

        return result
