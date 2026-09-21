from __future__ import annotations

from pathlib import Path

import torch


class ActivationStore:
    """Persist and load 2-D activation matrices as <split>.pt files in a local directory."""

    def __init__(
        self,
        location: str | Path = "data/activations",
    ) -> None:
        location_str = str(location)

        if location_str.startswith("s3://"):
            raise ValueError(
                "ActivationStore is local-only; pass a directory path "
                "such as data/activations/<run>."
            )

        self.store_location = location_str
        self._local_path = Path(location)

    @staticmethod
    def _validate(
        activations: torch.Tensor,
    ) -> None:
        if not isinstance(
            activations,
            torch.Tensor,
        ):
            raise TypeError(
                "Activations must be a torch.Tensor."
            )

        if activations.ndim != 2:
            raise ValueError(
                "Activations must be a 2-dimensional tensor "
                "with shape (num_vectors, input_dim)."
            )

    def _split_path(
        self,
        split: str,
    ) -> Path:
        return self._local_path / f"{split}.pt"

    def save_activations(
        self,
        split: str,
        activations: torch.Tensor,
    ) -> None:
        self._validate(activations)

        activations = (
            activations
            .detach()
            .cpu()
        )

        path = self._split_path(split)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            activations,
            path,
        )

    def load_activations(
        self,
        split: str,
    ) -> torch.Tensor:
        path = self._split_path(split)

        if not path.is_file():
            raise FileNotFoundError(
                f"No activations found for split "
                f"{split!r} at {path}."
            )

        activations = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(
            activations,
            torch.Tensor,
        ):
            raise TypeError(
                f"Stored activations for split "
                f"{split!r} are not a tensor."
            )

        self._validate(activations)

        return activations
