from pathlib import Path

import torch


class ActivationStore:
    """Persist and load activation matrices by dataset split."""

    def __init__(self, location: str):
        self.store_location = location
        self._is_s3 = location.startswith("s3://")

        if not self._is_s3:
            self._local_path = Path(location)

    def _split_path(self, split: str) -> Path:
        if self._is_s3:
            raise NotImplementedError(
                "S3 activation storage will be implemented separately."
            )

        return self._local_path / f"{split}.pt"

    def save_activations(
        self,
        split: str,
        activations: torch.Tensor,
    ) -> None:
        if activations.ndim != 2:
            raise ValueError(
                "Activations must be a 2-dimensional tensor "
                "with shape (num_vectors, input_dim)."
            )

        path = self._split_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            activations.detach().cpu(),
            path,
        )

    def load_activations(self, split: str) -> torch.Tensor:
        path = self._split_path(split)

        if not path.is_file():
            raise FileNotFoundError(
                f"No activations found for split {split!r} at {path}."
            )

        activations = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(activations, torch.Tensor):
            raise TypeError(
                f"Stored activations for split {split!r} are not a tensor."
            )

        if activations.ndim != 2:
            raise ValueError(
                "Stored activations must be a 2-dimensional tensor."
            )

        return activations
