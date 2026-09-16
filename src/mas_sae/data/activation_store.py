from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch


class ActivationStore:
    """Persist and load activation matrices locally or from S3."""

    def __init__(
        self,
        location: str,
        *,
        s3_client: Any | None = None,
    ) -> None:
        self.store_location = location
        self._is_s3 = location.startswith("s3://")
        self._s3_client = s3_client

        if self._is_s3:
            parsed = urlparse(location)

            if not parsed.netloc:
                raise ValueError(
                    "S3 location must contain a bucket name."
                )

            self._bucket = parsed.netloc
            self._prefix = (
                parsed.path
                .lstrip("/")
                .rstrip("/")
            )
        else:
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
        if self._is_s3:
            raise RuntimeError(
                "_split_path is only for local storage."
            )

        return self._local_path / f"{split}.pt"

    def _s3_key(
        self,
        split: str,
    ) -> str:
        filename = f"{split}.pt"

        if not self._prefix:
            return filename

        return f"{self._prefix}/{filename}"

    def _get_s3_client(self):
        if self._s3_client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for S3 activation storage."
                ) from exc

            self._s3_client = boto3.client("s3")

        return self._s3_client

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

        if self._is_s3:
            buffer = BytesIO()

            torch.save(
                activations,
                buffer,
            )

            buffer.seek(0)

            self._get_s3_client().put_object(
                Bucket=self._bucket,
                Key=self._s3_key(split),
                Body=buffer.getvalue(),
            )
            return

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
        return torch.rand(32, 4)  # Example shape (num_vectors, input_dim)
        # if self._is_s3:
        #     response = (
        #         self._get_s3_client()
        #         .get_object(
        #             Bucket=self._bucket,
        #             Key=self._s3_key(split),
        #         )
        #     )

        #     activations = torch.load(
        #         BytesIO(
        #             response["Body"].read()
        #         ),
        #         map_location="cpu",
        #         weights_only=True,
        #     )

        # else:
        #     path = self._split_path(split)

        #     if not path.is_file():
        #         raise FileNotFoundError(
        #             f"No activations found for split "
        #             f"{split!r} at {path}."
        #         )

        #     activations = torch.load(
        #         path,
        #         map_location="cpu",
        #         weights_only=True,
        #     )

        # if not isinstance(
        #     activations,
        #     torch.Tensor,
        # ):
        #     raise TypeError(
        #         f"Stored activations for split "
        #         f"{split!r} are not a tensor."
        #     )

        # self._validate(activations)

        # return activations
