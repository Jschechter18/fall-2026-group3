from io import BytesIO

import torch

from mas_sae.data.activation_store import ActivationStore


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(
        self,
        *,
        Bucket,
        Key,
        Body,
    ):
        self.objects[(Bucket, Key)] = Body

    def get_object(
        self,
        *,
        Bucket,
        Key,
    ):
        return {
            "Body": BytesIO(
                self.objects[(Bucket, Key)]
            )
        }


def test_s3_round_trip():
    client = FakeS3Client()

    store = ActivationStore(
        "s3://test-bucket/issue14",
        s3_client=client,
    )

    expected = torch.randn(
        3,
        2560,
    )

    store.save_activations(
        "pilot",
        expected,
    )

    actual = store.load_activations(
        "pilot",
    )

    assert torch.equal(
        expected,
        actual,
    )


def test_s3_prefix_creates_expected_key():
    client = FakeS3Client()

    store = ActivationStore(
        "s3://test-bucket/project/activations",
        s3_client=client,
    )

    expected = torch.randn(
        2,
        4,
    )

    store.save_activations(
        "train",
        expected,
    )

    assert (
        "test-bucket",
        "project/activations/train.pt",
    ) in client.objects
