import random

import numpy as np
import torch

from mas_sae.experiments.reproducibility import seed_everything


def test_python_random_repeats():
    seed_everything(42)
    first = random.random()

    seed_everything(42)
    second = random.random()

    assert first == second


def test_numpy_random_repeats():
    seed_everything(42)
    first = np.random.rand()

    seed_everything(42)
    second = np.random.rand()

    assert first == second


def test_torch_random_repeats():
    seed_everything(42)
    first = torch.rand(4)

    seed_everything(42)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_different_seeds_differ():
    seed_everything(1)
    first = torch.rand(4)

    seed_everything(2)
    second = torch.rand(4)

    assert not torch.equal(first, second)
