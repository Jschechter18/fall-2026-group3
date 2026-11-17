"""Epoch-boundary state for continuing an SAE trial without restarting it."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from mas_sae.sae.callbacks.checkpointing import CheckpointEvaluatorCallback
    from mas_sae.sae.callbacks.early_stopping import EarlyStoppingCallback
    from mas_sae.sae.model_runner import ModelRunner


def save_training_state(path: str | Path, runner: ModelRunner,
                        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                        checkpoint_evaluator: CheckpointEvaluatorCallback,
                        early_stopping: EarlyStoppingCallback, history: list[dict],
                        stopped: bool) -> None:
    """Write everything needed to resume after this epoch to `path` (last_checkpoint.pt).

    Saves model, optimizer and scheduler state, the best-checkpoint and early-stopping counters,
    the epoch history and stopped flag, the runner's activity counters, the RNG states (Python,
    NumPy, torch, and CUDA/MPS when available) and the device type. The file is written to a
    .pt.tmp sibling and then renamed over `path`, so `path` is never left half-written.
    """
    state = {
        "model": runner.model.state_dict(),
        "optimizer": runner.optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": checkpoint_evaluator.best_loss,
        "early_best_loss": early_stopping.best_loss,
        "early_counter": early_stopping.counter,
        "stopped": stopped,
        "history": history,
        "training_examples_seen": runner.training_examples_seen,
        "last_fired_example": runner.last_fired_example,
        **capture_rng_state(),
        "device_type": next(runner.model.parameters()).device.type,
    }
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(state: dict, runner: ModelRunner,
                           scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                           checkpoint_evaluator: CheckpointEvaluatorCallback,
                           early_stopping: EarlyStoppingCallback) -> None:
    """Load a save_training_state() dict back into freshly built objects, including RNG state.

    The epoch history and stopped flag are not restored here; train_sae reads them from `state`.
    """
    runner.model.load_state_dict(state["model"])
    runner.optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    checkpoint_evaluator.best_loss = state["best_loss"]
    early_stopping.best_loss = state["early_best_loss"]
    early_stopping.counter = state["early_counter"]
    runner.training_examples_seen = state["training_examples_seen"]
    runner.last_fired_example = state["last_fired_example"]
    restore_rng_state(state)


def capture_rng_state() -> dict:
    """Capture RNG state in a weights-only-loadable representation."""
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):
        raise TypeError("Expected NumPy legacy RNG state as a tuple.")
    return {
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Restore RNG state captured at a committed training boundary."""
    random.setstate(state["python_rng"])
    numpy_state = state["numpy_rng"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if state["mps_rng"] is not None:
        torch.mps.set_rng_state(state["mps_rng"])
