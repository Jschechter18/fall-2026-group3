"""Train/validation activation splits, batching, and feature lookup (cached or encoded live)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.artifacts import sha256_file

if TYPE_CHECKING:
    import argparse

    from mas_sae.sae.pretrained import JumpReLUSAE

SPLITS = ('train', 'validation')                   # the only activation files ever opened


@dataclass
class Split:
    """One activation split (train or validation) and everything derived from it."""
    name: str
    activations: torch.Tensor   # (n_examples, d_model), kept on the CPU; batches move to the device
    path: Path
    sha256: str                 # of the activation file: provenance, and part of the cache key
    mean: torch.Tensor          # float64 per-dimension mean, the baseline for variance explained
    # Validation only, if --probe-config is given: which rows the behaviour probe uses (see probe_cohort_rows).
    probe_cohort: np.ndarray | None = None
    probe_cohort_mean: torch.Tensor | None = None
    # Decoder mode only: the frozen encoder's output for every row, memory-mapped from disk.
    cached_features: np.ndarray | None = None
    cache_path: Path | None = None


def load_splits(args: 'argparse.Namespace', model: 'JumpReLUSAE') -> dict[str, Split]:
    """Load and check the train and validation activations. Returns {'train': Split, 'validation': Split}."""
    store = ActivationStore(args.activations_dir)
    splits = {}
    for name in SPLITS:
        filename = args.activation_file.format(split=name)
        if Path(filename).name != filename or not filename.endswith('.pt'):
            raise ValueError('activation-file must be a filename pattern ending in .pt.')
        path = args.activations_dir / filename
        activations = store.load_activations(Path(filename).stem)
        if len(activations) == 0 or activations.shape[1] != model.input_dim or not torch.isfinite(activations).all():
            raise ValueError(f'Invalid activation shape or nonfinite values: {path}')
        splits[name] = Split(name, activations, path, sha256_file(path), mean=activations.double().mean(dim=0))
    validation = splits['validation']
    validation.probe_cohort = probe_cohort_rows(args.probe_config, args, len(validation.activations))
    if validation.probe_cohort is not None:
        validation.probe_cohort_mean = validation.activations[validation.probe_cohort].double().mean(dim=0)
    return splits


def probe_cohort_rows(config_path: Path | None, args: 'argparse.Namespace',
                      n_validation: int) -> np.ndarray | None:
    """Boolean mask over validation rows: the examples the behaviour probe is evaluated on.

    These are the validation rows with usable behaviour labels, as chosen by the feature-selection
    config. Reconstruction is reported on them separately, because those are the examples the probe
    actually sees. They are never used for training or for choosing the checkpoint. Returns None
    without a config.
    """
    if config_path is None:
        return None
    from mas_sae.probe import feature_selection, layer_selection
    config = feature_selection.FeatureSelectionConfig.from_yaml(config_path)
    expected_dir = Path(config.capstone_data_root) / config.activations_dir / f'layer_{config.layer:02d}'
    if (expected_dir.resolve() != args.activations_dir.resolve() or config.layer != args.layer
            or config.activation_file != args.activation_file
            or config.fit_partition != 'train' or config.eval_partition != 'validation'):
        raise ValueError('Probe config must use the same layer, activation files, and train/validation partitions.')
    selected, _, columns = layer_selection.load_inputs(config.layer_selection_config())
    validation = selected[selected[columns.partition] == 'validation']
    if set(validation['source_split']) != {'validation'}:
        raise ValueError('Expected prepared, partitioned validation activation rows.')
    rows = validation['activation_row'].to_numpy()
    if len(rows) == 0 or rows.min() < 0 or rows.max() >= n_validation:
        raise ValueError('Probe cohort activation indices are out of range.')
    mask = np.zeros(n_validation, dtype=bool)
    mask[rows] = True
    return mask


def iterate_batches(activations: torch.Tensor, batch_size: int, device: str | torch.device,
                    shuffle: bool = False) -> Iterator[tuple[np.ndarray, torch.Tensor]]:
    """Yield (row indices, batch on device). Shuffling uses the torch RNG, so --seed controls it."""
    order = torch.randperm(len(activations)).numpy() if shuffle else np.arange(len(activations))
    for start in range(0, len(activations), batch_size):
        rows = order[start:start + batch_size]
        yield rows, activations[rows].to(device).float()


def features_for(model: 'JumpReLUSAE', split: Split, rows: np.ndarray,
                 batch: torch.Tensor) -> torch.Tensor:
    """Encoder output for a batch: read from the cache if the split has one (decoder mode), else encoded now."""
    if split.cached_features is not None:
        # Only these rows are read from the memory-mapped file.
        return torch.from_numpy(np.array(split.cached_features[rows], copy=True)).to(batch.device)
    with torch.no_grad():
        return model.encoder(batch)
