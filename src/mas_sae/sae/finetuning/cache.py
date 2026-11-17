"""On-disk cache of a frozen encoder's features (decoder-only fine-tuning).

The frozen encoder's output for 80k examples at width 16k is ~5 GB, so it is written to disk once
and memory-mapped afterwards. The cache directory is named by a hash of everything that determines
its contents (encoder tensors + activation file hash), so a stale cache can never be reused.
A build writes to a hidden temporary directory and is renamed into place only when complete, and a
lock file stops two processes building the same cache at once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm.auto import tqdm

from mas_sae.experiments.artifacts import sha256_json, write_json_atomic

if TYPE_CHECKING:
    from mas_sae.sae.pretrained import JumpReLUSAE

    from .data import Split

logger = logging.getLogger(__name__)

ENCODER_KEYS = ('w_enc', 'b_enc', 'threshold')     # the tensors that determine the features
BYTES_PER_FLOAT32 = 4
CACHE_DISK_HEADROOM_BYTES = 1024 ** 2


def encoder_digest(model: JumpReLUSAE) -> str:
    """SHA-256 of the encoder tensors (names, shapes, dtypes and bytes). Equal digests mean identical features."""
    digest = hashlib.sha256()
    for name in ENCODER_KEYS:
        tensor = getattr(model, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def attach_feature_cache(split: Split, model: JumpReLUSAE, cache_dir: str | Path, batch_size: int,
                         device: str | torch.device) -> None:
    """Point split.cached_features at the (possibly newly built) on-disk encoder output."""
    identity = {'split': split.name, 'activation_sha256': split.sha256}
    split.cached_features, split.cache_path = cached_features(
        model, split.activations, identity, cache_dir, batch_size, device)


def cached_features(model: JumpReLUSAE, activations: torch.Tensor, identity: dict[str, str],
                    cache_dir: str | Path, batch_size: int,
                    device: str | torch.device) -> tuple[np.ndarray, Path]:
    """Return (memory-mapped float32 features, cache directory), building the cache if it doesn't exist.

    `identity` names the input (split name and activation file hash). The encoder digest is added to it,
    and the hash of the result is the cache directory name. Decoder weights do not affect the key.
    """
    identity = {**identity, 'encoder_sha256': encoder_digest(model), 'format': 1,
                'dtype': 'float32', 'shape': [len(activations), model.latent_dim]}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / sha256_json(identity)
    if not destination.exists():
        _build_cache(model, activations, identity, cache_dir, destination, batch_size, device)
    else:
        logger.info('Reusing encoder cache: %s', destination)
    return _open_cache(destination, identity), destination


def _open_cache(destination, identity):
    """Memory-map a finished cache, checking it really is the one `identity` describes."""
    saved = json.loads((destination / 'manifest.json').read_text())
    if saved['identity'] != identity:
        raise ValueError(f'Cache identity mismatch: {destination}')
    features = np.load(destination / 'features.npy', mmap_mode='r', allow_pickle=False)
    if list(features.shape) != identity['shape'] or features.dtype != np.float32:
        raise ValueError(f'Invalid feature cache: {destination}')
    return features


def _build_cache(model, activations, identity, cache_dir, destination, batch_size, device):
    """Encode every row into a temporary directory, then rename it to `destination` in one step."""
    lock = destination.with_suffix('.lock')
    try:
        # O_EXCL makes creating the lock fail if another process already holds it.
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError as exc:
        raise RuntimeError(f'Cache build already locked: {lock}. If its process stopped, remove this lock.') from exc
    temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=cache_dir))
    try:
        size_bytes = len(activations) * model.latent_dim * BYTES_PER_FLOAT32
        logger.info('Building %s encoder cache (%.2f GB, float32)', identity['split'], size_bytes / 1e9)
        if shutil.disk_usage(cache_dir).free < size_bytes + CACHE_DISK_HEADROOM_BYTES:
            raise ValueError('Not enough free disk space for the encoder cache.')
        features = np.lib.format.open_memmap(temporary / 'features.npy', mode='w+', dtype=np.float32,
                                             shape=(len(activations), model.latent_dim))
        with torch.no_grad():
            for start in tqdm(range(0, len(activations), batch_size), desc=f"Cache {identity['split']}", unit='batch'):
                batch_features = model.encoder(activations[start:start + batch_size].to(device).float())
                if not torch.isfinite(batch_features).all():
                    raise ValueError('Encoder produced nonfinite features.')
                features[start:start + len(batch_features)] = batch_features.cpu().numpy()
        features.flush()
        del features   # close the memory map before the directory is renamed
        write_json_atomic(temporary / 'manifest.json', {'identity': identity})
        temporary.rename(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)   # nothing left after a successful rename
        lock.unlink(missing_ok=True)
