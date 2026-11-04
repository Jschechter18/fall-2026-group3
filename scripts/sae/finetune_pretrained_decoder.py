"""Fine-tune a Gemma Scope SAE: the decoder only (default), or encoder and decoder together.

--train decoder: reuse immutable, disk-backed encoder features; only w_dec/b_dec update.
--train full:    w_enc, b_enc, threshold, w_dec, b_dec all update from the original weights, with
                 straight-through JumpReLU gradients and an L0 penalty that holds sparsity near the
                 original. Features are re-encoded every batch; no cache is used.

Example (from the repo root):
    PYTHONPATH=src python scripts/sae/finetune_pretrained_decoder.py \
        --sae-dir results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big \
        --activations-dir artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25 \
        --layer 25 --probe-config configs/feature_selection_mixed_final_001_layer25.yaml

Only train/validation are read. No model calls or probe retraining are performed.
Add --train full for the encoder-and-decoder checkpoint (written under results/sae/full_finetuning).
The output best/ directory is loadable by the existing kind: gemma_scope loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.artifacts import get_git_provenance, sha256_file, sha256_json, write_json_atomic
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.sae.pretrained import load_gemma_scope_sae
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)
ENCODER_KEYS = ('w_enc', 'b_enc', 'threshold')


def encoder_digest(model):
    digest = hashlib.sha256()
    for name in ENCODER_KEYS:
        tensor = getattr(model, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cached_features(model, X, identity, cache_dir, batch_size, device):
    """Build once, publish atomically, and memory-map float32 features on later runs.

    The key covers actual encoder tensors and activation file contents, not just paths.
    Decoder parameters do not affect the key. Incomplete caches are never published.
    """
    identity = {**identity, 'encoder_sha256': encoder_digest(model), 'format': 1,
                'dtype': 'float32', 'shape': [len(X), model.latent_dim]}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / sha256_json(identity)
    if dest.exists():
        saved = json.loads((dest / 'manifest.json').read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Cache identity mismatch: {dest}')
        features = np.load(dest / 'features.npy', mmap_mode='r', allow_pickle=False)
        if list(features.shape) != identity['shape'] or features.dtype != np.float32:
            raise ValueError(f'Invalid feature cache: {dest}')
        logger.info('Reusing encoder cache: %s', dest)
        return features, dest
    lock = dest.with_suffix('.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Cache build already locked: {lock}. If its process stopped, remove this lock.') from exc
    os.close(fd)
    temporary = Path(tempfile.mkdtemp(prefix='.building-', dir=cache_dir))
    try:
        size_gb = len(X) * model.latent_dim * 4 / 1e9
        logger.info('Building %s encoder cache (%.2f GB, float32)', identity['split'], size_gb)
        if shutil.disk_usage(cache_dir).free < len(X) * model.latent_dim * 4 + 1024**2:
            raise ValueError('Not enough free disk space for the encoder cache.')
        features = np.lib.format.open_memmap(temporary / 'features.npy', mode='w+', dtype=np.float32,
                                             shape=(len(X), model.latent_dim))
        with torch.no_grad():
            for start in tqdm(range(0, len(X), batch_size), desc=f"Cache {identity['split']}", unit='batch'):
                z = model.encoder(X[start:start + batch_size].to(device).float())
                if not torch.isfinite(z).all():
                    raise ValueError('Encoder produced nonfinite features.')
                features[start:start + len(z)] = z.cpu().numpy()
        features.flush()
        del features
        write_json_atomic(temporary / 'manifest.json', {'identity': identity})
        temporary.rename(dest)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        lock.unlink(missing_ok=True)
    return np.load(dest / 'features.npy', mmap_mode='r', allow_pickle=False), dest


def batches(X, Z, batch_size, device, shuffle=False, encode=None):
    """Yield (rows, x, z). z comes from the cache Z, else from encode(x), else is None."""
    indices = torch.randperm(len(X)).numpy() if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        rows = indices[start:start + batch_size]
        x = X[rows].to(device).float()
        if Z is not None:
            # Only this batch is materialized from the memory-mapped feature matrix.
            z = torch.from_numpy(np.array(Z[rows], copy=True)).to(device)
        elif encode is not None:
            with torch.no_grad():
                z = encode(x)
        else:
            z = None
        yield rows, x, z


@torch.no_grad()
def evaluate(model, X, Z, batch_size, device, mean, mask=None, cohort_mean=None):
    """Reconstruction metrics, mean L0 and dead-feature fraction. Z=None encodes live."""
    sse = baseline_sse = selected_sse = selected_baseline = 0.0
    selected_n = active_total = 0
    ever_active = torch.zeros(model.latent_dim, dtype=torch.bool, device=device)
    for rows, x, z in batches(X, Z, batch_size, device, encode=model.encoder):
        active = z > 0
        active_total += int(active.sum())
        ever_active |= active.any(dim=0)
        error = (model.decoder(z) - x).double().square().sum(dim=1)
        baseline = (x.double() - mean.to(device)).square().sum(dim=1)
        sse += error.sum().item()
        baseline_sse += baseline.sum().item()
        if mask is not None:
            chosen = torch.from_numpy(mask[rows]).to(device)
            selected_sse += error[chosen].sum().item()
            selected_baseline += (x[chosen].double() - cohort_mean.to(device)).square().sum().item()
            selected_n += int(chosen.sum())
    def metrics(error, reference, n):
        return {'n': n, 'mse': error / (n * X.shape[1]),
                'variance_explained': 1 - error / reference if reference > 0 else None}
    result = metrics(sse, baseline_sse, len(X))
    result['l0'] = active_total / len(X)
    result['dead_fraction'] = 1 - ever_active.float().mean().item()
    if mask is not None:
        result['probe_cohort'] = metrics(selected_sse, selected_baseline, selected_n)
    return result


def within_l0(l0, target, tolerance):
    """A checkpoint is eligible only if its mean L0 stays within a relative tolerance of the original."""
    return abs(l0 - target) <= tolerance * target


def default_bandwidth(X, batch_size):
    """Gemma Scope's 0.001 bandwidth, rescaled from unit-normalized to raw activations.

    Their training inputs were scaled so E||x|| = sqrt(d); ours are raw, so the equivalent
    bandwidth is 0.001 * E||x|| / sqrt(d).
    """
    total = sum(X[i:i + batch_size].double().norm(dim=1).sum().item() for i in range(0, len(X), batch_size))
    return 0.001 * total / len(X) / X.shape[1] ** 0.5


@torch.no_grad()
def feature_drift(original, adapted, X, batch_size, device):
    """How far each feature moved: activation correlation, active-set overlap and weight cosines.

    Returns (summary, per-feature arrays). Correlation is undefined (nan) for a feature that is
    constant, e.g. never active, under either encoder.
    """
    sums = {k: torch.zeros(original.latent_dim, dtype=torch.float64, device=device)
            for k in ('a', 'b', 'aa', 'bb', 'ab')}
    jaccard = 0.0
    for _, x, _ in batches(X, None, batch_size, device):
        a, b = original.encoder(x).double(), adapted.encoder(x).double()
        for key, value in (('a', a), ('b', b), ('aa', a * a), ('bb', b * b), ('ab', a * b)):
            sums[key] += value.sum(dim=0)
        A, B = a > 0, b > 0
        union = (A | B).sum(dim=1)
        overlap = (A & B).sum(dim=1).double() / union.clamp(min=1)
        jaccard += torch.where(union > 0, overlap, torch.ones_like(overlap)).sum().item()
    n = len(X)
    mean_a, mean_b = sums['a'] / n, sums['b'] / n
    var_a, var_b = sums['aa'] / n - mean_a ** 2, sums['bb'] / n - mean_b ** 2
    defined = (var_a > 0) & (var_b > 0)
    corr = torch.full_like(var_a, float('nan'))
    corr[defined] = ((sums['ab'] / n - mean_a * mean_b)[defined] / (var_a * var_b)[defined].sqrt()).clamp(-1, 1)
    cos = torch.nn.functional.cosine_similarity
    encoder_cos = cos(original.w_enc.double(), adapted.w_enc.double(), dim=0)
    decoder_cos = cos(original.w_dec.double(), adapted.w_dec.double(), dim=1)
    valid = corr[defined]
    summary = {
        'n_examples': n, 'n_features_with_defined_correlation': int(defined.sum()),
        'activation_correlation_median': valid.median().item() if len(valid) else None,
        'fraction_correlation_below_0.9': (valid < 0.9).double().mean().item() if len(valid) else None,
        'active_set_jaccard_mean': jaccard / n,
        'encoder_cosine_median': encoder_cos.median().item(), 'encoder_cosine_min': encoder_cos.min().item(),
        'decoder_cosine_median': decoder_cos.median().item(), 'decoder_cosine_min': decoder_cos.min().item(),
    }
    arrays = {'activation_correlation': corr, 'encoder_cosine': encoder_cos, 'decoder_cosine': decoder_cos,
              'threshold_ratio': adapted.threshold.double() / original.threshold.double()}
    return summary, {k: v.cpu().numpy() for k, v in arrays.items()}


def save_model(model, config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
              destination / 'params.safetensors')
    write_json_atomic(destination / 'config.json', config)


def probe_mask(config_path, args, n_validation):
    if config_path is None:
        return None
    from mas_sae.probe import feature_selection as fs, layer_selection as ls
    cfg = fs.FeatureSelectionConfig.from_yaml(config_path)
    expected = Path(cfg.capstone_data_root) / cfg.activations_dir / f'layer_{cfg.layer:02d}'
    if (expected.resolve() != args.activations_dir.resolve() or cfg.layer != args.layer
            or cfg.activation_file != args.activation_file
            or cfg.fit_partition != 'train' or cfg.eval_partition != 'validation'):
        raise ValueError('Probe config must use the same layer, activation files, and train/validation partitions.')
    selected, _, cols = ls.load_inputs(cfg.layer_selection_config())
    val = selected[selected[cols.partition] == 'validation']
    if set(val['source_split']) != {'validation'}:
        raise ValueError('Expected prepared, partitioned validation activation rows.')
    mask = np.zeros(n_validation, dtype=bool)
    rows = val['activation_row'].to_numpy()
    if len(rows) == 0 or rows.min() < 0 or rows.max() >= n_validation:
        raise ValueError('Probe cohort activation indices are out of range.')
    mask[rows] = True
    return mask


def run(args):
    for name in ('epochs', 'batch_size', 'patience'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive.')
    full = args.train == 'full'
    if args.train not in ('decoder', 'full'):
        raise ValueError("train must be 'decoder' or 'full'.")
    lr_enc = args.lr if args.lr_enc is None else args.lr_enc
    for name, value in (('lr', args.lr), ('lr_enc', lr_enc)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if args.l0_coef < 0 or args.l0_tolerance < 0:
        raise ValueError('l0_coef and l0_tolerance must be nonnegative.')
    if args.bandwidth is not None and not args.bandwidth > 0:
        raise ValueError('bandwidth must be positive.')
    output_dir = args.output_dir or Path(f'results/sae/{args.train}_finetuning')
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, config = load_gemma_scope_sae(args.sae_dir, expected_layer=args.layer)
    frozen_hash = encoder_digest(model)
    model.to(device)
    model.w_dec.requires_grad_(True)
    model.b_dec.requires_grad_(True)
    log_threshold = None
    if full:
        model.w_enc.requires_grad_(True)
        model.b_enc.requires_grad_(True)
        if not args.freeze_threshold:
            if not (model.threshold > 0).all():
                raise ValueError('Trainable thresholds are stored in log space and must be positive; '
                                 'use --freeze-threshold.')
            log_threshold = torch.nn.Parameter(model.threshold.detach().log())
    inputs, paths, hashes = {}, {}, {}
    store = ActivationStore(args.activations_dir)
    for split in ('train', 'validation'):
        filename = args.activation_file.format(split=split)
        if Path(filename).name != filename or not filename.endswith('.pt'):
            raise ValueError('activation-file must be a filename pattern ending in .pt.')
        path = args.activations_dir / filename
        X = store.load_activations(Path(filename).stem)
        if len(X) == 0 or X.shape[1] != model.input_dim or not torch.isfinite(X).all():
            raise ValueError(f'Invalid activation shape or nonfinite values: {path}')
        inputs[split], paths[split], hashes[split] = X, path, sha256_file(path)
    mask = probe_mask(args.probe_config, args, len(inputs['validation']))
    caches, cache_paths = {s: None for s in inputs}, {}
    if not full:  # in full mode the encoder changes, so cached features would be stale
        for split, X in inputs.items():
            caches[split], cache_paths[split] = cached_features(model, X,
                {'split': split, 'activation_sha256': hashes[split]}, args.cache_dir, args.batch_size, device)
    # Use each evaluation cohort's own mean for variance explained, identically before and after.
    means = {s: X.double().mean(dim=0) for s, X in inputs.items()}
    cohort_mean = inputs['validation'][mask].double().mean(dim=0) if mask is not None else None
    def measure():
        return {s: evaluate(model, inputs[s], caches[s], args.batch_size, device, means[s],
                            mask if s == 'validation' else None, cohort_mean) for s in inputs}
    original = measure()
    # Cheap bias-only comparator: optimal train residual-mean correction, no fitting on validation.
    original_bias = model.b_dec.detach().clone()
    residual = torch.zeros(model.input_dim, dtype=torch.float64, device=device)
    with torch.no_grad():
        for _, x, z in batches(inputs['train'], caches['train'], args.batch_size, device, encode=model.encoder):
            residual += (x - model.decoder(z)).double().sum(dim=0)
        model.b_dec.add_((residual / len(inputs['train'])).to(model.b_dec.dtype))
    bias_only = measure()
    with torch.no_grad():
        model.b_dec.copy_(original_bias)
    bandwidth = None
    if full:
        bandwidth = args.bandwidth or default_bandwidth(inputs['train'], args.batch_size)
        if original['train']['l0'] == 0:
            raise ValueError('Original SAE has no active features on train; the L0 penalty is undefined.')
    out = create_versioned_run_dir(output_dir)
    manifest = {'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'source_files_sha256': {str(args.sae_dir / n): sha256_file(args.sae_dir / n)
                                        for n in ('config.json', 'params.safetensors')},
                'activation_sha256': hashes, 'encoder_sha256': frozen_hash,
                'cache_paths': {s: str(p) for s, p in cache_paths.items()},
                'git': get_git_provenance(), 'script_sha256': sha256_file(Path(__file__)),
                'probe_config_sha256': sha256_file(args.probe_config) if args.probe_config else None,
                'bandwidth': bandwidth, 'lr_enc': lr_enc if full else None, 'status': 'running'}
    write_json_atomic(out / 'manifest.json', manifest)
    write_json_atomic(out / 'baseline.json', {'original': original, 'bias_only': bias_only})
    best_loss, best_epoch = original['validation']['mse'], 0
    l0_target = original['validation']['l0']
    save_model(model, config, out / 'best')
    groups = [{'params': [model.w_dec, model.b_dec], 'lr': args.lr}]
    if full:
        encoder_params = [model.w_enc, model.b_enc] + ([log_threshold] if log_threshold is not None else [])
        groups.append({'params': encoder_params, 'lr': lr_enc})
    optimizer = torch.optim.Adam(groups)
    # Full mode: both terms are relative to the original SAE on train, so l0_coef=1 trades a
    # 1% rise in L0 against a 1% fall in MSE.
    mse_scale, l0_scale = original['train']['mse'], original['train']['l0']
    history, stale = [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        count = (len(inputs['train']) + args.batch_size - 1) // args.batch_size
        totals = {'mse': 0.0, 'l0': 0.0, 'loss': 0.0}
        for _, x, z in tqdm(batches(inputs['train'], caches['train'], args.batch_size, device, shuffle=True),
                             total=count, desc=f'Epoch {epoch}/{args.epochs}', unit='batch'):
            optimizer.zero_grad(set_to_none=True)
            if full:
                threshold = log_threshold.exp() if log_threshold is not None else None
                z, l0 = model.encoder_ste(x, bandwidth, threshold)
                mse = torch.nn.functional.mse_loss(model.decoder(z), x)
                l0 = l0.mean()
                loss = mse / mse_scale + args.l0_coef * l0 / l0_scale
                totals['mse'] += mse.item() * len(x)
                totals['l0'] += l0.item() * len(x)
            else:
                loss = torch.nn.functional.mse_loss(model.decoder(z), x)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss.')
            totals['loss'] += loss.item() * len(x)
            loss.backward()
            optimizer.step()
            if log_threshold is not None:
                with torch.no_grad():
                    model.threshold.copy_(log_threshold.exp())
        model.eval()
        measured = measure()
        if not np.isfinite(measured['validation']['mse']):
            raise ValueError('Nonfinite validation loss.')
        eligible = within_l0(measured['validation']['l0'], l0_target, args.l0_tolerance)
        objective = {k: v / len(inputs['train']) for k, v in totals.items() if full or k == 'loss'}
        history.append({'epoch': epoch, **measured, 'train_objective': objective, 'l0_eligible': eligible})
        write_json_atomic(out / 'history.json', history)
        logger.info('Epoch %d validation MSE %.6g (original %.6g), L0 %.4g (original %.4g)%s',
                    epoch, measured['validation']['mse'], original['validation']['mse'],
                    measured['validation']['l0'], l0_target, '' if eligible else ' [outside L0 tolerance]')
        if eligible and measured['validation']['mse'] < best_loss:
            best_loss, best_epoch, stale = measured['validation']['mse'], epoch, 0
            save_model(model, config, out / 'best')
        else:
            stale += 1
        if stale >= args.patience:
            break
    if not full and encoder_digest(model) != frozen_hash:
        raise RuntimeError('Frozen encoder or thresholds changed.')
    best, _ = load_gemma_scope_sae(out / 'best', expected_layer=args.layer)
    model.load_state_dict(best.state_dict())
    final = measure()
    summary = {'train_mode': args.train, 'best_epoch': best_epoch, 'original': original,
               'bias_only': bias_only, 'adapted': final,
               'encoder_unchanged': encoder_digest(model) == frozen_hash,
               'l0_target': l0_target, 'l0_tolerance': args.l0_tolerance}
    if full:
        source, _ = load_gemma_scope_sae(args.sae_dir, expected_layer=args.layer)
        drift, arrays = feature_drift(source.to(device), model, inputs['validation'], args.batch_size, device)
        np.savez(out / 'feature_drift.npz', **arrays)
        summary.update(bandwidth=bandwidth, l0_coef=args.l0_coef, threshold_trained=log_threshold is not None,
                       feature_drift=drift,
                       note='Encoder, thresholds and decoder adapted from the original weights. Features changed: '
                            're-run feature selection for this SAE. The best checkpoint is the lowest validation '
                            'MSE among epochs whose validation L0 stays within l0_tolerance of the original. '
                            'Variance explained uses each evaluation cohort own mean, identically before and after.')
    else:
        summary['note'] = ('Only decoder weights and bias adapted. Feature values and probe inputs are unchanged. '
                           'Variance explained uses each evaluation cohort own mean, identically before and after.')
    write_json_atomic(out / 'summary.json', summary)
    manifest['status'] = 'completed'
    write_json_atomic(out / 'manifest.json', manifest)
    logger.info('Best checkpoint: %s (epoch %d)', out / 'best', best_epoch)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sae-dir', type=Path, required=True)
    parser.add_argument('--activations-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--activation-file', default='{split}_attempt2.pt')
    parser.add_argument('--probe-config', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/cache/pretrained_decoder'))
    parser.add_argument('--train', choices=('decoder', 'full'), default='decoder',
                        help='decoder: decoder only, encoder frozen. full: encoder, thresholds and decoder.')
    parser.add_argument('--output-dir', type=Path,
                        help='Default: results/sae/decoder_finetuning or results/sae/full_finetuning.')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5, help='Decoder learning rate.')
    parser.add_argument('--lr-enc', type=float, help='Full mode: encoder/threshold learning rate (default --lr).')
    parser.add_argument('--l0-coef', type=float, default=1.0,
                        help='Full mode: weight of the relative L0 penalty against relative MSE.')
    parser.add_argument('--l0-tolerance', type=float, default=0.1,
                        help='Eligible checkpoints keep validation L0 within this fraction of the original.')
    parser.add_argument('--bandwidth', type=float,
                        help='Full mode: STE kernel bandwidth (default 0.001 * mean ||x|| / sqrt(d)).')
    parser.add_argument('--freeze-threshold', action='store_true',
                        help='Full mode: keep JumpReLU thresholds fixed; train only w_enc/b_enc and the decoder.')
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    print(run(args))


if __name__ == '__main__':
    main()
