"""Fine-tune a pretrained Gemma Scope SAE on our activations.

Two modes. Each writes its own checkpoint, so both can be compared with the untouched original:

    --train decoder   The encoder is frozen; only the decoder (w_dec, b_dec) learns. Features never
                      change, so they are computed once and cached on disk.
                      Output: results/sae/decoder_finetuning/<timestamp>/
    --train full      The encoder (w_enc, b_enc, threshold) and decoder all learn, starting from the
                      original weights. An L0 penalty keeps sparsity near the original.
                      Output: results/sae/full_finetuning/<timestamp>/

The pipeline (read run() first; everything below it is a step it calls):
    1. Load the SAE and the train/validation activations. Test and intervention are never read.
    2. Measure two reference points: the original SAE, and the original with only its decoder
       bias re-centred ("bias-only").
    3. Train for up to --epochs, scoring validation after each epoch. Keep the best epoch: the
       lowest validation MSE among eligible epochs, i.e. those whose L0 stays near the original's
       and whose features have not drifted far from it (see eligibility()).
    4. Score the kept checkpoint and write the summary: reconstruction, dictionary health
       (mas_sae.sae.diagnostics) and stability against the original.

Files in the run directory:
    manifest.json       settings, input hashes and code provenance; status running -> completed
    baseline.json       metrics and dictionary health for the original and bias-only SAEs
    history.json        metrics, compact health/stability and eligibility after every epoch
    summary.json        best epoch, before/after metrics, health and stability
    best/               the kept checkpoint (config.json + params.safetensors), loadable as kind: gemma_scope
    feature_health.npz  per-feature arrays: health of both SAEs, and drift between them

Example (from the repo root):
    PYTHONPATH=src python scripts/sae/finetune_pretrained_sae.py --train full \\
        --sae-dir results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big \\
        --activations-dir artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25 \\
        --layer 25 --probe-config configs/feature_selection_mixed_final_001_layer25.yaml
"""

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.nn.functional import mse_loss
from safetensors.torch import save_file
from tqdm.auto import tqdm

from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.artifacts import get_git_provenance, sha256_file, sha256_json, write_json_atomic
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.sae import diagnostics
from mas_sae.sae.pretrained import load_gemma_scope_sae
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)

SPLITS = ('train', 'validation')                   # the only activation files ever opened
ENCODER_KEYS = ('w_enc', 'b_enc', 'threshold')     # the tensors that determine the features
GEMMA_SCOPE_BANDWIDTH = 0.001                      # STE bandwidth Gemma Scope used on unit-scaled inputs
BYTES_PER_FLOAT32 = 4
CACHE_DISK_HEADROOM_BYTES = 1024 ** 2

DECODER_NOTE = ('Only decoder weights and bias adapted. Feature values and probe inputs are unchanged. '
                'Variance explained uses each evaluation cohort own mean, identically before and after.')
FULL_NOTE = ('Encoder, thresholds and decoder adapted from the original weights. Features changed: '
             're-run feature selection for this SAE. The best checkpoint is the lowest validation '
             'MSE among eligible epochs: validation L0 within l0_tolerance of the original, and '
             'stability (activation correlation, active-set Jaccard, alive-feature count) within the '
             'configured limits. Variance explained uses each evaluation cohort own mean, identically '
             'before and after.')
BIAS_ONLY_HEALTH_NOTE = 'Identical to the original: only b_dec differs, so features and the dictionary are unchanged.'


# =============================================================================================
# 1. The pipeline
# =============================================================================================

def run(args):
    """The whole pipeline, top to bottom. Returns the run directory."""
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_encoder = args.train == 'full'

    # Step 1: load the pretrained SAE and the train/validation activations.
    model, sae_config = load_model(args, device)
    source = copy.deepcopy(model).eval()   # the untouched original, for stability comparisons
    source_encoder_sha256 = encoder_digest(model)
    splits = load_splits(args, model)
    if not train_encoder:
        # The encoder is frozen, so every feature can be computed once and reused by every epoch and run.
        for split in splits.values():
            attach_feature_cache(split, model, args.cache_dir, args.batch_size, device)

    # Step 2: reference points.
    original = evaluate_splits(model, splits, args.batch_size, device)
    bias_only_model = with_bias_only_fix(model, splits['train'], args.batch_size, device)
    bias_only = evaluate_splits(bias_only_model, splits, args.batch_size, device)
    original_health = dictionary_health(model, splits, args.batch_size, device)

    # Step 3: train, saving the best epoch to <run>/best.
    if train_encoder:
        trainer = FullTrainer(model, args, splits['train'], original['train'])
    else:
        trainer = DecoderOnlyTrainer(model, args)
    out, manifest = start_run_dir(args, splits, source_encoder_sha256, trainer)
    write_json_atomic(out / 'baseline.json', {
        'original': original, 'bias_only': bias_only,
        'health': {'original': original_health.summary, 'bias_only': BIAS_ONLY_HEALTH_NOTE}})
    best_epoch = train(args, model, source, trainer, splits, original, sae_config, out, device)
    if not train_encoder and encoder_digest(model) != source_encoder_sha256:
        raise RuntimeError('Frozen encoder or thresholds changed.')

    # Step 4: score the kept checkpoint and write the summary.
    best_model, _ = load_gemma_scope_sae(out / 'best', expected_layer=args.layer)
    model.load_state_dict(best_model.state_dict())
    adapted = evaluate_splits(model, splits, args.batch_size, device)
    adapted_health = dictionary_health(model, splits, args.batch_size, device)
    stability, stability_per_feature = stability_report(source, model, splits['validation'], args.batch_size, device)
    save_feature_arrays(out / 'feature_health.npz', original_health, adapted_health, stability_per_feature)
    summary = {'train_mode': args.train, 'best_epoch': best_epoch, 'original': original,
               'bias_only': bias_only, 'adapted': adapted,
               'encoder_unchanged': encoder_digest(model) == source_encoder_sha256,
               'l0_target': original['validation']['l0'], 'l0_tolerance': args.l0_tolerance,
               'stability_limits': stability_limits(args),
               'health': {'original': original_health.summary, 'adapted': adapted_health.summary},
               'stability': stability}
    if train_encoder:
        summary.update(bandwidth=trainer.bandwidth, l0_coef=args.l0_coef,
                       threshold_trained=trainer.log_threshold is not None, note=FULL_NOTE)
    else:
        summary['note'] = DECODER_NOTE
    write_json_atomic(out / 'summary.json', summary)
    manifest['status'] = 'completed'
    write_json_atomic(out / 'manifest.json', manifest)
    logger.info('Best checkpoint: %s (epoch %d)', out / 'best', best_epoch)
    return out


# =============================================================================================
# 2. Pipeline steps, in the order run() calls them
# =============================================================================================

def validate_args(args):
    """Reject bad settings before any data is loaded."""
    for name in ('epochs', 'batch_size', 'patience'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if args.train not in ('decoder', 'full'):
        raise ValueError("train must be 'decoder' or 'full'.")
    for name, value in (('lr', args.lr), ('lr_enc', encoder_lr(args))):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if args.l0_coef < 0 or args.l0_tolerance < 0:
        raise ValueError('l0_coef and l0_tolerance must be nonnegative.')
    if args.bandwidth is not None and not args.bandwidth > 0:
        raise ValueError('bandwidth must be positive.')
    if not -1 <= args.min_activation_correlation <= 1 or not 0 <= args.min_jaccard <= 1:
        raise ValueError('min_activation_correlation must be in [-1, 1] and min_jaccard in [0, 1].')
    if not args.max_alive_change >= 0:
        raise ValueError('max_alive_change must be nonnegative (inf disables it).')


def encoder_lr(args):
    """Learning rate for the encoder and thresholds in full mode: --lr-enc, else --lr."""
    return args.lr if args.lr_enc is None else args.lr_enc


def load_model(args, device):
    """Load the source SAE (never modified on disk). Returns (model on device, its config.json)."""
    model, sae_config = load_gemma_scope_sae(args.sae_dir, expected_layer=args.layer)
    # Checked here, before minutes of data loading, rather than when training starts.
    if args.train == 'full' and not args.freeze_threshold and not (model.threshold > 0).all():
        raise ValueError('Trainable thresholds are stored in log space and must be positive; '
                         'use --freeze-threshold.')
    return model.to(device), sae_config


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


def load_splits(args, model):
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


def attach_feature_cache(split, model, cache_dir, batch_size, device):
    """Decoder mode: point split.cached_features at the (possibly newly built) on-disk encoder output."""
    identity = {'split': split.name, 'activation_sha256': split.sha256}
    split.cached_features, split.cache_path = cached_features(
        model, split.activations, identity, cache_dir, batch_size, device)


def evaluate_splits(model, splits, batch_size, device):
    """evaluate() on every split. Returns {'train': metrics, 'validation': metrics}."""
    return {name: evaluate(model, split, batch_size, device) for name, split in splits.items()}


@torch.no_grad()
def with_bias_only_fix(model, train_split, batch_size, device):
    """A copy of `model` whose decoder bias absorbs the mean train reconstruction error.

    This is the cheapest possible adaptation: one constant shift, fitted on train only. If fine-tuning
    barely beats it, most of the gain is just re-centring. `model` itself is not changed.
    """
    total_residual = torch.zeros(model.input_dim, dtype=torch.float64, device=device)
    for rows, batch in iterate_batches(train_split.activations, batch_size, device):
        reconstruction = model.decoder(features_for(model, train_split, rows, batch))
        total_residual += (batch - reconstruction).double().sum(dim=0)
    shifted = copy.deepcopy(model)
    shifted.b_dec.add_((total_residual / len(train_split.activations)).to(shifted.b_dec.dtype))
    return shifted


class DecoderOnlyTrainer:
    """--train decoder: features come from the frozen encoder's cache; only w_dec and b_dec learn.

    The loss is plain reconstruction MSE.
    """
    logged_terms = ('loss',)

    def __init__(self, model, args):
        self.model = model
        model.w_dec.requires_grad_(True)
        model.b_dec.requires_grad_(True)
        self.optimizer = torch.optim.Adam([{'params': [model.w_dec, model.b_dec], 'lr': args.lr}])

    def manifest_fields(self):
        return {'bandwidth': None, 'lr_enc': None}

    def loss(self, split, rows, batch):
        """Returns (loss to minimise, {term name: value} for history.json)."""
        reconstruction = self.model.decoder(features_for(self.model, split, rows, batch))
        loss = mse_loss(reconstruction, batch)
        return loss, {'loss': loss}

    def after_step(self):
        pass


class FullTrainer:
    """--train full: the encoder, thresholds and decoder all learn.

    Training a JumpReLU encoder directly has two problems, handled like this:

    1. The gate `pre > threshold` is a step, so it has no gradient. encoder_ste() uses the
       straight-through estimators of Rajamanoharan et al. (2024): within a window of width
       `bandwidth` around each threshold, a pseudo-gradient tells the threshold which way to move.
    2. With MSE alone, the encoder can lower its error just by switching more features on. So the
       loss adds an L0 penalty, with both terms relative to the original SAE on train:

           loss = MSE / MSE_original + l0_coef * L0 / L0_original

       With l0_coef = 1, a 1% rise in L0 costs as much as a 1% fall in MSE gains.

    Thresholds are optimised as log(threshold) so they stay positive. After every step, the result
    is written back to model.threshold, which is what evaluation and checkpoints use.
    """
    logged_terms = ('mse', 'l0', 'loss')

    def __init__(self, model, args, train_split, original_train_metrics):
        if original_train_metrics['l0'] == 0:
            raise ValueError('Original SAE has no active features on train; the L0 penalty is undefined.')
        self.model = model
        self.l0_coef = args.l0_coef
        self.lr_enc = encoder_lr(args)
        self.mse_scale = original_train_metrics['mse']
        self.l0_scale = original_train_metrics['l0']
        self.bandwidth = args.bandwidth or default_bandwidth(train_split.activations, args.batch_size)
        for parameter in (model.w_enc, model.b_enc, model.w_dec, model.b_dec):
            parameter.requires_grad_(True)
        self.log_threshold = None if args.freeze_threshold else torch.nn.Parameter(model.threshold.detach().log())
        encoder_parameters = [model.w_enc, model.b_enc]
        if self.log_threshold is not None:
            encoder_parameters.append(self.log_threshold)
        self.optimizer = torch.optim.Adam([
            {'params': [model.w_dec, model.b_dec], 'lr': args.lr},
            {'params': encoder_parameters, 'lr': self.lr_enc},
        ])

    def manifest_fields(self):
        return {'bandwidth': self.bandwidth, 'lr_enc': self.lr_enc}

    def loss(self, split, rows, batch):
        """Returns (loss to minimise, {term name: value} for history.json)."""
        threshold = None if self.log_threshold is None else self.log_threshold.exp()
        features, l0_per_example = self.model.encoder_ste(batch, self.bandwidth, threshold)
        mse = mse_loss(self.model.decoder(features), batch)
        l0 = l0_per_example.mean()
        loss = mse / self.mse_scale + self.l0_coef * l0 / self.l0_scale
        return loss, {'mse': mse, 'l0': l0, 'loss': loss}

    def after_step(self):
        if self.log_threshold is not None:
            with torch.no_grad():
                self.model.threshold.copy_(self.log_threshold.exp())


def start_run_dir(args, splits, source_encoder_sha256, trainer):
    """Create <output-dir>/<timestamp>/ and write manifest.json with status 'running'.

    Returns (run directory, manifest dict); run() marks the manifest completed at the end.
    """
    output_dir = args.output_dir or Path(f'results/sae/{args.train}_finetuning')
    out = create_versioned_run_dir(output_dir)
    manifest = {
        'settings': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'source_files_sha256': {str(args.sae_dir / name): sha256_file(args.sae_dir / name)
                                for name in ('config.json', 'params.safetensors')},
        'activation_sha256': {name: split.sha256 for name, split in splits.items()},
        'encoder_sha256': source_encoder_sha256,
        'cache_paths': {name: str(split.cache_path) for name, split in splits.items() if split.cache_path},
        'git': get_git_provenance(),
        'script_sha256': sha256_file(Path(__file__)),
        'probe_config_sha256': sha256_file(args.probe_config) if args.probe_config else None,
        **trainer.manifest_fields(),
        'status': 'running',
    }
    write_json_atomic(out / 'manifest.json', manifest)
    return out, manifest


def train(args, model, source, trainer, splits, original, sae_config, out, device):
    """Train for up to args.epochs and save the best epoch to out/best. Returns the best epoch number.

    Choosing the best epoch:
      - An epoch is eligible only if it passes eligibility(): L0 near the original's, and features
        still close to the original's (`source`). In decoder mode the encoder is frozen, so every
        epoch is eligible.
      - Among eligible epochs, the lowest validation MSE wins.
      - The untouched original counts as epoch 0. If training never helps, the original is kept.
    Training stops early after `patience` epochs in a row without a new best.
    """
    train_split = splits['train']
    n_train = len(train_split.activations)
    target_l0 = original['validation']['l0']
    best_mse, best_epoch = original['validation']['mse'], 0
    save_model(model, sae_config, out / 'best')
    history, epochs_without_improvement = [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        term_totals = dict.fromkeys(trainer.logged_terms, 0.0)
        batches = iterate_batches(train_split.activations, args.batch_size, device, shuffle=True)
        for rows, batch in tqdm(batches, total=math.ceil(n_train / args.batch_size),
                                desc=f'Epoch {epoch}/{args.epochs}', unit='batch'):
            trainer.optimizer.zero_grad(set_to_none=True)
            loss, terms = trainer.loss(train_split, rows, batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss.')
            for name, value in terms.items():
                term_totals[name] += value.item() * len(batch)
            loss.backward()
            trainer.optimizer.step()
            trainer.after_step()
        model.eval()

        metrics = evaluate_splits(model, splits, args.batch_size, device)
        validation = metrics['validation']
        if not np.isfinite(validation['mse']):
            raise ValueError('Nonfinite validation loss.')
        paired = paired_stats(source, model, splits['validation'], args.batch_size, device)
        health = {**diagnostics.compact_health(paired.adapted), **diagnostics.compact_stability(paired)}
        eligible, reasons = eligibility(validation['l0'], health, target_l0, args)
        history.append({'epoch': epoch, **metrics,
                        'train_objective': {name: total / n_train for name, total in term_totals.items()},
                        'validation_health': health,
                        'l0_eligible': within_l0(validation['l0'], target_l0, args.l0_tolerance),
                        'eligible': eligible, 'ineligible_reasons': reasons})
        write_json_atomic(out / 'history.json', history)
        logger.info('Epoch %d validation MSE %.6g (original %.6g), L0 %.4g (original %.4g), '
                    'correlation median %s, Jaccard mean %.4g%s',
                    epoch, validation['mse'], original['validation']['mse'], validation['l0'], target_l0,
                    _format(health['activation_correlation_median']), health['active_set_jaccard_mean'],
                    '' if eligible else f' [ineligible: {"; ".join(reasons)}]')

        if eligible and validation['mse'] < best_mse:
            best_mse, best_epoch, epochs_without_improvement = validation['mse'], epoch, 0
            save_model(model, sae_config, out / 'best')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    return best_epoch


@dataclass
class Health:
    """dictionary_health() output: the JSON summary, and per-feature arrays for feature_health.npz."""
    summary: dict
    per_feature: dict


def dictionary_health(model, splits, batch_size, device):
    """Utilization, concentration and activation health per split, plus weight geometry.

    The near-duplicate search is also run among features alive on train (the larger split).
    Per-feature arrays are taken from validation, the split stability is measured on.
    """
    stats = {name: feature_stats(model, split, batch_size, device) for name, split in splits.items()}
    activation = {name: diagnostics.activation_health(s) for name, s in stats.items()}
    parameters, parameter_arrays = diagnostics.parameter_health(model, alive=stats['train'].alive())
    summary = {name: report for name, (report, _) in activation.items()}
    summary['parameters'] = parameters
    return Health(summary, {**activation['validation'][1], **parameter_arrays})


def stability_report(source, model, split, batch_size, device):
    """Adapted vs original on one split, including split/merge detection (a second pass)."""
    paired = paired_stats(source, model, split, batch_size, device)
    summary, per_feature = diagnostics.stability(paired, source, model)
    summary['split_merge'], split_merge_arrays = diagnostics.split_merge(
        lambda: paired_batches(source, model, split, batch_size, device), paired)
    return summary, {**per_feature, **split_merge_arrays}


def save_feature_arrays(path, original_health, adapted_health, stability_per_feature):
    arrays = {**{f'original_{k}': v for k, v in original_health.per_feature.items()},
              **{f'adapted_{k}': v for k, v in adapted_health.per_feature.items()},
              **stability_per_feature}
    np.savez(path, **{name: value.detach().cpu().numpy() for name, value in arrays.items()})


@torch.no_grad()
def feature_stats(model, split, batch_size, device):
    stats = diagnostics.FeatureStats(model.latent_dim, device)
    for rows, batch in iterate_batches(split.activations, batch_size, device):
        stats.add(features_for(model, split, rows, batch))
    return stats


def paired_batches(source, model, split, batch_size, device):
    """Yield (original features, current features) for every batch of a split."""
    with torch.no_grad():
        for rows, batch in iterate_batches(split.activations, batch_size, device):
            yield source.encoder(batch), features_for(model, split, rows, batch)


def paired_stats(source, model, split, batch_size, device):
    paired = diagnostics.PairedStats(model.latent_dim, device)
    for original_features, features in paired_batches(source, model, split, batch_size, device):
        paired.add(original_features, features)
    return paired


# =============================================================================================
# 3. Metrics
# =============================================================================================

@torch.no_grad()
def evaluate(model, split, batch_size, device):
    """Reconstruction quality and sparsity of `model` on one split.

    Returns {'n', 'mse', 'variance_explained', 'l0', 'dead_fraction'}, plus 'probe_cohort' (n, mse,
    variance_explained on the probe's rows) when the split has one.

      - mse: mean squared error per activation dimension.
      - variance_explained = 1 - (squared error) / (squared distance from the split's mean). The
        probe cohort uses its own mean. The means are fixed per split, so before/after numbers compare.
      - l0: mean number of active features per example.
      - dead_fraction: share of features that never fire anywhere in the split.
    """
    everyone = ReconstructionTotals()
    cohort = ReconstructionTotals()
    total_active = 0
    ever_active = torch.zeros(model.latent_dim, dtype=torch.bool, device=device)
    split_mean = split.mean.to(device)
    for rows, batch in iterate_batches(split.activations, batch_size, device):
        features = features_for(model, split, rows, batch)
        is_active = features > 0
        total_active += int(is_active.sum())
        ever_active |= is_active.any(dim=0)
        squared_error = (model.decoder(features) - batch).double().square().sum(dim=1)   # one per example
        everyone.add(squared_error, (batch.double() - split_mean).square().sum(dim=1))
        if split.probe_cohort is not None:
            in_cohort = torch.from_numpy(split.probe_cohort[rows]).to(device)
            cohort_batch = batch[in_cohort].double()
            cohort.add(squared_error[in_cohort],
                       (cohort_batch - split.probe_cohort_mean.to(device)).square().sum(dim=1))
    n_dims = split.activations.shape[1]
    result = everyone.metrics(n_dims)
    result['l0'] = total_active / len(split.activations)
    result['dead_fraction'] = 1 - ever_active.float().mean().item()
    if split.probe_cohort is not None:
        result['probe_cohort'] = cohort.metrics(n_dims)
    return result


class ReconstructionTotals:
    """Running sums of squared error and squared distance from the mean, for MSE and variance explained."""

    def __init__(self):
        self.n = 0
        self.squared_error = 0.0
        self.squared_from_mean = 0.0

    def add(self, squared_error, squared_from_mean):
        """Add per-example values for one batch."""
        self.n += len(squared_error)
        self.squared_error += squared_error.sum().item()
        self.squared_from_mean += squared_from_mean.sum().item()

    def metrics(self, n_dims):
        variance_explained = (1 - self.squared_error / self.squared_from_mean
                              if self.squared_from_mean > 0 else None)
        return {'n': self.n, 'mse': self.squared_error / (self.n * n_dims),
                'variance_explained': variance_explained}


def within_l0(l0, target, tolerance):
    """True if a checkpoint's mean L0 is within a relative tolerance of the original's (target)."""
    return abs(l0 - target) <= tolerance * target


def stability_limits(args):
    return {'min_activation_correlation': args.min_activation_correlation, 'min_jaccard': args.min_jaccard,
            'max_alive_change': args.max_alive_change}


def eligibility(l0, health, target_l0, args):
    """Can this epoch be kept? Returns (eligible, reasons it is not).

    The fine-tune should be light, so beyond lowering MSE an epoch must keep:
      - validation L0 within l0_tolerance of the original;
      - the median activation correlation with the original (features alive in both) at least
        min_activation_correlation (0 disables);
      - the mean active-set Jaccard with the original at least min_jaccard (0 disables);
      - the number of features alive on validation within max_alive_change of the original (inf disables).
    `health` is the epoch's compact health/stability block.
    """
    reasons = []
    if not within_l0(l0, target_l0, args.l0_tolerance):
        reasons.append(f'L0 {l0:.4g} not within {args.l0_tolerance:g} of {target_l0:.4g}')
    correlation = health['activation_correlation_median']
    if args.min_activation_correlation > 0 and (correlation is None or correlation < args.min_activation_correlation):
        reasons.append(f'activation correlation median {_format(correlation)} < {args.min_activation_correlation:g}')
    jaccard = health['active_set_jaccard_mean']
    if args.min_jaccard > 0 and jaccard < args.min_jaccard:
        reasons.append(f'active-set Jaccard mean {jaccard:.4g} < {args.min_jaccard:g}')
    change = health['alive_relative_change']
    if change is not None and abs(change) > args.max_alive_change:
        reasons.append(f'alive features changed by {change:+.2%} (limit {args.max_alive_change:g})')
    return not reasons, reasons


def _format(value):
    return 'n/a' if value is None else f'{value:.4g}'


def default_bandwidth(activations, batch_size):
    """Gemma Scope's STE bandwidth, rescaled from their unit-scaled inputs to our raw activations.

    Gemma Scope scaled its inputs so that mean ||x|| = sqrt(d) and used a bandwidth of 0.001.
    For raw inputs, the equivalent bandwidth is 0.001 * mean ||x|| / sqrt(d).
    """
    total_norm = sum(activations[start:start + batch_size].double().norm(dim=1).sum().item()
                     for start in range(0, len(activations), batch_size))
    mean_norm = total_norm / len(activations)
    return GEMMA_SCOPE_BANDWIDTH * mean_norm / activations.shape[1] ** 0.5


# =============================================================================================
# 4. Batching and feature lookup
# =============================================================================================

def iterate_batches(activations, batch_size, device, shuffle=False):
    """Yield (row indices, batch on device). Shuffling uses the torch RNG, so --seed controls it."""
    order = torch.randperm(len(activations)).numpy() if shuffle else np.arange(len(activations))
    for start in range(0, len(activations), batch_size):
        rows = order[start:start + batch_size]
        yield rows, activations[rows].to(device).float()


def features_for(model, split, rows, batch):
    """Encoder output for a batch: read from the cache if the split has one (decoder mode), else encoded now."""
    if split.cached_features is not None:
        # Only these rows are read from the memory-mapped file.
        return torch.from_numpy(np.array(split.cached_features[rows], copy=True)).to(batch.device)
    with torch.no_grad():
        return model.encoder(batch)


# =============================================================================================
# 5. Encoder feature cache (decoder mode only)
# =============================================================================================
# The frozen encoder's output for 80k examples at width 16k is ~5 GB, so it is written to disk once
# and memory-mapped afterwards. The cache directory is named by a hash of everything that determines
# its contents (encoder tensors + activation file hash), so a stale cache can never be reused.
# A build writes to a hidden temporary directory and is renamed into place only when complete, and a
# lock file stops two processes building the same cache at once.

def encoder_digest(model):
    """SHA-256 of the encoder tensors (names, shapes, dtypes and bytes). Equal digests mean identical features."""
    digest = hashlib.sha256()
    for name in ENCODER_KEYS:
        tensor = getattr(model, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def cached_features(model, activations, identity, cache_dir, batch_size, device):
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


# =============================================================================================
# 6. Small helpers
# =============================================================================================

def save_model(model, sae_config, destination):
    """Write params.safetensors + config.json in the Gemma Scope layout (loadable as kind: gemma_scope)."""
    destination.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
              destination / 'params.safetensors')
    write_json_atomic(destination / 'config.json', sae_config)


def probe_cohort_rows(config_path, args, n_validation):
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


# =============================================================================================
# Command line
# =============================================================================================

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    inputs = parser.add_argument_group('inputs')
    inputs.add_argument('--sae-dir', type=Path, required=True,
                        help='Gemma Scope SAE folder (config.json + params.safetensors). Never modified.')
    inputs.add_argument('--activations-dir', type=Path, required=True, help='Folder with the activation .pt files.')
    inputs.add_argument('--layer', type=int, required=True, help='Layer the activations come from; checked against the SAE.')
    inputs.add_argument('--activation-file', default='{split}_attempt2.pt',
                        help="Filename pattern; {split} becomes 'train' / 'validation'.")
    inputs.add_argument('--probe-config', type=Path,
                        help='Feature-selection config. If given, validation metrics are also reported on the probe cohort.')

    training = parser.add_argument_group('training')
    training.add_argument('--train', choices=('decoder', 'full'), default='decoder',
                          help='decoder: decoder only, encoder frozen. full: encoder, thresholds and decoder.')
    training.add_argument('--epochs', type=int, default=5, help='Maximum number of epochs.')
    training.add_argument('--batch-size', type=int, default=256)
    training.add_argument('--lr', type=float, default=1e-5, help='Decoder learning rate (Adam).')
    training.add_argument('--patience', type=int, default=5, help='Stop after this many epochs without a new best.')
    training.add_argument('--l0-tolerance', type=float, default=0.1,
                          help='An epoch can be kept only if its validation L0 is within this fraction of the original.')
    training.add_argument('--min-activation-correlation', type=float, default=0.95,
                          help='An epoch can be kept only if the median per-feature activation correlation with the '
                               'original (features alive in both) is at least this. 0 disables.')
    training.add_argument('--min-jaccard', type=float, default=0.85,
                          help='An epoch can be kept only if the mean active-set Jaccard with the original is at '
                               'least this. 0 disables.')
    training.add_argument('--max-alive-change', type=float, default=0.05,
                          help='An epoch can be kept only if the number of features alive on validation changed by '
                               'at most this fraction. inf disables.')
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    full = parser.add_argument_group('full mode only (--train full)')
    full.add_argument('--lr-enc', type=float, default=1e-6,
                      help='Encoder and threshold learning rate (default: 1e-6).')
    full.add_argument('--l0-coef', type=float, default=0.1,
                      help='Weight of the relative L0 penalty against the relative MSE (default: 0.1).')
    full.add_argument('--bandwidth', type=float,
                      help='STE window width around each threshold (default: 0.001 * mean ||x|| / sqrt(d)).')
    full.add_argument('--freeze-threshold', action='store_true',
                      help='Keep JumpReLU thresholds fixed; train only w_enc/b_enc and the decoder.')

    outputs = parser.add_argument_group('outputs')
    outputs.add_argument('--output-dir', type=Path,
                         help='Default: results/sae/decoder_finetuning or results/sae/full_finetuning.')
    outputs.add_argument('--cache-dir', type=Path, default=Path('artifacts/cache/pretrained_decoder'),
                         help='Decoder mode: where encoder features are cached.')
    return parser


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    print(run(args))


if __name__ == '__main__':
    main()
