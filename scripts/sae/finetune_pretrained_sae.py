"""Fine-tune a pretrained Gemma Scope SAE on our activations.

Two modes. Each writes its own checkpoint, so both can be compared with the untouched original:

    --train decoder   The encoder is frozen; only the decoder (w_dec, b_dec) learns. Features never
                      change, so they are computed once and cached on disk.
                      Output: results/sae/decoder_finetuning/<timestamp>/
    --train full      The encoder (w_enc, b_enc, threshold) and decoder all learn, starting from the
                      original weights. An L0 penalty keeps sparsity near the original.
                      Output: results/sae/full_finetuning/<timestamp>/

The pipeline (run() below; the steps live in mas_sae.sae.finetuning, called as modules):
    1. Load the SAE and the train/validation activations. Test and intervention are never read.
    2. Measure two reference points: the original SAE, and the original with only its decoder
       bias re-centred ("bias-only").
    3. Train for up to --epochs, scoring validation after each epoch. Keep the best epoch: the
       lowest validation MSE among eligible epochs, i.e. those whose L0 stays near the original's
       and whose features have not drifted far from it (see selection.eligibility()).
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
        --layer 25 --probe-config configs/feature_selection/feature_selection_mixed_final_001_layer25.yaml
"""

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import torch

from mas_sae.experiments.artifacts import get_git_provenance, sha256_file, write_json_atomic
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.sae.finetuning import cache, data, evaluation, loop, reporting, selection, trainers
from mas_sae.sae.pretrained import load_gemma_scope_sae
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)

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
    source_encoder_sha256 = cache.encoder_digest(model)
    splits = data.load_splits(args, model)
    if not train_encoder:
        # The encoder is frozen, so every feature can be computed once and reused by every epoch and run.
        for split in splits.values():
            cache.attach_feature_cache(split, model, args.cache_dir, args.batch_size, device)

    # Step 2: reference points.
    original = evaluation.evaluate_splits(model, splits, args.batch_size, device)
    bias_only_model = evaluation.with_bias_only_fix(model, splits['train'], args.batch_size, device)
    bias_only = evaluation.evaluate_splits(bias_only_model, splits, args.batch_size, device)
    original_health = reporting.dictionary_health(model, splits, args.batch_size, device)

    # Step 3: train, saving the best epoch to <run>/best.
    if train_encoder:
        trainer = trainers.FullTrainer(model, args, splits['train'], original['train'])
    else:
        trainer = trainers.DecoderOnlyTrainer(model, args)
    out, manifest = start_run_dir(args, splits, source_encoder_sha256, trainer)
    write_json_atomic(out / 'baseline.json', {
        'original': original, 'bias_only': bias_only,
        'health': {'original': original_health.summary, 'bias_only': BIAS_ONLY_HEALTH_NOTE}})
    best_epoch = loop.train(args, model, source, trainer, splits, original, sae_config, out, device)
    if not train_encoder and cache.encoder_digest(model) != source_encoder_sha256:
        raise RuntimeError('Frozen encoder or thresholds changed.')

    # Step 4: score the kept checkpoint and write the summary.
    best_model, _ = load_gemma_scope_sae(out / 'best', expected_layer=args.layer)
    model.load_state_dict(best_model.state_dict())
    adapted = evaluation.evaluate_splits(model, splits, args.batch_size, device)
    adapted_health = reporting.dictionary_health(model, splits, args.batch_size, device)
    stability, stability_per_feature = reporting.stability_report(
        source, model, splits['validation'], args.batch_size, device)
    reporting.save_feature_arrays(out / 'feature_health.npz', original_health, adapted_health,
                                  stability_per_feature)
    summary = {'train_mode': args.train, 'best_epoch': best_epoch, 'original': original,
               'bias_only': bias_only, 'adapted': adapted,
               'encoder_unchanged': cache.encoder_digest(model) == source_encoder_sha256,
               'l0_target': original['validation']['l0'], 'l0_tolerance': args.l0_tolerance,
               'stability_limits': selection.stability_limits(args),
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
# 2. Script-only steps: argument checks, model loading, the run directory
# =============================================================================================

def validate_args(args):
    """Reject bad settings before any data is loaded."""
    for name in ('epochs', 'batch_size', 'patience'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if args.train not in ('decoder', 'full'):
        raise ValueError("train must be 'decoder' or 'full'.")
    for name, value in (('lr', args.lr), ('lr_enc', trainers.encoder_lr(args))):
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


def load_model(args, device):
    """Load the source SAE (never modified on disk). Returns (model on device, its config.json)."""
    model, sae_config = load_gemma_scope_sae(args.sae_dir, expected_layer=args.layer)
    # Checked here, before minutes of data loading, rather than when training starts.
    if args.train == 'full' and not args.freeze_threshold and not (model.threshold > 0).all():
        raise ValueError('Trainable thresholds are stored in log space and must be positive; '
                         'use --freeze-threshold.')
    return model.to(device), sae_config


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
