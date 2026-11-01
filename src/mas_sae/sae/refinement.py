"""Fork a validation-only TopK trial into matched feature-revival continuations."""
import json
import math
from pathlib import Path

import torch

from mas_sae.experiments.artifacts import write_json_atomic, sha256_file, get_git_provenance
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.sae.training_state import capture_rng_state, restore_rng_state
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.dataloader import ActivationDataset
from mas_sae.sae.resampling import (
    scan_features, select_resampling_features, resample_features, LinearWarmup,
)


def resolve_branches(settings):
    """Validate branch definitions, preserving the original two-branch pilot.

    Parameters
    ----------
    settings : dict
        Experiment YAML; branches specify name, strategy, and warmup_steps.

    Returns
    -------
    list[dict]
        Resolved unique branches with explicit strategy and warmup settings.
    """
    branches = settings.get('branches', [
        {'name': 'control', 'strategy': 'none'},
        {'name': 'resampling', 'strategy': 'residual'},
    ])
    if not isinstance(branches, list) or not branches:
        raise ValueError('branches must be a nonempty list.')
    resolved = []
    for branch in branches:
        name = branch.get('name', '')
        strategy = branch.get('strategy', 'none')
        steps = branch.get('warmup_steps', 0)
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Branch names must contain only letters, digits, underscores, or hyphens.')
        if strategy not in {'none', 'residual', 'bricken'}:
            raise ValueError(f'Unknown strategy: {strategy}')
        if type(steps) is not int or steps < 0 or steps == 1:
            raise ValueError('warmup_steps must be zero or an integer of at least two.')
        resolved.append({'name': name, 'strategy': strategy, 'warmup_steps': steps})
    if len({b['name'] for b in resolved}) != len(resolved):
        raise ValueError('Branch names must be unique.')
    return resolved


def save_checkpoint(path, runner, config, epoch, warmup, **continuation):
    """Save an inference-compatible checkpoint without selecting by reconstruction.

    Parameters
    ----------
    path : pathlib.Path
        New experiment checkpoint destination.
    runner : ModelRunner
        Model, optimizer, and activity state to preserve.
    config : dict
        Source model configuration used by the existing checkpoint loader.
    epoch : int
        Completed additional epochs, zero immediately after intervention.
    warmup : LinearWarmup
        Applied learning-rate schedule and target rates.

    Returns
    -------
    None
        Atomically replaces the experiment checkpoint.
    """
    temporary = path.with_suffix('.pt.tmp')
    torch.save({'model_state_dict': runner.model.state_dict(), 'model_config': config,
                'optimizer_state_dict': runner.optimizer.state_dict(),
                'training_examples_seen': runner.training_examples_seen,
                'last_fired_example': runner.last_fired_example,
                'additional_epoch': epoch, 'optimizer_updates': len(warmup.history),
                'target_learning_rates': warmup.targets, 'warmup_history': warmup.history,
                **continuation}, temporary)
    temporary.replace(path)


def measure_activity(runner, train, validation_loader, baseline_main, tracked_ids, batch_size):
    """Evaluate frozen coverage and the same candidate features in every branch.

    Parameters
    ----------
    runner : ModelRunner
        SAE runner; evaluation does not update training activity counters.
    train : ActivationDataset
        Training inputs used for a frozen per-feature scan.
    validation_loader : DataLoader
        Held-out validation inputs used for activity and fidelity measurements.
    baseline_main : torch.Tensor
        Main-path counts from the untouched source checkpoint.
    tracked_ids : torch.Tensor
        Shared candidate IDs, including when no replacements are performed.
    batch_size : int
        Frozen training-scan batch size.

    Returns
    -------
    tuple[dict, dict]
        Aggregate metrics and per-feature train counts/validation frequencies.
        Repeated validation activity means at least ten validation examples.
    """
    validation = runner.val_epoch(validation_loader)
    frequencies = runner.last_feature_frequencies.clone()
    positive, main, _ = scan_features(runner.model, train.activations, batch_size)
    validation_counts = (frequencies * len(validation_loader.dataset)).round().long()
    tracked = tracked_ids.tolist()
    metrics = {
        'validation': validation,
        'frozen_train_coverage': (main > 0).float().mean().item(),
        'net_active_gain': int((main > 0).sum() - (baseline_main > 0).sum()),
        'displaced_initially_active': int(((baseline_main > 0) & (main == 0)).sum()),
        'newly_active': int(((baseline_main == 0) & (main > 0)).sum()),
        'tracked_positive': int((positive[tracked] > 0).sum()),
        'tracked_main_active': int((main[tracked] > 0).sum()),
        'tracked_validation_active': int((validation_counts[tracked] > 0).sum()),
        'tracked_repeated_10_examples': int((main[tracked] >= 10).sum()),
        'tracked_validation_repeated_10_examples': int((validation_counts[tracked] >= 10).sum()),
    }
    counts = {'positive': positive.tolist(), 'main': main.tolist(),
              'validation_frequency': frequencies.tolist(), 'validation_count': validation_counts.tolist()}
    return metrics, counts


def run_refinement(settings, root, *, dry_run=False, resume=False):
    """Run isolated matched continuations; resume only committed epoch boundaries."""
    root = Path(root)
    if not isinstance(settings, dict):
        raise ValueError('Refinement configuration must be a mapping.')
    if not settings.get('source_run', settings.get('source_trial')):
        raise ValueError('Require source_run (or legacy source_trial).')
    source = root / settings.get('source_run', settings.get('source_trial'))
    output = root / settings['output_dir']
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Refinement output must be outside the source run.')
    source_manifest = source / 'manifest.json'
    if source_manifest.is_file() and json.loads(source_manifest.read_text()).get('status') != 'completed':
        raise ValueError('Source run must be completed before refinement.')
    config = json.loads((source / 'config.json').read_text())
    epochs = settings.get('additional_epochs', 20)
    budget = settings.get('resampling_budget', 512)
    fraction = settings.get('positive_fraction', .01)
    start_factor = settings.get('warmup_start_factor', .1)
    seeds = settings.get('seeds', [42])
    branches = resolve_branches(settings)
    if config['sparsity_mode'] != 'topk' or config.get('evaluate_test', True):
        raise ValueError('Source must be a validation-only TopK trial.')
    if type(epochs) is not int or epochs < 1 or type(budget) is not int or budget < 1:
        raise ValueError('Epoch and resampling budgets must be positive integers.')
    if not 0 < fraction < .5 or not 0 < start_factor <= 1:
        raise ValueError('Require 0 < positive_fraction < .5 and 0 < warmup_start_factor <= 1.')
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Require unique integer seeds in [0, 2**32).')
    if output.exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite {output}')
    checkpoint = source / 'checkpoints/last_checkpoint.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f'{source}\n-> {output}\n{len(seeds)} seed(s), {len(branches)} branches, {epochs} additional epochs')
    location = root / 'data/activations' / config['activation_run_name'] / f"layer_{config['layer']:02d}"
    train = ActivationDataset('train', location)
    validation = ActivationDataset('validation', location)
    total_steps = epochs * math.ceil(len(train) / config['batch_size'])
    if any(b['warmup_steps'] > total_steps for b in branches):
        raise ValueError('Warmup cannot exceed the total continuation updates.')
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if dry_run:
        print(json.dumps(branches, indent=2))
        return
    print(f'Using device: {device}')
    manifest = {
        **settings, 'branches': branches, 'source_config': config, 'device': str(device),
        'source_checkpoint_sha256': sha256_file(checkpoint),
        'activation_sha256': {split: sha256_file(location / f'{split}.pt') for split in ['train', 'validation']},
        'schedule': 'Optional step warmup to inherited LR, then constant; no scheduler or early stopping',
        'seed_scope': 'Continuation order and resampling only; source training seed is unchanged',
        **get_git_provenance(),
    }
    manifest_path = output / 'experiment.json'
    if resume:
        if not manifest_path.is_file():
            raise ValueError('Resume requires an existing refinement manifest.')
        previous = json.loads(manifest_path.read_text())
        provenance_keys = set(get_git_provenance())
        if {k: v for k, v in previous.items() if k not in provenance_keys} != {k: v for k, v in manifest.items() if k not in provenance_keys}:
            raise ValueError('Refinement config, source, data, or device changed; refusing resume.')
    else:
        output.mkdir(parents=True)
        write_json_atomic(manifest_path, manifest)
    training_mean = train.activations.float().mean(0)
    for seed in seeds:
        source_scan = None
        tracked_ids = None
        for branch in branches:
            name, strategy = branch['name'], branch['strategy']
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = SparseAutoencoder(config['input_dim'], config['hidden_dim'], config['latent_dim'],
                                      'topk', config['top_k']).to(device)
            model.load_state_dict(state['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            optimizer.load_state_dict(state['optimizer'])
            runner = ModelRunner(model, config['sparsity_coefficient'], optimizer,
                                 training_mean=training_mean,
                                 inactivity_window_examples=config['inactivity_window_examples'],
                                 aux_coefficient=config['aux_coefficient'], aux_k=config['aux_k'],
                                 aux_dead_window_examples=config['aux_dead_window_examples'],
                                 weight_l2_coefficient=config.get('weight_l2_coefficient', 0.0))
            runner.training_examples_seen = state['training_examples_seen']
            runner.last_fired_example = state['last_fired_example'].clone().cpu()
            if source_scan is None:
                source_scan = scan_features(model, train.activations, config['batch_size'])
                tracked_ids = select_resampling_features(runner, source_scan[0], source_scan[2], budget, seed)
                write_json_atomic(output / f'candidates_seed_{seed}.json', {'feature_ids': tracked_ids.tolist()})
            directory = output / f'seed_{seed}' / name
            directory.mkdir(parents=True, exist_ok=resume)
            resume_path = directory / 'last_checkpoint.pt'
            saved = torch.load(resume_path, map_location='cpu', weights_only=True) if resume and resume_path.is_file() else None
            if saved is not None and 'rng' not in saved:
                raise ValueError('Legacy experiment checkpoint lacks refinement resume state.')
            warmup = LinearWarmup(optimizer, branch['warmup_steps'], start_factor)
            baseline_main = source_scan[1]
            if saved is None:
                save_checkpoint(directory / 'initial_checkpoint.pt', runner, config, 0, warmup)
                event = {'feature_ids': [], 'strategy': 'none'}
                if strategy != 'none':
                    event = resample_features(runner, train.activations, budget, seed, fraction, config['batch_size'],
                                             strategy=strategy, feature_ids=tracked_ids, scan=source_scan)
                event['tracked_feature_ids'] = tracked_ids.tolist()
                write_json_atomic(directory / 'resampling.json', event)
                write_json_atomic(directory / 'branch.json', {**branch, 'seed': seed, 'target_learning_rates': warmup.targets,
                                                             'warmup_start_factor': start_factor})
                baseline_main = source_scan[1]
                write_json_atomic(directory / 'baseline_counts.json', {'main': baseline_main.tolist()})
                save_checkpoint(directory / 'post_intervention_checkpoint.pt', runner, config, 0, warmup)
            else:
                event = saved['event']
                tracked_ids = torch.tensor(event['tracked_feature_ids'], dtype=torch.long)
            # Explicit generators isolate the training order from evaluation and global RNG.
            seed_everything(seed)
            if device.type == 'mps':
                torch.mps.manual_seed(seed)
            loader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'],
                shuffle=config['train_shuffle'], generator=torch.Generator().manual_seed(seed))
            val_loader = torch.utils.data.DataLoader(validation, batch_size=config['batch_size'],
                generator=torch.Generator().manual_seed(seed))
            if saved is None:
                immediate, counts = measure_activity(runner, train, val_loader, baseline_main, tracked_ids, config['batch_size'])
                write_json_atomic(directory / 'post_intervention_metrics.json', immediate)
                write_json_atomic(directory / 'post_intervention_counts.json', counts)
                rows = []
            else:
                runner.model.load_state_dict(saved['model_state_dict'])
                optimizer.load_state_dict(saved['optimizer_state_dict'])
                runner.training_examples_seen = saved['training_examples_seen']
                runner.last_fired_example = saved['last_fired_example'].clone().cpu()
                warmup.targets = saved['target_learning_rates']
                warmup.history = saved['warmup_history']
                rows = saved['history']
                counts = saved['counts']
                loader.generator.set_state(saved['loader_rng'])
                val_loader.generator.set_state(saved['validation_loader_rng'])
                restore_rng_state(saved['rng'])
                publish_metrics(directory, rows, counts, warmup)
            def commit(path, epoch):
                save_checkpoint(path, runner, config, epoch, warmup, history=rows, counts=counts,
                                event=event, loader_rng=loader.generator.get_state(),
                                validation_loader_rng=val_loader.generator.get_state(), rng=capture_rng_state())
            if saved is None:
                commit(resume_path, 0)
            write_json_atomic(directory / 'status.json', {'status': 'running', 'completed_epochs': len(rows)})
            for epoch in range(len(rows) + 1, epochs + 1):
                train_metrics = runner.train_epoch(loader, before_optimizer_step=warmup)
                metrics, counts = measure_activity(runner, train, val_loader, baseline_main, tracked_ids, config['batch_size'])
                ids = event['feature_ids']
                # Keep original pilot fields for existing notebook consumers.
                main, positive = torch.tensor(counts['main']), torch.tensor(counts['positive'])
                val_counts = torch.tensor(counts['validation_count'])
                row = {'epoch': epoch, 'train': train_metrics, **metrics,
                       'learning_rate': optimizer.param_groups[0]['lr'], 'optimizer_updates': len(warmup.history),
                       'replacement_positive': int((positive[ids] > 0).sum()),
                       'replacement_main_active': int((main[ids] > 0).sum()),
                       'replacement_validation_active': int((val_counts[ids] > 0).sum()),
                       'replacement_repeated_10_examples': int((main[ids] >= 10).sum())}
                rows.append(row)
                commit(resume_path, epoch)
                publish_metrics(directory, rows, counts, warmup)
                print(f'{seed} {name} epoch {epoch}: coverage={row["frozen_train_coverage"]:.1%}; '
                      f'tracked active={row["tracked_main_active"]}; lr={row["learning_rate"]:.6g}')
            if saved is None or saved['additional_epoch'] < epochs or not (directory / 'final_checkpoint.pt').is_file():
                commit(directory / 'final_checkpoint.pt', epochs)
            write_json_atomic(directory / 'status.json', {'status': 'complete', 'completed_epochs': epochs})
    write_comparison(output, seeds, branches)
    return output


def publish_metrics(directory, rows, counts, warmup):
    """Rebuild derived artifacts from the authoritative committed checkpoint."""
    write_json_atomic(directory / 'history.json', rows)
    write_json_atomic(directory / 'learning_rates.json', {'rates_per_update': warmup.history})
    write_json_atomic(directory / 'feature_counts.json', counts)
    if rows:
        write_json_atomic(directory / f"feature_counts_epoch_{len(rows):03d}.json", counts)
    for path in directory.glob('feature_counts_epoch_*.json'):
        if int(path.stem.rsplit('_', 1)[1]) > len(rows):
            path.unlink()


def write_comparison(output, seeds, branches):
    """Summarize final epochs without automatically promoting a dictionary."""
    records = []
    for seed in seeds:
        for branch in branches:
            directory = output / f'seed_{seed}' / branch['name']
            row = json.loads((directory / 'history.json').read_text())[-1]
            record = {key: row[key] for key in ('net_active_gain', 'displaced_initially_active',
                'tracked_main_active', 'tracked_validation_active', 'tracked_repeated_10_examples',
                'tracked_validation_repeated_10_examples')}
            records.append(dict(seed=seed, branch=branch['name'], strategy=branch['strategy'],
                warmup_steps=branch['warmup_steps'], validation_loss=row['validation']['loss'],
                checkpoint=str((directory / 'final_checkpoint.pt').resolve()), **record))
    for row in records:
        control = next((r for r in records if r['seed'] == row['seed'] and r['strategy'] == 'none'
                        and r['warmup_steps'] == row['warmup_steps']), None)
        row['matched_control'] = control['branch'] if control else None
        row['relative_validation_loss_change'] = (row['validation_loss'] / control['validation_loss'] - 1
                                                   if control and control['validation_loss'] else None)
    write_json_atomic(output / 'comparison.json', {'branches': records, 'selected_checkpoint': None})
    lines = ['# Refinement comparison', '', 'Final epochs; no automatic checkpoint promotion.', '',
             '| Seed | Branch | Validation MSE | Change vs matched control | Train active | Validation active | Net gain | Displaced |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        delta = row['relative_validation_loss_change']
        change = f'{delta:+.3%}' if delta is not None else 'unmatched'
        lines.append(f"| {row['seed']} | {row['branch']} | {row['validation_loss']:.8g} | {change} | "
                     f"{row['tracked_main_active']} | {row['tracked_validation_active']} | "
                     f"{row['net_active_gain']} | {row['displaced_initially_active']} |")
    lines += ['', 'Activity columns track the same candidate IDs in controls and treatments.',
              'Review repeated firing in comparison.json and rerun downstream feature analysis before selecting a checkpoint.']
    temporary = output / 'comparison.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'comparison.md')


def select_checkpoint(output, checkpoint):
    """Record an explicit selection from a completed comparison, preserving lineage."""
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    report = json.loads((output / 'comparison.json').read_text())
    record = next((row for row in report['branches'] if Path(row['checkpoint']).resolve() == checkpoint), None)
    if record is None or not checkpoint.is_file():
        raise ValueError('Select a final checkpoint listed in this completed comparison.')
    manifest = json.loads((output / 'experiment.json').read_text())
    selection = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
                 'source_checkpoint_sha256': manifest['source_checkpoint_sha256'],
                 'source_run': manifest.get('source_run', manifest.get('source_trial')),
                 'seed': record['seed'], 'branch': record['branch'],
                 'requires_new_feature_analysis': True}
    write_json_atomic(output / 'selection.json', selection)
    return selection


def refinement_cli(root):
    """Shared CLI for the refinement and legacy experiment entry points."""
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--select-checkpoint', type=Path,
                        help='Record an explicit selection from an existing completed comparison.')
    args = parser.parse_args()
    if args.select_checkpoint and (args.resume or args.dry_run):
        parser.error('--select-checkpoint cannot be combined with --resume or --dry-run.')
    settings = yaml.safe_load(args.config.read_text())
    if args.select_checkpoint:
        select_checkpoint(Path(root) / settings['output_dir'], args.select_checkpoint)
    else:
        run_refinement(settings, root, dry_run=args.dry_run, resume=args.resume)
