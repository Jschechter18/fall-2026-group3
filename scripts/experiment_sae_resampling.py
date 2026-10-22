"""Fork a validation-only TopK trial into paired fixed-budget continuations."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.dataloader import ActivationDataset
from mas_sae.sae.resampling import scan_features, resample_features


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load(args.config.read_text())
    source = root / settings['source_trial']
    output = root / settings['output_dir']
    config = json.loads((source / 'config.json').read_text())
    epochs = settings.get('additional_epochs', 20)
    budget = settings.get('resampling_budget', 512)
    fraction = settings.get('positive_fraction', .01)
    seeds = settings.get('seeds', [42])
    if config['sparsity_mode'] != 'topk' or config.get('evaluate_test', True):
        raise ValueError('Source must be a validation-only TopK trial.')
    if type(epochs) is not int or epochs < 1 or type(budget) is not int or budget < 1:
        raise ValueError('Epoch and resampling budgets must be positive integers.')
    if not 0 < fraction < .5 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require 0 < positive_fraction < .5 and unique nonempty seeds.')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    checkpoint = source / 'checkpoints/last_checkpoint.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f'{source}\n-> {output}\n{len(seeds)} seed(s), two branches, {epochs} additional epochs')
    if args.dry_run:
        return
    location = root / 'data/activations' / config['activation_run_name'] / f"layer_{config['layer']:02d}"
    train = ActivationDataset('train', location)
    validation = ActivationDataset('validation', location)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    output.mkdir(parents=True)
    write_json(output / 'experiment.json', {**settings, 'source_config': config,
               'schedule': 'Constant inherited learning rate; scheduler and early stopping disabled in both branches'})
    for seed in seeds:
        for branch in ['control', 'resampling']:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = SparseAutoencoder(config['input_dim'], config['hidden_dim'], config['latent_dim'],
                                      'topk', config['top_k']).to(device)
            model.load_state_dict(state['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            optimizer.load_state_dict(state['optimizer'])
            runner = ModelRunner(model, config['sparsity_coefficient'], optimizer,
                                 training_mean=train.activations.float().mean(0),
                                 inactivity_window_examples=config['inactivity_window_examples'],
                                 aux_coefficient=config['aux_coefficient'], aux_k=config['aux_k'],
                                 aux_dead_window_examples=config['aux_dead_window_examples'])
            runner.training_examples_seen = state['training_examples_seen']
            runner.last_fired_example = state['last_fired_example'].clone()
            directory = output / f'seed_{seed}' / branch
            directory.mkdir(parents=True)
            torch.save(state, directory / 'initial_checkpoint.pt')
            _, baseline_main, _ = scan_features(model, train.activations, config['batch_size'])
            event = resample_features(runner, train.activations, budget, seed, fraction, config['batch_size']) if branch == 'resampling' else {'feature_ids': []}
            write_json(directory / 'resampling.json', event)
            write_json(directory / 'baseline_counts.json', {'main': baseline_main.tolist()})
            def save(name):
                torch.save({'model_state_dict': model.state_dict(), 'model_config': config,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_examples_seen': runner.training_examples_seen,
                            'last_fired_example': runner.last_fired_example}, directory / name)
            save('post_intervention_checkpoint.pt')
            # Reset data-order RNG after intervention so both branches see identical batches.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            loader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=config['train_shuffle'])
            val_loader = torch.utils.data.DataLoader(validation, batch_size=config['batch_size'])
            rows = []
            for epoch in range(1, epochs + 1):
                train_metrics = runner.train_epoch(loader)
                val_metrics = runner.val_epoch(val_loader)
                positive, main, _ = scan_features(model, train.activations, config['batch_size'])
                ids = event['feature_ids']
                row = {'epoch': epoch, 'train': train_metrics, 'validation': val_metrics,
                       'frozen_train_coverage': (main > 0).float().mean().item(),
                       'displaced_initially_active': int(((baseline_main > 0) & (main == 0)).sum()),
                       'replacement_positive': int((positive[ids] > 0).sum()),
                       'replacement_main_active': int((main[ids] > 0).sum()),
                       'replacement_validation_active': int((runner.last_feature_frequencies[ids] > 0).sum()),
                       'replacement_repeated_10_examples': int((main[ids] >= 10).sum())}
                rows.append(row)
                write_json(directory / 'history.json', rows)
                write_json(directory / 'feature_counts.json', {'positive': positive.tolist(), 'main': main.tolist(),
                           'validation_frequency': runner.last_feature_frequencies.tolist()})
                save('last_checkpoint.pt')
                print(f'{seed} {branch} epoch {epoch}: coverage={row["frozen_train_coverage"]:.1%}; replacements active={row["replacement_main_active"]}')
            save('final_checkpoint.pt')


if __name__ == '__main__':
    main()
