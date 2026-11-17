"""The epoch loop (keep the best eligible epoch) and checkpoint writing."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from mas_sae.experiments.artifacts import write_json_atomic
from mas_sae.sae import diagnostics

from .data import iterate_batches
from .evaluation import evaluate_splits
from .reporting import paired_stats
from .selection import _format, eligibility, within_l0

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from mas_sae.sae.pretrained import JumpReLUSAE

    from .data import Split
    from .trainers import DecoderOnlyTrainer, FullTrainer

logger = logging.getLogger(__name__)


def train(args: argparse.Namespace, model: JumpReLUSAE, source: JumpReLUSAE,
          trainer: DecoderOnlyTrainer | FullTrainer, splits: dict[str, Split],
          original: dict[str, dict[str, Any]], sae_config: dict, out: Path,
          device: str | torch.device) -> int:
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


def save_model(model: JumpReLUSAE, sae_config: dict, destination: Path) -> None:
    """Write params.safetensors + config.json in the Gemma Scope layout (loadable as kind: gemma_scope)."""
    destination.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
              destination / 'params.safetensors')
    write_json_atomic(destination / 'config.json', sae_config)
