"""The two fine-tuning modes. Each trainer owns its optimizer and its loss.

Both expose what loop.train() uses: `logged_terms`, `optimizer`, `loss(split, rows, batch)`,
`after_step()` and `manifest_fields()`.
"""

import torch
from torch.nn.functional import mse_loss

from .data import features_for

GEMMA_SCOPE_BANDWIDTH = 0.001                      # STE bandwidth Gemma Scope used on unit-scaled inputs


def encoder_lr(args):
    """Learning rate for the encoder and thresholds in full mode: --lr-enc, else --lr."""
    return args.lr if args.lr_enc is None else args.lr_enc


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


def default_bandwidth(activations, batch_size):
    """Gemma Scope's STE bandwidth, rescaled from their unit-scaled inputs to our raw activations.

    Gemma Scope scaled its inputs so that mean ||x|| = sqrt(d) and used a bandwidth of 0.001.
    For raw inputs, the equivalent bandwidth is 0.001 * mean ||x|| / sqrt(d).
    """
    total_norm = sum(activations[start:start + batch_size].double().norm(dim=1).sum().item()
                     for start in range(0, len(activations), batch_size))
    mean_norm = total_norm / len(activations)
    return GEMMA_SCOPE_BANDWIDTH * mean_norm / activations.shape[1] ** 0.5
