"""One-time feature revival strategies for controlled SAE continuations."""
import torch

REFERENCE = 'https://transformer-circuits.pub/2023/monosemantic-features/index.html'


@torch.no_grad()
def scan_features(model, activations, batch_size=256):
    """Count feature activity and reconstruction errors with frozen weights.

    Parameters
    ----------
    model : SparseAutoencoder
        SAE whose normalization and TopK selection are reused.
    activations : torch.Tensor
        Nonempty training activation matrix in original input units.
    batch_size : int, optional
        Number of examples processed together, by default 256.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        CPU tensors containing per-feature positive counts, main-path firing
        counts, and per-example reconstruction squared errors.
    """
    if len(activations) == 0 or batch_size < 1:
        raise ValueError('Require nonempty activations and a positive batch size.')
    device = next(model.parameters()).device
    positive = torch.zeros(model.latent_dim, dtype=torch.long, device=device)
    main = torch.zeros_like(positive)
    errors = []
    for batch in activations.split(batch_size):
        x = batch.to(device)
        sparse, reconstructed, pre = model.forward_with_pre_topk(x)
        positive += (pre > 0).sum(0)
        main += (sparse > 0).sum(0)
        errors.append(((x - reconstructed) / model.input_scale).square().sum(-1).cpu())
    return positive.cpu(), main.cpu(), torch.cat(errors)


def select_resampling_features(runner, positive_counts, errors, budget=512, seed=42):
    """Select a shared set of eligible, never-positive replacement slots.

    Parameters
    ----------
    runner : ModelRunner
        Runner with saved training-example and inactivity counters.
    positive_counts : torch.Tensor
        Frozen training-scan positive counts, one per feature, on CPU.
    errors : torch.Tensor
        Per-example squared reconstruction errors from the same scan.
    budget : int, optional
        Maximum replacement count, by default 512.
    seed : int, optional
        Seed for a local CPU generator, by default 42.

    Returns
    -------
    torch.Tensor
        CPU feature IDs, capped by eligible slots and positive-error examples.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError('Require a positive integer resampling budget.')
    if runner.last_fired_example is None or runner.training_examples_seen < runner.aux_dead_window_examples:
        raise ValueError('Resampling requires activity state beyond the AuxK warmup.')
    if not torch.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('Reconstruction errors must be finite and nonnegative.')
    eligible = runner.training_examples_seen - runner.last_fired_example.cpu() >= runner.aux_dead_window_examples
    candidates = torch.where(eligible & (positive_counts == 0))[0]
    count = min(budget, len(candidates), int((errors > 0).sum()))
    generator = torch.Generator().manual_seed(seed)
    return candidates[torch.randperm(len(candidates), generator=generator)[:count]]


@torch.no_grad()
def resample_features(runner, activations, budget=512, seed=42,
                      positive_fraction=0.01, batch_size=256, *,
                      strategy='residual', feature_ids=None, scan=None):
    """Reinitialize unused encoder rows, biases, and decoder columns once.

    Parameters
    ----------
    runner : ModelRunner
        Training runner whose model, Adam moments, and activity counters change.
    activations : torch.Tensor
        Training inputs only, in original units.
    budget : int, optional
        Maximum replacement count when IDs are not supplied, by default 512.
    seed : int, optional
        Local sampling seed, by default 42.
    positive_fraction : float, optional
        Training positivity target for residual bias calibration, by default .01.
    batch_size : int, optional
        Frozen-scan batch size, by default 256.
    strategy : {'residual', 'bricken'}, optional
        Residual directions with calibrated biases, or normalized input
        directions with zero biases and squared-loss sampling weights.
    feature_ids : torch.Tensor, optional
        Shared replacement slots selected from the untouched source model.
    scan : tuple, optional
        Cached source scan returned by scan_features.

    Returns
    -------
    dict
        Replaced feature IDs, sampled input indices, and initialization provenance.
        Only corresponding Adam moments reset; shared scalar steps are retained.
        The Bricken strategy adapts the published recipe to TopK/AuxK, global RMS
        normalization, a shared candidate set, and the full training dataset.
    """
    if strategy not in {'residual', 'bricken'}:
        raise ValueError('Unknown resampling strategy.')
    if not 0 < positive_fraction < 0.5:
        raise ValueError('positive_fraction must be between 0 and 0.5.')
    model = runner.model
    if scan is None:
        scan = scan_features(model, activations, batch_size)
    positive, _, errors = scan
    candidates = select_resampling_features(runner, positive, errors, budget, seed)
    ids = candidates if feature_ids is None else torch.as_tensor(feature_ids, dtype=torch.long).cpu()
    if ids.ndim != 1 or len(ids.unique()) != len(ids) or len(ids) > budget:
        raise ValueError('Feature IDs must be unique and within the replacement budget.')
    if len(ids) and ((ids < 0).any() or (ids >= model.latent_dim).any()):
        raise ValueError('Feature IDs are outside the dictionary.')
    eligible = runner.training_examples_seen - runner.last_fired_example.cpu() >= runner.aux_dead_window_examples
    if len(ids) and not (eligible[ids] & (positive[ids] == 0)).all():
        raise ValueError('Replacement IDs must be eligible and never positive on the source scan.')
    event = {'feature_ids': ids.tolist(), 'source_indices': [], 'seed': seed,
             'strategy': strategy, 'positive_fraction': positive_fraction if strategy == 'residual' else None,
             'encoder_scale_multiplier': 0.2, 'reference': REFERENCE,
             'sampling_weight': 'reconstruction_sse' if strategy == 'residual' else 'reconstruction_mse_squared',
             'optimizer_reset': 'Selected moments only; shared Adam step retained'}
    if not len(ids):
        return event
    generator = torch.Generator().manual_seed(seed)
    # Normalizing first avoids overflow; constant reductions do not change probabilities.
    weights = errors.double() / errors.max()
    if strategy == 'bricken':
        weights = weights.square()
        weights[activations.norm(dim=-1).cpu() == 0] = 0
    if int((weights > 0).sum()) < len(ids):
        raise ValueError('Too few nonzero sampling directions for the shared replacement set.')
    indices = torch.multinomial(weights, len(ids), replacement=False, generator=generator)
    device = next(model.parameters()).device
    x = activations[indices].to(device)
    vectors = x / model.input_scale
    if strategy == 'residual':
        _, reconstructed = model(x)
        vectors = (x - reconstructed) / model.input_scale
    directions = torch.nn.functional.normalize(vectors, dim=-1)
    encoder, decoder = model.encoder_layer[0], model.decoder_layer[0]
    live = (positive > 0) if strategy == 'residual' else ~eligible
    if not live.any():
        raise ValueError('Cannot calibrate encoder scale without live reference features.')
    scale = encoder.weight[live.to(device)].norm(dim=-1).mean()
    rows = directions * scale * 0.2
    biases = torch.zeros(len(ids), device=device)
    if strategy == 'residual':
        # Quantile on CPU also supports Apple MPS, without relying on MPS quantile.
        projections = torch.cat([((batch.to(device) / model.input_scale) @ rows.T).cpu()
                                 for batch in activations.split(batch_size)], dim=0)
        biases = -torch.quantile(projections, 1 - positive_fraction, dim=0).to(device)
    if not torch.isfinite(rows).all() or not torch.isfinite(biases).all() or scale <= 0:
        raise ValueError('Replacement initialization must be finite with nonzero encoder scale.')
    ids_device = ids.to(device)
    encoder.weight[ids_device] = rows
    encoder.bias[ids_device] = biases
    decoder.weight[:, ids_device] = directions.T
    for parameter, axis in [(encoder.weight, 0), (encoder.bias, 0), (decoder.weight, 1)]:
        for value in runner.optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value.index_fill_(axis, ids.to(value.device), 0)
    runner.last_fired_example[ids] = runner.training_examples_seen
    event['source_indices'] = indices.tolist()
    event['encoder_reference_norm'] = float(scale)
    event['sampling_replacement'] = False
    event['direction'] = 'input' if strategy == 'bricken' else 'residual'
    return event


class LinearWarmup:
    """Set each optimizer group's learning rate before an optimizer update.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer containing inherited target learning rates.
    steps : int, optional
        Updates spanning the ramp, by default 0 (constant learning rate).
    start_factor : float, optional
        Fraction of target rate used on the first update, by default .1.
    """

    def __init__(self, optimizer, steps=0, start_factor=0.1):
        if type(steps) is not int or steps < 0 or not 0 < start_factor <= 1:
            raise ValueError('Require nonnegative integer warmup steps and 0 < start_factor <= 1.')
        if steps == 1:
            raise ValueError('Warmup needs at least two updates to include both endpoints.')
        self.optimizer = optimizer
        self.steps = steps
        self.start_factor = start_factor
        self.targets = [group['lr'] for group in optimizer.param_groups]
        self.history = []

    def __call__(self):
        """Apply and record the learning rates for the next optimizer update.

        Returns
        -------
        None
            Updates optimizer groups in place; history records every applied rate.
        """
        progress = min(len(self.history) / (self.steps - 1), 1) if self.steps else 1
        factor = self.start_factor + (1 - self.start_factor) * progress
        rates = [rate * factor for rate in self.targets]
        for group, rate in zip(self.optimizer.param_groups, rates):
            group['lr'] = rate
        self.history.append(rates)
