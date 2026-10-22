"""One-time residual-direction resampling (an adaptation of Bricken et al., 2023)."""
import torch


@torch.no_grad()
def scan_features(model, activations, batch_size=256):
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


@torch.no_grad()
def resample_features(runner, activations, budget=512, seed=42,
                      positive_fraction=0.01, batch_size=256):
    if budget < 1 or not 0 < positive_fraction < 0.5:
        raise ValueError('Require positive budget and positive_fraction between 0 and 0.5.')
    model = runner.model
    if runner.last_fired_example is None or runner.training_examples_seen < runner.aux_dead_window_examples:
        raise ValueError('Resampling requires activity state beyond the AuxK warmup.')
    positive, _, errors = scan_features(model, activations, batch_size)
    eligible = runner.training_examples_seen - runner.last_fired_example >= runner.aux_dead_window_examples
    candidates = torch.where(eligible & (positive == 0))[0]
    generator = torch.Generator().manual_seed(seed)
    count = min(budget, len(candidates), int((errors > 0).sum()))
    ids = candidates[torch.randperm(len(candidates), generator=generator)[:count]]
    if not count:
        return {'feature_ids': [], 'source_indices': [], 'seed': seed}
    indices = torch.multinomial(errors.double(), count, replacement=False, generator=generator)
    device = next(model.parameters()).device
    x = activations[indices].to(device)
    _, reconstructed = model(x)
    directions = torch.nn.functional.normalize((x - reconstructed) / model.input_scale, dim=-1)
    encoder = model.encoder_layer[0]
    decoder = model.decoder_layer[0]
    live = positive > 0
    scale = encoder.weight[live.to(device)].norm(dim=-1).mean() if live.any() else encoder.weight.new_tensor(1.)
    # Conservative encoder scale; calibrate bias to a 1% positive rate on TRAIN.
    rows = directions * scale * 0.2
    projections = torch.cat([(batch.to(device) / model.input_scale) @ rows.T
                             for batch in activations.split(batch_size)], dim=0)
    biases = -torch.quantile(projections, 1 - positive_fraction, dim=0)
    ids_device = ids.to(device)
    encoder.weight[ids_device] = rows
    encoder.bias[ids_device] = biases
    decoder.weight[:, ids_device] = directions.T
    # Adam's scalar step remains shared; clear only affected per-parameter moments.
    for parameter, axis in [(encoder.weight, 0), (encoder.bias, 0), (decoder.weight, 1)]:
        for value in runner.optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value.index_fill_(axis, ids_device, 0)
    runner.last_fired_example[ids] = runner.training_examples_seen
    return {'feature_ids': ids.tolist(), 'source_indices': indices.tolist(),
            'seed': seed, 'positive_fraction': positive_fraction,
            'encoder_scale_multiplier': 0.2,
            'method': 'Residual-direction adaptation of Bricken et al. (2023)',
            'reference': 'https://transformer-circuits.pub/2023/monosemantic-features/index.html'}
