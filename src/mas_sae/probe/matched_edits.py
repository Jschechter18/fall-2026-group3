"""Re-evaluate saved feature selections at equal, error-preserving activation edit sizes.

No SAE training or feature selection is repeated. The original tuned raw readout is refit
once because older feature-selection runs did not persist it. All effects are offline
readout predictions, not regenerated Solver behavior.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm.auto import tqdm

from mas_sae.causal import offline
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.probe import behavior_data as bd, feature_selection as fs, layer_selection as ls
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)


def direction_scores(delta, X, w):
    """Readout-score change per unit relative dose; zero edits are explicitly invalid."""
    norm = np.linalg.norm(np.asarray(delta, dtype=np.float64), axis=1)
    xnorm = np.linalg.norm(np.asarray(X, dtype=np.float64), axis=1)
    valid = (norm > 1e-12) & (xnorm > 0)
    scale = np.divide(xnorm, norm, out=np.zeros_like(norm), where=valid)
    shift = (delta @ w) * scale
    shift[~valid] = np.nan
    return shift, norm, scale, valid


def validate_source(source, cfg, current):
    """Refuse stale rankings or changed data, dictionaries, target, or grouping."""
    for key in ('labels_sha256', 'manifest_sha256', 'excluded_episodes_file_sha256',
                'input_sha256', 'sae_sha256'):
        if source['provenance'].get(key) != current.get(key):
            raise ValueError(f'Source run inputs differ: {key}; use its original config and files.')
    for key in ('layer', 'fit_partition', 'eval_partition', 'target_labels', 'label_column',
                'target_column', 'eligible_column', 'group_column', 'seed'):
        if source['config'].get(key) != getattr(cfg, key):
            raise ValueError(f'Source run config differs: {key}')


def run(cfg, source_run: Path, doses=(0.01, 0.025, 0.05, 0.1), config_path=None):
    doses = np.asarray(doses, dtype=float)
    if not len(doses) or not np.all(np.isfinite(doses) & (doses > 0)):
        raise ValueError('Doses must be finite positive fractions, e.g. 0.01 0.025 0.05 0.1.')
    source_path = Path(source_run) / 'feature_selection_summary.json'
    source = json.loads(source_path.read_text())
    prov = fs.provenance(cfg, config_path)
    validate_source(source, cfg, prov)
    selected, diag, cols = ls.load_inputs(cfg.layer_selection_config())
    X = bd.load_layer_matrix(selected, Path(cfg.capstone_data_root) / cfg.activations_dir /
                             f'layer_{cfg.layer:02d}', cfg.activation_file)
    y = (selected['target_name'] == cfg.target_labels[0]).astype(int).to_numpy()
    groups, _ = bd.episode_groups(selected, cfg.group_column)
    fit = selected[cols.partition].to_numpy() == cfg.fit_partition
    ev = selected[cols.partition].to_numpy() == cfg.eval_partition
    if set(groups[fit]) & set(groups[ev]):
        raise ValueError('Question families overlap between fit and evaluation partitions.')
    if (int(fit.sum()), int(ev.sum())) != (source['readout']['n_fit'], source['readout']['n_eval']):
        raise ValueError('Source run row counts differ.')
    logger.info('Refitting raw readout once at saved C=%s; feature probes and rankings are reused',
                source['readout']['chosen_C'])
    readout = offline._readout_model(source['readout']['chosen_C'], cfg.seed).fit(X[fit], y[fit])
    w, b = ls.raw_unit_weights(readout, 1)
    out = create_versioned_run_dir(Path(cfg.output_dir) / 'matched_edits')
    np.savez(out / 'readout.npz', w=w, b=b, C=source['readout']['chosen_C'],
             scaler_mean=readout.named_steps['scale'].mean_,
             scaler_scale=readout.named_steps['scale'].scale_, positive_class=1)
    Xe = X[ev]
    questions = selected.loc[ev, 'ep_question_id'].astype(str).to_numpy()
    ge = groups[ev]
    if len(np.unique(questions)) != len(questions):
        raise ValueError('Expected one validation row per question.')
    base_score = Xe @ w + b
    base = expit(base_score)
    xnorm = np.linalg.norm(Xe.astype(np.float64), axis=1)
    rows, norm_rows, comparisons = [], [], {}
    expected_mean = source['readout'].get('mean_p_accept_eval')
    if expected_mean is not None and not np.isclose(base.mean(), expected_mean, atol=1e-5, rtol=0):
        raise ValueError('Refitted readout does not reproduce the source validation mean prediction.')

    def evaluate(name, kind, draw, delta):
        shift, norm, scale, valid = direction_scores(delta, Xe, w)
        norm_rows.append(pd.DataFrame({
            'representation': name, 'condition': kind, 'draw': draw, 'question_id': questions,
            'activation_norm': xnorm, 'proposed_edit_norm': norm,
            'proposed_relative_norm': np.divide(norm, xnorm, out=np.zeros_like(norm), where=xnorm > 0),
            'scale_per_unit_dose': scale, 'valid': valid}))
        probs = expit(base_score[None, :] + doses[:, None] * shift[None, :])
        for i, dose in enumerate(doses):
            mask = np.isfinite(probs[i])
            record = {'representation': name, 'condition': kind, 'draw': draw, 'dose': dose,
                      'n_valid': int(mask.sum()), 'n_noop': int((~mask).sum()),
                      'mean_applied_relative_norm': float(dose) if mask.any() else None}
            if mask.any():
                if kind == "targeted":
                    record.update(offline.paired_effect(probs[i, mask], base[mask], questions[mask],
                                                      cfg.n_bootstrap, cfg.n_permutations, cfg.seed, ge[mask]))
                record['predicted_adoption_pct'] = 100 * float(probs[i, mask].mean())
            rows.append(record)
        return probs

    # The raw readout's negative gradient is an intentionally favorable offline benchmark.
    raw_probs = evaluate('Raw probe direction', 'targeted', -1, np.broadcast_to(-w, Xe.shape))
    rng = np.random.default_rng(cfg.seed)
    direction_probs = []
    for d in tqdm(range(cfg.n_random_draws), desc='Random directions', unit='draw'):
        direction_probs.append(evaluate('Random direction', 'random_direction', d,
                               np.broadcast_to(rng.normal(size=Xe.shape[1]), Xe.shape)))
    np.savez_compressed(out / 'predictions_raw_controls.npz', question_id=questions,
                        baseline=base, doses=doses, raw_probe=raw_probs,
                        random_directions=np.stack(direction_probs, axis=2))

    for name, spec in tqdm(cfg.saes.items(), desc='Representations', unit='basis'):
        old = source['saes'][name]
        sae, _ = fs.load_sae(spec, Path(cfg.capstone_data_root), cfg.layer)
        Z = offline.encode(sae, Xe)
        recon = offline.decode(sae, Z)
        targeted = evaluate(name, 'targeted', -1,
                            offline.decode(sae, offline.ablate(Z, old['targeted_features'])) - recon)
        random = []
        for d, features in enumerate(tqdm(old['random_control_draws'], desc=f'{name}: matched controls', unit='draw')):
            random.append(evaluate(name, 'random', d,
                                   offline.decode(sae, offline.ablate(Z, features)) - recon))
        random = np.stack(random, axis=2)  # dose x question x draw
        comparisons[name] = []
        for i, dose in enumerate(doses):
            mask = np.isfinite(targeted[i]) & np.isfinite(random[i]).any(axis=1)
            record = {'dose': float(dose), 'n_valid': int(mask.sum()), 'n_excluded': int((~mask).sum())}
            if mask.any():
                record.update(offline.targeted_vs_random(targeted[i, mask], random[i, mask], base[mask],
                              questions[mask], cfg.n_bootstrap, cfg.n_permutations, cfg.seed,
                              clusters=ge[mask]))
            comparisons[name].append(record)
        np.savez_compressed(out / f'predictions_{name}.npz', question_id=questions,
                            baseline=base, doses=doses, targeted=targeted, random=random)
    pd.DataFrame(rows).to_csv(out / 'matched_effects.csv', index=False)
    norms = pd.concat(norm_rows, ignore_index=True)
    for dose in doses:
        norms[f'applied_norm_at_{dose:g}'] = np.where(norms.valid, dose * norms.activation_norm, 0)
        norms[f'scale_at_{dose:g}'] = dose * norms.scale_per_unit_dose
    norms.to_csv(out / 'edit_norms.csv', index=False)
    summary = {'source_run': str(source_run), 'source_summary_sha256': sha256_file(source_path),
               'provenance': prov, 'data_diagnostics': diag, 'doses': doses.tolist(),
               'readout': {'chosen_C': source['readout']['chosen_C'], 'saved': 'readout.npz',
                           'mean_p_accept_eval': float(base.mean())},
               'targeted_vs_random': comparisons,
               'outcome_note': 'Offline predicted adoption only. All edits preserve reconstruction error. '
               'Zero edits are excluded, never replaced or treated as matched. Comparisons report coverage. '
               'Random sets and feature selections are reused from source. The negative raw readout '
               'direction is an optimistic benchmark for this same readout, not causal evidence.'}
    (out / 'matched_edit_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    logger.info('Wrote matched-size evaluation to %s', out)
    return out
