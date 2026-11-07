# Pretrained SAE fine-tuning (decoder-only and full)

Run from the repository root in the capstone environment:

```bash
time PYTHONPATH=src python scripts/sae/finetune_pretrained_sae.py \
  --sae-dir results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big \
  --activations-dir artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25 \
  --layer 25 \
  --probe-config configs/feature_selection/feature_selection_mixed_final_001_layer25.yaml
```

Defaults: 5 epochs, batch size 256, Adam decoder learning rate 1e-5, patience 5,
seed 42, CUDA when available, otherwise CPU. Only decoder weights and bias update.
The original source is never overwritten. No input scaling, decoder normalization,
L1 penalty, or changes to the frozen JumpReLU encoder/thresholds are applied.

The default activation pattern is `{split}_attempt2.pt`, covering all saved
attempt-2 examples, without behavioral-label filtering. Use `--activation-file
'{split}.pt'` for the combined attempt-1/attempt-2 tensors used by the custom SAE
trainer; omit `--probe-config` in that case because its indices refer to attempt 2.
Only train and validation activation tensors are loaded; test/intervention are sealed.
The optional probe config selects a validation reporting cohort, never training rows.

## Code layout

The script is only the entry point. It holds `run()`, argument checks, model loading, the run
directory and manifest, and the command line. `run()` calls the steps as modules of
`mas_sae.sae.finetuning`:

| Module | Contents |
|---|---|
| `data` | `Split`, `load_splits`, `probe_cohort_rows`, `iterate_batches`, `features_for` |
| `cache` | encoder feature cache (decoder mode), `encoder_digest` |
| `evaluation` | `evaluate`, `evaluate_splits`, bias-only reference |
| `trainers` | `DecoderOnlyTrainer`, `FullTrainer`, `default_bandwidth`, `encoder_lr` |
| `loop` | `train` (the epoch loop and checkpoint selection), `save_model` |
| `selection` | `within_l0`, `eligibility`, `stability_limits` |
| `reporting` | dictionary health and stability, `feature_health.npz` |

The metrics themselves are in `mas_sae.sae.diagnostics`.

## Encoder cache

The first run writes float32 features to `artifacts/cache/pretrained_decoder/`.
Later runs with identical activation contents and encoder tensors reuse these caches,
including runs with different decoder weights or learning rates. Each epoch reads
batches through a memory map; the entire feature matrix is not loaded onto the GPU.
At width 16,384, dense storage costs 65,536 bytes per example: approximately 5.2 GB
for 80,000 examples, plus validation. There must be enough local disk space.

Cache keys include encoder weights, bias, thresholds, tensor shapes/dtypes, and
activation-file checksums. An atomic directory rename publishes only complete caches.
Concurrent builds of the same cache are rejected using a lock. An interrupted build
may leave a lock and hidden `.building-*` directory; remove those only after checking
that no process is still using them. Cache hits avoid encoding, but still load and
hash input files to verify identity.

## Results

Each run creates `results/sae/decoder_finetuning/<timestamp>/`:

- `baseline.json`: original reconstruction, a bias-only diagnostic fitted on train, and the
  original's dictionary health (see [Health and stability metrics](#health-and-stability-metrics)).
- `history.json`: train/validation reconstruction after each epoch, a compact validation health and
  stability block, and whether the epoch was `eligible` (with `ineligible_reasons`).
- `summary.json`: best epoch, before/after metrics, frozen-encoder verification, `health` (original
  vs adapted) and `stability` (adapted vs original, on validation).
- `feature_health.npz`: per-feature arrays (see below).
- `manifest.json`: settings, source/input hashes, cache locations, and code provenance.
- `best/config.json` and `best/params.safetensors`: compatible with `kind: gemma_scope`.

The unmodified model is epoch 0 and remains the selected checkpoint if training never
improves validation MSE. Variance explained uses each evaluation cohort's own mean
consistently before and after adaptation. The best checkpoint is selected using full
validation MSE, not the optional label-filtered cohort.

Encoder features and feature-probe inputs are unchanged. Existing feature rankings
remain applicable, but the decoder changes intervention directions and magnitudes.
Re-evaluate interventions with the new decoder. The old `--matched-from` evaluation
intentionally rejects changed checkpoint hashes; it must not silently reuse old
provenance with adapted weights.

# Full (encoder + decoder) adaptation

`--train full` starts again from the **original** Gemma Scope weights (not the decoder-only
checkpoint). It trains `w_enc`, `b_enc`, the JumpReLU `threshold` and the decoder. The result is a
third, separate checkpoint, kept beside the original and the decoder-only SAE:

```bash
time PYTHONPATH=src python scripts/sae/finetune_pretrained_sae.py --train full \
  --sae-dir results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big \
  --activations-dir artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25 \
  --layer 25 \
  --probe-config configs/feature_selection/feature_selection_mixed_final_001_layer25.yaml
```

Output goes to `results/sae/full_finetuning/<timestamp>/` with the same files as above.

- **Gradients through the gate.** The step `pre > threshold` has zero gradient, so training uses the
  straight-through estimators of Rajamanoharan et al. (2024), as Gemma Scope did
  (`JumpReLUSAE.encoder_ste`). Inference (`encoder`) is unchanged. Thresholds are trained in log
  space, so they stay positive. `--freeze-threshold` keeps them fixed.
- **Bandwidth.** The default is `0.001 * mean ||x|| / sqrt(d)` on train. This is Gemma Scope's
  bandwidth rescaled to raw (unnormalized) activations. Override it with `--bandwidth`.
- **Sparsity.** MSE alone would let the encoder become denser. The loss is
  `MSE / MSE_orig + l0_coef * L0 / L0_orig`, with both reference values measured on train before
  training. The default `--l0-coef 0.1` makes a 10% rise in L0 cost as much as a 1% fall in MSE.
- **Checkpoint selection.** The fine-tune is meant to be light, so the best checkpoint is the lowest
  validation MSE among *eligible* epochs. An epoch is eligible when, on validation, all of these hold:
  - L0 is within `--l0-tolerance` (default 10%) of the original;
  - the median per-feature activation correlation with the original is at least
    `--min-activation-correlation` (default 0.95; 0 disables);
  - the mean active-set Jaccard with the original is at least `--min-jaccard` (default 0.85; 0 disables);
  - the number of live features changed by at most `--max-alive-change` (default 5%; `inf` disables).

  The original (epoch 0) is always eligible. `history.json` records `eligible`, the failed
  `ineligible_reasons` and, as before, `l0_eligible`. In decoder mode the features never change, so
  the stability limits always pass.
- **Learning rates.** `--lr` is for the decoder and `--lr-enc` for the encoder and thresholds.
  The latter defaults to `1e-6` so the sensitive encoder and gate parameters move conservatively.
- **Early stopping.** `--patience` defaults to 5 epochs without a new eligible best checkpoint.
- **No cache.** Features change every step, so each batch is re-encoded live.

**Downstream:** this SAE's features differ from the original's. Re-run feature selection with it
added (as `kind: gemma_scope`, path `.../best`) before using it in causal configs. The original
and decoder-only SAEs share an encoder; this one does not.

# Health and stability metrics

MSE alone cannot tell whether a fine-tune stayed light. Both modes therefore report the following,
computed by `mas_sae.sae.diagnostics` (`stats`, `health`, `geometry`, `stability`). Health is
reported per split for the original (`baseline.json`) and for the kept checkpoint (`summary.json`).
Stability is measured on validation. Features are matched by index, because fine-tuning starts from
the original weights.

**Dictionary utilization** (`health.<split>.utilization`)
- The fraction of features that fire at least once, and the fraction that fire in at least 0.01%,
  0.1% and 1% of examples.
- Firing-frequency quantiles (p50/p90/p99/max), over all features and over live features.

**Usage concentration** (`concentration`)
- Entropy of the aggregate activation-mass distribution `p_i = Σ z_i / Σ z`, and the effective
  dictionary size `exp(entropy)`. Both are also given for firing counts.
- The share of mass held by the top 1%, 5% and 10% of all features. This detects a small subset
  monopolizing the representation.

**Activation health** (`activations`)
- The mean nonzero magnitude.
- Approximate quantiles of nonzero magnitudes, read from a 0.01-decade log histogram.
- Quantiles of per-feature mean magnitudes.
- The count of features firing on more than 50% and more than 90% of examples (near-always-active).

**Dictionary geometry and thresholds** (`health.parameters`, weights only)
- Decoder-row norm quantiles.
- The mean pairwise |cosine|.
- Each feature's nearest-neighbour |cosine|: quantiles, plus counts over 0.9 and 0.95
  (near-duplicates). These are given over all features and over features live on train.
- Encoder–decoder alignment `cos(w_enc[:, i], w_dec[i])`.
- Threshold quantiles.

**Stability against the original** (`summary.stability`)
- Active-set Jaccard per example: the mean and the p1/p5/p10/p50 tail.
- Activation-correlation quantiles (p1/p5/p10/p50), and the fraction below 0.9.
- The change in firing frequency per feature: `log2` ratio quantiles, and the count that changed by
  more than 2×.
- Died (live → dead) and revived (dead → live) counts.
- Encoder and decoder cosines and threshold-ratio quantiles: how far the thresholds moved.
- `split_merge`, a heuristic built on a cross-correlation between live original and live adapted
  features.
  - *Split:* an original feature that is no longer intact (same-index correlation below 0.9) and that
    correlates at least 0.5 with two or more adapted features.
  - *Merge:* the reverse, an adapted feature that is no longer intact and that correlates at least 0.5
    with two or more original features.

  Pairs that were already correlated in the original are not counted, so comparing the original with
  itself gives zero.

Each epoch, `history.json` logs a compact subset on validation: live fraction, effective size, top-1%
share, dense count, median correlation, Jaccard mean/p10 and died/revived. Checkpoint eligibility
reads these.

`feature_health.npz` holds the per-feature arrays:
- for each SAE, with the prefixes `original_` and `adapted_`: frequency, mean and max activation
  (validation), decoder norm, nearest-neighbour cosine, alignment and threshold;
- the drift between them: `activation_correlation`, `frequency_log2_ratio`, `died`, `revived`,
  `encoder_cosine`, `decoder_cosine`, `threshold_ratio`, `split`, `merged`.
