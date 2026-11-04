# Pretrained SAE fine-tuning (decoder-only and full)

Run from the repository root in the capstone environment:

```bash
time PYTHONPATH=src python scripts/sae/finetune_pretrained_sae.py \
  --sae-dir results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big \
  --activations-dir artifacts/collection/mixed_final_001/prepared/activations/mixed_final_001/layer_25 \
  --layer 25 \
  --probe-config configs/feature_selection_mixed_final_001_layer25.yaml
```

Defaults: 5 epochs, batch size 256, Adam learning rate 1e-5, patience 2,
seed 42, CUDA when available, otherwise CPU. Only decoder weights and bias update.
The original source is never overwritten. No input scaling, decoder normalization,
L1 penalty, or changes to the frozen JumpReLU encoder/thresholds are applied.

The default activation pattern is `{split}_attempt2.pt`, covering all saved
attempt-2 examples, without behavioral-label filtering. Use `--activation-file
'{split}.pt'` for the combined attempt-1/attempt-2 tensors used by the custom SAE
trainer; omit `--probe-config` in that case because its indices refer to attempt 2.
Only train and validation activation tensors are loaded; test/intervention are sealed.
The optional probe config selects a validation reporting cohort, never training rows.

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

- `baseline.json`: original reconstruction and a bias-only diagnostic fitted on train.
- `history.json`: train/validation reconstruction after each epoch.
- `summary.json`: best epoch, before/after metrics, and frozen-encoder verification.
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
  --probe-config configs/feature_selection_mixed_final_001_layer25.yaml
```

Output goes to `results/sae/full_finetuning/<timestamp>/` with the same files as above, plus
`feature_drift.npz`.

- **Gradients through the gate.** The step `pre > threshold` has zero gradient, so training uses the
  straight-through estimators of Rajamanoharan et al. (2024), as Gemma Scope did
  (`JumpReLUSAE.encoder_ste`). Inference (`encoder`) is unchanged. Thresholds are trained in log
  space, so they stay positive. `--freeze-threshold` keeps them fixed.
- **Bandwidth.** The default is `0.001 * mean ||x|| / sqrt(d)` on train. This is Gemma Scope's
  bandwidth rescaled to raw (unnormalized) activations. Override it with `--bandwidth`.
- **Sparsity.** MSE alone would let the encoder become denser. The loss is
  `MSE / MSE_orig + l0_coef * L0 / L0_orig`, with both reference values measured on train before
  training. The default `--l0-coef 1` trades a 1% rise in L0 against a 1% fall in MSE.
- **Checkpoint selection.** The best checkpoint is the lowest validation MSE among epochs whose
  validation L0 is within `--l0-tolerance` (default 10%) of the original. The original (epoch 0) is
  always eligible. `history.json` marks each epoch's `l0_eligible`.
- **Learning rates.** `--lr` is for the decoder and `--lr-enc` for the encoder and thresholds; it
  defaults to `--lr`.
- **No cache.** Features change every step, so each batch is re-encoded live.

`summary.json` reports `feature_drift`, measured on validation against the original encoder:
- the median per-feature activation correlation, and the fraction of features below 0.9;
- the mean Jaccard overlap of active sets;
- encoder-column and decoder-row cosines.

Per-feature arrays are in `feature_drift.npz`. Every run, in both modes, also reports `l0` and
`dead_fraction` for each split.

**Downstream:** this SAE's features differ from the original's. Re-run feature selection with it
added (as `kind: gemma_scope`, path `.../best`) before using it in causal configs. The original
and decoder-only SAEs share an encoder; this one does not.
