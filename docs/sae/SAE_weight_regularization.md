# L1 weight-regularization pilot

Run from the repository root in the training environment:

```bash
PYTHONPATH=src python scripts/sae/train_sae.py --hp-tune --config configs/sae_hp_tuning/layer_25_l1_weight_l2.yaml --dry-run
PYTHONPATH=src python scripts/sae/train_sae.py --hp-tune --config configs/sae_hp_tuning/layer_25_l1_weight_l2.yaml
```

This is an explicit L2 weight penalty, not AdamW weight decay. It is supported
for both L1 and TopK, defaults to zero, and excludes biases:

`training_loss = existing_training_loss + weight_l2_coefficient * (sum(encoder_weight**2) + sum(decoder_weight**2))`

Decoder columns remain unit norm, making their squared-norm sum constant at
optimizer-step boundaries. The effective shrinkage therefore acts on the encoder.
Adam itself retains its existing zero weight decay. Coefficients multiply a sum,
not a mean; do not transfer them across model sizes or normalization conventions
without checking relative loss scales.

The four-trial pilot fixes the previous layer-25 L1 baseline (learning rate 0.0001,
activation sparsity coefficient 2.0, 12,888 latents, tied initialization, seed 42)
and screens coefficients 0, 1e-8, 1e-6, and 1e-4. Intermediate values can be
tested after observing the response; prior L1 trials do not establish which L2
coefficients will work. The maximum is 200 epochs per
trial with the existing validation scheduler and early stopping. It uses training
and validation only, with no resampling or automatic model selection.

`history.json` records `train_weight_l2_loss` and
`train_weighted_weight_l2_loss` alongside reconstruction and activation sparsity.
The existing validation objective excludes the weight penalty: reconstruction
plus the L1 activation penalty for L1, or reconstruction for TopK. It controls
checkpoint selection, scheduling, and early stopping. Across trials, compare
validation reconstruction, actual active features per input, and coverage rather
than the regularized training total. Regularization may reduce feature activity;
more dead features alone neither establishes failure nor success.

This one-seed sweep screens coefficients. Follow promising settings with
independent training seeds, cross-seed feature matching, and downstream causal
evaluation. A smaller unregularized dictionary is a useful later control.

Motivation: [Stable and Steerable Sparse Autoencoders with Weight Regularization](https://arxiv.org/abs/2603.04198).
This is an adaptation to our existing L1 pipeline, not a reproduction of the
paper's language-model TopK experiments.
