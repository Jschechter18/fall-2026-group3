# Feature selection: add retention features (small change to existing code)

Keep the diff small. Reuse the existing feature-selection pipeline
(`src/mas_sae/probe/feature_selection.py`, `scripts/feature_selection.py`) and change only what
is listed below.

## Changes

1. **New config** `configs/feature_selection/feature_selection_intervention_layer25.yaml`.
   Copy `feature_selection_pretrained_adapted_layer25.yaml`, then:
   - drop the TopK SAE;
   - keep Raw activations, L1 SAE, Pretrained SAE, and Fine-tuned decoder
     (`results/sae/decoder_finetuning/20261104T164310458065Z/best`);
   - add Fully fine-tuned (`results/sae/full_finetuning/20261104T151230848807Z/best`,
     kind `gemma_scope`);
   - give it a new `output_dir`.
2. **Retention features.** In `analyse_sae`, the adoption features are selected by
   `promote = ranking[ranking.coef_toward_accept > 0]`. Next to that, select retention features the
   same way with `< 0`, and give them matched random draws using the same matching call already
   used for the adoption set.
   - Both sets' draws must exclude the features of both targeted sets.
   - Add `retention_features` and `random_control_draws_retention` to the summary. Leave the
     existing keys unchanged.
3. **`feature_sets.json`.** Write it in the run directory with one entry per representation:
   - its path and checkpoint sha256;
   - its adoption features and retention features, each with their coefficients;
   - the first 3 random draws for each set.
4. **Test.** Add one unit test showing that the two sets are disjoint and that every draw excludes
   both sets.

## Run it and report

Run `python scripts/feature_selection.py --config configs/feature_selection/feature_selection_intervention_layer25.yaml`, then report:
- for each representation: the probe's balanced accuracy, the adoption features and the retention
  features;
- any representation with fewer than 10 retention features;
- whether Pretrained and Fine-tuned decoder selected identical features (they share an encoder,
  so they should).

Stop there.
