# Synthetic smoke test, not a result

These files come from the 2026-09-21 pipeline smoke test (commit 965bbe2): a probe trained on synthetic
activations from an untrained SAE, run only to check that the probe, SHAP and LIME code ran end to end.
The metrics (`probe_metrics.json`), SHAP features and LIME explanations describe that synthetic data, not
Gemma or the Solver–Critic episodes. Do not cite them.

The real probe results are produced by `scripts/layer_selection.py` and `scripts/feature_selection.py`
on the collected activations.
