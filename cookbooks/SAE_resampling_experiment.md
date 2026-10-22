# Paired resampling pilot

From the repository root, activate the training environment and run:

```bash
PYTHONPATH=src python scripts/experiment_sae_resampling.py --config configs/sae_hp_tuning/layer_25_resampling.yaml --dry-run
PYTHONPATH=src python scripts/experiment_sae_resampling.py --config configs/sae_hp_tuning/layer_25_resampling.yaml
```

The source must be a completed, validation-only TopK trial with a last checkpoint. Outputs are separate; an existing output directory is rejected. The original sweep is untouched. This script does not resume interrupted experiment branches: choose a new output directory to restart.

Both branches restore model, Adam state, and inactivity counters from the same source, then run a fixed additional epoch budget with constant inherited learning rate. Scheduler and early stopping are intentionally disabled in both. Each seed determines a matched new data order, rather than replaying the source's RNG state.

Treatment selects up to 512 eligible features that never become positive on a frozen training scan. Training examples are sampled without replacement proportional to residual squared norm. Decoder columns use normalized residual directions; encoder rows use those directions at 0.2 times the average live encoder row norm. Biases are calibrated to the configured positive fraction (1%) on training inputs. This enables positive activations but does not guarantee winning TopK. Matching Adam moments are zeroed; shared scalar optimizer steps remain unchanged. Replacement inactivity counters start at the current example position, granting one inactivity window before AuxK eligibility returns.

Each branch saves initial, post-intervention, last, and final checkpoints, per-epoch aggregate metrics, final per-feature positive/main counts and validation frequencies. Treatment logs replaced IDs and sampled training indices. Notebook comparison cells report repeated firing, validation coverage, and displacement of initially active features. Test data is never loaded. Use `final_checkpoint.pt` with the existing checkpoint loader for downstream evaluation.

Neuron resampling is motivated by [Bricken et al. (2023), Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Residual-direction initialization, bias calibration, and replacement budget are experimental adaptations, not an exact reproduction of their method.
