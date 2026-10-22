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

## Initialization and warmup comparison

```bash
PYTHONPATH=src python scripts/experiment_sae_resampling.py --config configs/sae_hp_tuning/layer_25_resampling_comparison.yaml --dry-run
PYTHONPATH=src python scripts/experiment_sae_resampling.py --config configs/sae_hp_tuning/layer_25_resampling_comparison.yaml
```

The comparison uses a separate output directory with five branches: `control`, `warmup_control`, `residual`, `residual_warmup`, and `bricken_warmup`. The legacy two-branch YAML remains supported. All branches restore the same source model/Adam state. Treatments share replacement IDs per seed; controls track those IDs without replacing them. Training loaders have explicit matched generators independent of diagnostic scans. Repeated seeds vary continuation order and replacement sampling; they do not constitute independent source SAE training seeds.

Warmup is applied before optimizer updates: update 1 uses 10% of the inherited rate, update 250 uses 100%, and subsequent updates retain that rate. Warmup is part of the 20-epoch budget. No plateau scheduler or early stopping is used. CUDA, Apple MPS, and CPU are selected in that order; quantile calibration is performed on CPU for MPS compatibility. GPU arithmetic is not guaranteed bitwise reproducible.

### Reference recipe and adaptations

Bricken et al.'s neuron-resampling section specifies loss-squared input sampling, normalized input decoder directions, encoder norms at 0.2 times the live-feature average, zero encoder biases, and modified optimizer-state resets. We implement those initialization ingredients as `bricken`. For TopK, per-input main loss is reconstruction MSE; AuxK is excluded from sampling. Normalizing all losses by the same constant does not change sampling probabilities.

This is an adaptation, not a full reproduction: our global RMS scaling, no decoder-bias subtraction, TopK/AuxK, full fixed training data instead of the paper's random subset, shared eligible/never-positive candidate rule, input sampling without replacement, one-time timing, and retained shared scalar Adam step differ. The reference scale uses features outside the inactivity mask; the legacy residual strategy retains its positive-on-training scale reference. The 250-step warmup is our pilot choice, motivated by the post-resampling warmup in [Marks et al., Appendix B.1.2](https://arxiv.org/html/2403.19647v3#A2.SS1.SSS2), rather than a claim about Bricken's schedule.

### Artifacts and interpretation

`experiment.json` records source checkpoint/data hashes, git provenance, device, branch definitions, and schedule. `candidates_seed_*.json` records shared IDs. Every branch preserves initial, post-intervention, and final checkpoints, immediate activity metrics, each epoch's per-feature counts, and the rates actually used at every update. Repeated validation activity means firing on at least ten validation examples; raw counts allow other thresholds. Controls' `tracked_*` metrics measure the shared candidate IDs, while `replacement_*` metrics remain zero when nothing was replaced.

Use the new notebook section to distinguish immediate activation from sustained revival and compare paired branch differences. Primary pilot outcomes are repeated validation activity of tracked candidates and net training coverage gain. Report validation coverage, displacement, and fidelity alongside them. Causal utility still requires the existing feature-selection and intervention pipelines. Branches are not automatically selected by reconstruction loss. Interrupted experiment continuation is not supported; use a new output directory to restart.
