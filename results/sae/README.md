# SAE training and tuning results

Tuning candidates live inside their study. Only chosen, finished models are
published to `runs/`; ordinary single runs still write there directly. Studies
and final runs use the existing UTC timestamp + Git commit naming convention:

```text
results/sae/musique/
├── tuning/layer_<N>/<study_id>/
│   ├── config.json              # Original search space and optional selection budget
│   ├── manifest.json            # Status, source activation hashes, code provenance
│   ├── trials.json              # Resolved parameters, validation metrics, run links
│   ├── trials.csv               # Flat comparison table, including group and source
│   ├── selection.json           # Suggested winner, or Pareto review / no eligible trial
│   ├── trials/trial_<N>/        # Standard artifacts for each candidate
│   │   └── checkpoints/        # Best inference and last continuation checkpoints
│   ├── finishing/trial_<N>/     # Working copy during continuation (moved on success)
│   └── finalizations/trial_<N>.json # Source trial, published run path, final metrics
└── runs/layer_<N>/
    ├── sae_versions.csv         # Finalized models only
    └── <run_id>/
        ├── config.json          # Resolved settings and link back to the tuning study
        ├── manifest.json        # Activation hashes and continuation provenance
        ├── history.json
        ├── feature_frequencies.json
        └── checkpoints/
            ├── best_checkpoint.pt  # Standard downstream inference checkpoint
            └── last_checkpoint.pt  # Continuation state for tuning trials
```

## Run a study

```bash
python scripts/train_sae.py --hp-tune --config configs/sae_hp_tuning/hp_sweep.yaml --dry-run
python scripts/train_sae.py --hp-tune --config configs/sae_hp_tuning/hp_sweep.yaml
```

`--hp-tune` and `--config` require each other. The YAML owns the activation run,
layer, and training settings; single-run CLI overrides are rejected in tuning
mode. Dry runs expand and validate candidates without loading data or creating
outputs. Each real invocation creates a new study; automatic study restart is
not implemented. A failed trial is recorded and stops the study.

The search merges `Hyperparameters` defaults, then YAML `defaults`, then each
group's settings. `fixed` values remain constant; `grid` expands independent
combinations; each `variants` entry keeps related choices paired. Group sections
cannot overlap, but can override study defaults. Unknown fields, empty spaces,
duplicate candidates, invalid values, and mode-inapplicable settings are errors.
`input_dim` is inferred; `hidden_dim` does not affect the current architecture
and is excluded from search configuration.

Use separate groups for L1/TopK, architecture families, or compatible batch-size
and learning-rate pairs. Use a separate study per activation source and layer.
Use the exported research partitions, not mixed source splits. Trial CSVs include
source, layer, group, seed, epoch budget, and all parameters for filtering and
joining comparisons across studies. Source hashes in study manifests distinguish
changed datasets even if their run names match. Compare configurations at
comparable training budgets; a Pareto label is descriptive, not a statistical
claim across seeds or datasets.

## Compare and choose

Trials use train and validation only: test tensors are neither loaded nor hashed,
and trials are not appended to `sae_versions.csv`. Each trial still chooses its
best epoch by validation training loss, preserving the single-run behavior.
Across trials, compare `val_rec_loss` against `val_mean_active_features`, not
L1-weighted total losses with different coefficients. CSV rows include the
Pareto frontier (no other trial improves one of these metrics without worsening
the other), variance explained, and inactivity diagnostics.

Without `selection`, review the frontier and choose a usable run. To request an
automatic suggestion, add an activity budget, for example:

```yaml
selection:
  max_mean_active_features: 128
```

This number is illustrative. Selection minimizes validation reconstruction loss
among eligible trials; if none qualifies there is no winner. No trial is
finalized automatically. The user can finish the suggested winner or another
run after reviewing the results.

## Finish a chosen trial

```bash
python scripts/train_sae.py \
  --finish-run results/sae/musique/tuning/layer_33/<study_id>/trials/trial_0001 \
  --epochs 400
```

`--epochs` is the **total** budget, not an additional count. This command continues
a copy of the chosen trial from `last_checkpoint.pt`, restoring weights, optimizer, scheduler,
early stopping, activity counters, and RNG states. It checks that training and
validation activation contents have not changed. Additional training requires
the same device type; evaluation-only finalization can use another device.
For resumed training, exact reproducibility across hardware/software environments is not promised.
All settings except the epoch budget are fixed. Existing early stopping remains
in force; a trial already stopped will be finalized without further epochs.
An equal budget can finalize a trial without adding epochs.

After training, the best checkpoint across the full history is evaluated on the
available splits, including test if present. Only then is the working copy moved
to `runs/layer_<N>/<new_run_id>/` and registered in `sae_versions.csv`. Its manifest
links back to the source trial and study. Use the printed final checkpoint path
for downstream work. The standard artifact filenames are unchanged.

The original trial, its weights, comparison tables, and `selection.json` remain
unchanged. `finalizations/trial_<N>.json` holds the published checkpoint path and
final metrics. Several chosen trials may be published from one study. Finishing
an already published trial is rejected; finalized models cannot be continued.
Failed continuation stays under `finishing/`; rerun the same command to retry.
Interrupted publication can also be retried without retraining. No candidate is
automatically deleted.

Existing validation-only runs created under the old layout remain supported;
trials without study metadata continue in place. Existing artifacts are not
migrated automatically. Older checkpoints without `last_checkpoint.pt` cannot
be resumed through this workflow.
