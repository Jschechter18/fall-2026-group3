# PR #153 final labels: probe package

Generated 2026-10-29 from the approved PR #153 labels (policy `candidate-final-2026-10-27`, labels sha256 `21a5c7716ae4…`) with `scripts/export_final_labels_for_probes.py` at commit `72d4176`. No label was recomputed.
Tracking: #155 (export), #161 (probe retraining).

## Run
- Collection: `natural_4b_full9_candidates` (candidates presentation), `s3://dats-capstone/shared/group3/results/collection/natural_4b_full9_candidates`.
- Activations: that run's `layer_NN/{train,validation}_attempt2.pt`. The row index is `attempt2_activation_index`, which `align_episodes` checks against `full_index` in `split_manifest.csv`.
- Partitions: the frozen `behavior_v012_official_20261011/partitions.csv`, unchanged.

## Labels (`labels.csv`)
- `primary_target`: 1 = Adoption, 0 = Retention, blank otherwise.
- `eligible_primary`: True for Adoption and Retention only.
- `behavior_label`: all four PR #153 classes, with `label_reason`, `label_step`, `previous_label` and `flags`.

| Partition | Adoption | Retention | Agreement | Out |
|---|---|---|---|---|
| intervention | 534 | 438 | 732 | 276 |
| test | 529 | 416 | 755 | 263 |
| train | 4,146 | 3,465 | 6,063 | 2,114 |
| validation | 625 | 491 | 901 | 382 |

## Loading (layer selection config)
```yaml
results_dir: results/collection/natural_4b_full9_candidates
activations_dir: <local copy of the candidates run's activations>
labels_csv: results/behavior/candidate_final_20261027_probe/labels.csv
manifest_csv: results/behavior/candidate_final_20261027_probe/split_manifest.csv
index_column: full_index
source_splits: [train, validation]
excluded_episodes:
  "2hop__89162_77346": "collected, but the question has no frozen partition (PR #153 label: out)"
  "3hop1__408193_427833_126099": "collected, but the question has no frozen partition (PR #153 label: out)"
```
Defaults stay `target_column: primary_target` and `eligible_column: eligible_primary`.

Checked with the real loader: 22,130 episodes joined with verified activation indices, 2 excluded, and the train/validation targets equal the PR #153 counts. `test` and `intervention` stay sealed.

## Other files
- `excluded_episodes.json`: 2 collected questions without a frozen partition (PR #153 label: out).
- `missing_question_ids.json`: 202 frozen questions without an episode (collection failures; PR #153 label: out).
- `counts.json`: counts, source file hashes and the code commit.
- `SHA256SUMS`: checksums of the files in this folder.

`labels.csv` doesn't carry the v0.x flag columns that `behavior_labels.read_labels` requires, so causal replay keeps its own package.
