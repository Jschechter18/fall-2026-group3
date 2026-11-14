# Collection tooling and earlier runs

> The final results use the `mixed_final_001` collection
> (`configs/collection/distributed_final.yaml`); see the
> [experimental pipeline](../../README.md#1-collection). This page describes the collection tooling,
> its condition policy and the earlier single-machine runs.

## Data Collection

After activating the Conda environment and installing the project, run the following command from the repository root:

```bash
python scripts/data/get_musique_dataset.py
```

## Activation Collection

For new mixed-dataset runs across EC2 workers, follow the
[distributed collection workflow](../distributed_collection.md): prepare one
manifest, collect assigned chunks, merge, then split/export. Intervention
membership is reserved upfront, and its baselines are included in collection.
See also the [two-EC2 smoke-test commands](../distributed_collection_smoke_test.md).

### Existing single-machine MuSiQue runs

The following commands and paths describe the supported legacy collection path.
Run from the repository root:

    python scripts/collection/collect_activations.py --config configs/archive/collection/v1/smoke_train.yaml

To run 100 existing synthetic sample questions through the solver–critic pipeline:

```bash
PYTHONPATH=src python scripts/collection/collect_activations.py \
  --config configs/experiments/synthetic_pilot/synthetic_sample_100q_pilot.yaml
```

This pilot uses Gemma 3 4B as solver, Qwen3 4B as critic, the natural critic
condition, candidate presentation for the solver's second answer, and nine
activation layers. It selects 34 two-hop, 33 three-hop, and 33 four-hop questions
with seed 42. It requires the configured models and a suitable CUDA GPU.
Results go to `results/collection/synthetic_sample_100q_pilot/train/` and
activations under `data/activations/synthetic_sample_100q_pilot/layer_<N>/`.
Add `--resume` to continue an interrupted run from its saved chunks.

Collection configs can set `dataset.path` to a local MuSiQue-format JSONL file,
including synthetic `accepted.jsonl`, instead of loading from Hugging Face.
Paths are relative to the working directory; omit `dataset.revision` for local
files. `dataset.source_split` labels the output directory and does not filter the
file. Existing sampling strategies and `dataset.num_questions` apply to that
file. The input file hash is saved in provenance and checked on resume. To use
another sample, change `dataset.path` and choose a new `output.run_name`.

Collection run definitions are versioned under `configs/collection/`. Once a
config has produced a recorded experiment run, keep that version unchanged and
create a new version directory for protocol changes.

Use the corresponding train or validation config for larger runs.

Activation tensors are saved under:

    data/activations/<run_name>/layer_<N>/

Each source split produces:

- <split>\_attempt1.pt
- <split>\_attempt2.pt
- <split>.pt

<split>.pt is the SAE-facing tensor with Attempt 1 rows followed by Attempt 2 rows.

Run metadata and reproducibility information are saved under:

    results/collection/<run_name>/<split>/

This directory contains interactions.jsonl, resolved_config.yaml, summary.json,
collected_question_ids.json and exclusions.jsonl. The record fields are defined
in `cookbooks/collection/episode_record.md`, the provenance files and the identity used by
`--resume` in `cookbooks/collection/collection_provenance.md`, and the capture position, tensor
layout and row indices in `cookbooks/collection/activations.md` (including a historical MuSiQue
partition diagram).

The metadata of the finished production run (`natural_4b_full`: summaries,
resolved configs, question lists, provenance) and of its partition export are
committed under `results/collection/`. The labels, the partition manifest and
the frozen question-to-partition mapping are committed under
`results/behavior/behavior_v011_split80_10_10_20261006/`. Tensors and
`interactions.jsonl` stay in S3. Re-runs of these historical experiments take
their partitions from that mapping; see `cookbooks/labels/behavior_v01.md`, "Re-running the
collection". New mixed-dataset runs use their own prepared manifest and exported
membership instead.

Here `<split>` is the MuSiQue source split (train or validation), which mixes
the research partitions. For SAE and probe work use an exported partition run
instead: `scripts/archive/single_machine_collection/export_partition_activations.py` writes the same layout under
a new run name with the partition (train, validation, test) in place of the
source split, following the frozen `split_manifest.csv`. See
`cookbooks/labels/behavior_v01.md`, section "Keeping evaluation honest".

## Collection condition policy

Written for the v2 `natural_4b_full` production run. `mixed_final_001` follows the same policy
(Natural only, blind-first, Solver activations only) with three differences, all set in
`configs/collection/distributed_final.yaml`:

- **Questions:** mixed MuSiQue + synthetic, with a 10% intervention partition reserved first.
- **Layers:** nine Solver layers (4, 8, 17, 25, 29–33).
- **Critic token budget:** 1024.

Its behavior labels are the Claude labels (stage 2 of the pipeline), not the lexical labels
described below.

- Natural, Controlled Correct (B) and Controlled Incorrect (C) remain
  supported; `collection.active_conditions` selects a nonempty subset.
  Omitting it keeps V1's three-condition behaviour; `null` or `[]` is an error.
- The current production policy activates **Natural only**. B and C stay
  implemented and tested but inactive; disabled conditions generate no
  targets and no episodes.
- Natural is blind-first: the Critic answers independently, then reviews
  Solver A1. It never sees gold answers or aliases.
- Solver A1 is generated once per question and reused by every enabled condition.
- Only Solver activations are captured; Critic and Validator activations never are.
- Solver, Critic and Validator are independent role specs; identical specs
  share weights, different specs load separately.
- Raw Critic outputs are kept (`critic_blind_raw_output`, `critic_raw_output`)
  so both steps can be re-parsed later.
- Behaviour labels (`behavior_schema_version: lexical_v2`) are conservative
  lexical heuristics, not semantic judgments: `critic_position_relation`,
  `solver_behavior`, `direct_critic_adoption`, `solver_copied_nonanswer`.
  Refusals, malformed output and unresolved cases are never counted as
  rejection, and a lexical difference is not proven disagreement. Legacy
  `solver_accepted_feedback` counts are reported separately under
  `legacy_acceptance_counts`.
- Questions whose blind answer or controlled target cannot be constructed are
  skipped and written to `exclusions.jsonl`; `collected_question_ids.json`
  lists the questions that produced activation rows.
- Generation telemetry records token counts and whether the budget was reached.
- Collection saves progress in chunks (`output.chunk_size`, default 250);
  restart an interrupted collection with `--resume`.
- Train and validation progress are tracked separately under one `run_name`.
- Resume refuses if the config, questions, revisions, code or environment
  changed.
- Production collects full MuSiQue train (19,938) and validation (2,417)
  under one `run_name`, from Solver layers 8, 17, 25 and 33.
- Token budgets: Solver 64, Critic 256, Validator 4.
