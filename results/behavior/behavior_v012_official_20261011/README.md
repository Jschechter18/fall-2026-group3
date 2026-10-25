# Behavior v0.1.2 — Official Release

Status: Published and SHA256-verified, October 11, 2026.
Tracking: https://github.com/Jschechter18/fall-2026-group3/issues/108

## Official labels
s3://dats-capstone/shared/group3/results/behavior/behavior_v012_official_20261011/

labels.csv SHA256:
8d9d1a49a1ff562ec44ea531aea544b9844244cf785514d91409b0365fe2bf6b

22,332 episodes. Frozen partitions unchanged.
Policy: v0.1.2-qc

## Nine-layer activations
Train, validation and test, in one run since PR #147:
s3://dats-capstone/shared/group3/data/activations/natural_4b_full9_partitioned/

The earlier separate test export, natural_4b_full9_test_partitioned/, is kept as
history; its 27 tensor files are byte-identical to the test files above.

## Required downstream actions
- Josh: confirm SAE checkpoint and training provenance. Fit SAEs on train, use validation for tuning and checkpoint selection, and use test to compare completed SAE versions under Issue #97. Keep intervention reserved for causal experiments
- Raye: load v0.1.2 labels by question_id, rerun layer selection, report validation results.
- Ignore older v0.1.1 labels embedded in interactions.jsonl.
- Record label version and SHA256 in downstream results.
- Reserve intervention data for causal experiments.

## Research limitations
- Validation has 58 eligible negatives (13 retained A1, 45 third answer).
- Negative-class reliability remains uncertain.
- Earlier development exposed some test/intervention questions.
- Report final sensitivity analyses excluding previously exposed question IDs using manifest flags.

The data files and SHA256SUMS are committed in this folder and mirrored in S3.
No private QC files are included here.
