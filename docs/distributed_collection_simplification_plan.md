# Distributed collection: implementation record

The implementation is complete and locally validated. Live EC2/S3 smoke
validation remains to be run before scaling.

Use the [workflow guide](distributed_collection.md) for current commands,
configuration, output paths, and downstream integration. Use the
[two-EC2 smoke procedure](distributed_collection_smoke_test.md) for operational
validation. This record retains the accepted design decisions without repeating
operator instructions.

## Accepted design

- Prepare one master question manifest from MuSiQue and optional synthetic data.
- Reserve intervention membership before collection. Collect its baseline
  episodes and activations in the initial distributed run, then export them
  under a separate intervention run name.
- Freeze question seeds, candidate orders, and work chunks independently of
  worker count. Respect explicitly supplied question families.
- Assign train/validation/test to the remaining master questions after merging,
  before training, tuning, or using held-out outcomes for development decisions.
- Preserve existing single-machine configurations and historical artifact loading.

## Bookkeeping contract

The existing collector, checkpoint merger, activation writer, tensor exporter,
label classifiers, and downstream purpose checks are reused.

Final `source_split` names the actual tensor/interaction folder. Final
`full_index`, attempt indices, and SAE indices address that export's own tensors.
`collection_activation_index` retains the original merged address for traceability;
new exports omit the legacy `source_activation_index` override.

Every master question retains membership and a collection status. Excluded
questions receive reasons and no tensor addresses. Intervention ID lists retain
all reserved questions, including exclusions. Ordinary SAE/probe inputs never
contain intervention tensors.

Causal replay uses saved source contexts, the recorded candidate order, and the
configured label policy. It calls `Solver.choose` for candidates and retains
`Solver.revise` for legacy feedback. Candidate baseline cohorts use strict
behavioral eligibility; post-intervention third answers remain reported outcomes.
The baseline replay check must pass before feature interventions run.

## Validation record

- Full unit suite: **1,119 passed, 7 skipped**. Coverage includes serial versus
  distributed collection, reversed arrival order, exact row identity across
  layers and attempts, exclusions, resume, intervention isolation, and loading
  through the real SAE/probe/behavior/causal input paths.
- A tiny CPU model exercised the causal first-token hook and candidate prompt.
  Existing replay-preflight tests continue to block unreproduced baselines.
- The actual prepare, merge, and split CLI sequence passed with fixture
  collection in a temporary directory. CLI help and whitespace checks passed.

No live EC2/S3 transfer or full collection-model inference was performed.
Mixed-source fixtures validate the adapter contract; a real synthetic release
still needs producer schema and quality verification.
