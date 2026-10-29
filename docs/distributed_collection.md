# Distributed episode collection

For copy-and-paste commands on two EC2 workers, follow the
[MuSiQue-only smoke test](distributed_collection_smoke_test.md). It includes
school SSO setup, separate worker assignments, S3 transfer, finalization, and
checks against the original activation chunks.

The unit of work is a **question**, not an activation row. Preparation freezes
the question payload, partition, seed, and ordered chunk membership before any
model outcome is observed. A worker invocation loads its models, processes one
explicitly assigned package, writes the existing local chunk format, and
uploads that directory as an isolated S3 attempt. S3 is transport only;
`ActivationStore` remains local.

Implementation lives in `src/mas_sae/experiments/distributed_collection/`:
`core.py` holds preparation, S3 transport, and finalization; `cli.py` holds
argument parsing and command dispatch. `__init__.py` re-exports the public
functions to preserve existing imports. `scripts/distributed_collection.py`
remains the command-line entry point.

Synthetic data is not available yet. The synthetic example paths and hop field
are a proposed producer contract, and unit tests use handcrafted records only.
For current MuSiQue-only preparation, omit the synthetic input entry entirely.
Before adding a real synthetic release, verify its schema, deduplication, hop
metadata, and generation provenance with the producer.

## Policies and compatibility

- Global IDs are `<dataset_source>:<original_id>`. Original IDs and complete
  payloads remain in the master manifest. Hop metadata is persisted explicitly,
  so downstream code never has to parse a namespaced ID.
- The default `musique_validation_policy: validation` preserves the historical
  rule: every MuSiQue validation-source question is tuning data. This means its
  source composition is not controlled by the general four-way proportions.
  `stratify` deliberately changes that rule; `error` forces an explicit choice.
- Existing assignments are copied and checked; only new questions are assigned.
  This preserves prior development exposure metadata and never reshuffles an
  old question. Seeds are per-question hashes and do not depend on chunking or
  scheduling. They improve reproducibility but do **not** promise bitwise
  equality across GPU types, kernels, or library builds.
- `intervention` is present in the frozen manifest but absent from ordinary
  work plans and final SAE/probe outputs. It remains reserved for causal work.
- Collection success and disagreement eligibility are separate. Exclusions
  describe collection failures; existing label policies remain the authority
  for consumer-specific eligibility.
- `freeze` assigns `candidate_order` (`a1_first` or `critic_first`) before any
  chunking, using the partition configuration's seed and explicit hop metadata.
  Orders are balanced within each partition/hop group and are retained when
  extending a frozen manifest. Only newly added or previously missing orders
  are assigned; chunk sizes and worker assignments do not change them.
  Work packages carry the orders to the collector, and finalization checks
  candidates records against them. Feedback-only collection does not need them.

For an older frozen manifest without candidate orders, rerun `freeze` with
`--existing OLD_PARTITIONS.json` and a new output path, then rerun `plan` with
new output/package paths. This preserves old partition assignments and seeds
while adding the missing orders. Candidates workers reject missing/invalid
orders before loading models rather than deciding presentation order locally.

## Example commands

Use the same agreed collection YAML and prepared run ID across the team, assign
non-overlapping chunk IDs, and upload each chunk under that run. Code and package
versions, hardware, device placement, output paths, timestamps, and chunk facts
are provenance rather than compatibility gates. The required compatibility file
is the canonical output of `experiment_compatibility(resolved_config, sites)`.
It fixes role model IDs and resolved revisions, generation and prompt identities,
protocol/capabilities, active conditions, A2 presentation, dtype/quantization,
and activation sites. Missing model IDs or resolved revisions are rejected.
Workers derive the same contract from what they loaded, and finalization checks
every worker and its supplied resolved config against it. Schema-v2 contracts
require content hashes for every recorded solver, critic, and validator prompt,
including candidate labels. Editing a template without renaming its constant
therefore changes compatibility. Older contracts/plans/packages must be
regenerated under a new run ID; old chunk metadata cannot supply the missing
prompt fingerprints retrospectively.

Generate the contract on a model-capable worker before planning:

```bash
PYTHONPATH=src python scripts/collect_activations.py \
  --config REPLACE_COLLECTION.yaml \
  --prepare-compatibility artifacts/experiment-compatibility.json
```

This reuses the collector's model loading, agent configuration, and activation
site resolution, then exits without sampling or collecting episodes. It may
download model weights and requires the same model access and memory as collection.
It refuses an existing contract output and cannot be combined with collection
or resume arguments.

Checks retained for data correctness include matching master/partition IDs and
payload references, chunk membership and seeds, complete coverage, tensor/index
alignment, and S3 download checksums. Download receipts are retained so changed
local files are detected again during finalization. Actual worker provenance is
saved in chunk metadata and retained in the final `worker_provenance.json`;
it does not block execution based on Git commit, GPU, or package versions.

All `REPLACE_*` values below are placeholders, not project decisions. First
split `configs/distributed_collection.example.yaml` into the small inputs and
partition JSON/YAML files expected by the commands (or generate them from your
experiment configuration).

```bash
PYTHONPATH=src python scripts/distributed_collection.py master \
  --inputs REPLACE_INPUT_SPECS.yaml --output artifacts/master-v1.json

PYTHONPATH=src python scripts/distributed_collection.py freeze \
  --master artifacts/master-v1.json --config REPLACE_PARTITION_CONFIG.yaml \
  --output artifacts/partitions-v1.json

# On an extension, add --existing artifacts/partitions-v1.json and choose a new output.
PYTHONPATH=src python scripts/distributed_collection.py plan \
  --master artifacts/master-v1.json --partitions artifacts/partitions-v1.json \
  --run-id REPLACE_RUN_ID --chunk-size REPLACE_CHUNK_SIZE \
  --compatibility artifacts/experiment-compatibility.json \
  --output artifacts/work-plan.json --packages artifacts/work-packages
```

Upload immutable packages, then explicitly assign chunk IDs to the two workers
(for example, worker 1 gets 0–9 and worker 2 gets 10–19):

```bash
PYTHONPATH=src python scripts/distributed_collection.py upload-work \
  --bucket REPLACE_BUCKET --prefix REPLACE_PREFIX --run-id REPLACE_RUN_ID \
  --chunk-id REPLACE_CHUNK_ID --package artifacts/work-packages/REPLACE_CHUNK_ID.json

# Run this on the assigned worker.
PYTHONPATH=src python scripts/distributed_collection.py download-work \
  --bucket REPLACE_BUCKET --prefix REPLACE_PREFIX --run-id REPLACE_RUN_ID \
  --chunk-id REPLACE_CHUNK_ID --output work/REPLACE_CHUNK_ID.json
```

For each assigned package:

```bash
PYTHONPATH=src python scripts/collect_activations.py \
  --config REPLACE_COLLECTION_CONFIG.yaml \
  --work-package artifacts/work-packages/REPLACE_CHUNK_ID.json \
  --chunk-output local-results/REPLACE_CHUNK_ID

PYTHONPATH=src python scripts/distributed_collection.py upload \
  --bucket REPLACE_BUCKET --prefix REPLACE_PREFIX --run-id REPLACE_RUN_ID \
  --chunk-id REPLACE_CHUNK_ID --chunk-dir local-results/REPLACE_CHUNK_ID \
  --marker local-results/REPLACE_CHUNK_ID.marker.json

PYTHONPATH=src python scripts/distributed_collection.py accept \
  --bucket REPLACE_BUCKET --prefix REPLACE_PREFIX --run-id REPLACE_RUN_ID \
  --chunk-id REPLACE_CHUNK_ID --marker local-results/REPLACE_CHUNK_ID.marker.json
```

Schema-v2 work packages contain original, unmodified question payloads and the
complete frozen plan. A worker recomputes the plan, membership, and every
original payload SHA-256 before loading models. Schema-v1 packages cannot prove
that chain and are rejected with instructions to regenerate them. Work-package
objects use S3 conditional creation; identical retries are accepted, conflicting
bytes are rejected, and non-precondition S3 errors propagate.

An interrupted local attempt is retried into a new local directory and uploaded
with a new attempt ID. Acceptance is idempotent for byte-identical results and
rejects a different package for an already accepted logical chunk. It also
checks the CLI run/chunk identity and the remote attempt's `complete.json`. AWS access
uses the normal boto3 credential chain; no credential belongs in configuration.

Before each S3 transfer/accept command, the CLI makes an unsigned HTTPS HEAD
request to the configured S3 endpoint and bucket path, with a five-second socket
timeout. Any HTTP response, including 403, counts as reachable. DNS failures,
connection failures, and timeouts stop the command before its S3 operation.
This checks connectivity only, not permissions or future availability; it does
not run during local preparation, finalization, or the separate collector.

Download every accepted chunk into `<chunks>/<chunk-id>` and finalize:

```bash
PYTHONPATH=src python scripts/distributed_collection.py download \
  --bucket REPLACE_BUCKET --prefix REPLACE_PREFIX --run-id REPLACE_RUN_ID \
  --chunk-id REPLACE_CHUNK_ID --output accepted/REPLACE_CHUNK_ID

PYTHONPATH=src python scripts/distributed_collection.py finalize \
  --plan artifacts/work-plan.json --chunks accepted \
  --sites REPLACE_FULL_MODULE_SITE_1,REPLACE_FULL_MODULE_SITE_2 \
  --resolved-config REPLACE_RESOLVED_CONFIG.yaml --run-name REPLACE_FINAL_RUN
```

Finalization rejects missing/extra chunks, plan or membership mismatches,
corruption, site mismatches, gaps/overlaps, and bad record/tensor alignment. It
uses saved successful counts for exclusions, rebuilds all activation and SAE
indices, and writes each partition as `attempt1`, `attempt2`, then
`cat(attempt1, attempt2)`. The resulting layout is the one consumed by existing
SAE and probe commands; point their activation/result paths at the final run and
continue to select train/validation/test through the frozen manifest.

Each deterministic same-plan staging directory records `publication.json`
ownership before artifacts are built. Both trees receive `publication_ready.json`
only after every partition, provenance file, and summary has been written.
`publication_complete.json` markers are written after both trees are published.
Result, SAE, and probe readers check completion against the owning identity;
ordinary legacy activation stores remain readable.

Ordinary build exceptions clean up their private staging so the same command can
be retried. After a killed build or interrupted publication, rerun the identical
finalize command with `--recover-publication`. Recovery checks all existing trees'
run ID, run name, and plan digest before making changes. It discards and rebuilds
only matching unready private staging, or finishes publishing ready trees. It
never deletes or replaces unrelated outputs and never publishes an unready tree.
Older interrupted outputs without readiness metadata must be moved aside and
re-finalized from compatible accepted chunks. Existing completed publications
with matching ownership/completion identities remain readable; pre-marker
distributed results require re-finalization under a new output name.

## Layout and capacity

S3 attempts live under
`<prefix>/results/<run>/<chunk>/attempts/<attempt>/`; `complete.json` appears
only after every required object passes a size check. `accepted.json` is the
single logical-chunk decision. Locally, final files retain
`layer_NN/<partition>[_attempt1|_attempt2].pt` and
`results/<run>/<partition>/interactions.jsonl`.

Each completed chunk stores A1 and A2 once. During finalization, sites are
merged and written one at a time; allow roughly one partition-layer's A1+A2
tensor bytes, one loaded chunk container, plus PyTorch concatenation overhead,
and enough disk for chunks plus staged and final files
(approximately three copies at the rename boundary). If that estimate is too
large for the selected instance, use smaller partitions or provision a larger
local volume; do not merge by concatenating already-combined chunk tensors.

## Historical partitions and downstream loading

Historical MuSiQue assignments can be imported directly when freezing the master
question pool. Keep the original CSV unchanged:

```bash
PYTHONPATH=src python scripts/distributed_collection.py freeze \
  --master artifacts/master-v1.json --config REPLACE_PARTITION_CONFIG.yaml \
  --existing REPLACE_HISTORICAL_PARTITIONS.csv \
  --exposure-manifest REPLACE_HISTORICAL_SPLIT_MANIFEST.csv \
  --output artifacts/partitions-v1.json
```

`--exposure-manifest` is optional when the input CSV already includes
`in_qc100`/`in_development_data`. With partitions alone, unavailable development
history stays unknown and is reported; it is not evidence that a question is
untouched. The existing partition sampler does not select unknown-history rows
as unexposed data. Import matches `(musique, original_id)` to the master ID,
preserves membership, and rejects missing IDs or conflicting origin/hop metadata.
Synthetic questions with the same original ID do not inherit MuSiQue assignments.
Only questions absent from the imported mapping receive new partitions. For a
later master expansion, use the previous frozen JSON as `--existing`.

New work plans carry question metadata for finalization. Final results include:

- `partitions.csv`: permanent membership for **all** questions, including the
  reserved intervention pool and collection exclusions.
- `split_manifest.csv`: collected questions only, with actual per-split
  `full_index` activation addresses and available episode IDs.
- `run_layout.json`: the directories readers must open, including `test` when
  assigned. A missing expected file raises rather than silently dropping data.
- `collection_status.json`: collected, excluded (with reasons), reserved, and
  unknown-exposure IDs. Expected intervention/exclusions are not missing runs.
- `counts.json`: source-file hashes and collected counts by partition/source;
  this contains no behavioral labels.

For run records/manifests, `source_split` identifies the tensor and interaction
folder. `original_source_split` retains the original dataset split; `partition`
controls permitted use. `dataset_source`, `original_id`, explicit hop metadata,
and exposure flags are retained. Thus a question can originate in MuSiQue train,
be assigned research test, and live at `test/interactions.jsonl` and
`layer_NN/test_attempt2.pt` without losing its origin.

Historical readers keep `layout="legacy"` by default, but reject a directory
whose `run_layout.json` declares `partitioned`; unavailable requested partitions
also raise instead of silently disappearing. Use
`read_run_interactions(..., layout="partitioned")`,
`read_partitions(..., layout="partitioned")`, and
`load_labeled_rows(..., layout="partitioned")` for the new results. The existing
activation loader uses those manifests directly. SAE dataloaders already read
`train.pt`, `validation.pt`, and `test.pt`; intervention tensors are not produced.
Probe configurations point `manifest_csv` at the new `split_manifest.csv`, use
`index_column: full_index`, and name their input folders in `source_splits`.
Fitting/tuning uses `[train, validation]`; final test evaluation must explicitly
select test using the existing sealed-partition controls.

Labels remain a separate post-processing product. Existing behavioral labeling
can be exercised without changing its rules:

```bash
PYTHONPATH=src python scripts/prepare_behavior_v01.py \
  --full-results results/collection/REPLACE_FINAL_RUN --layout partitioned \
  --label-policy v0.2.0-candidate --output REPLACE_NEW_LABEL_PACKAGE
```

This reads frozen membership from the run's `partitions.csv` by default, opens
all recorded split folders, and accounts for reserved/excluded questions. The
historical source-run/layer-scan merge stays on the legacy path. The activation
export script also accepts `--layout partitioned` with the new frozen CSV.
Claude labeling is outside this change: a future label adapter can use stable
question/episode IDs without changing membership or tensors. Labels from an
older collection must not be blindly reused for newly generated answers.

Older work plans without embedded question metadata retain their prior merge
behavior but do not produce these integrated downstream tables. Regenerate the
plan/packages from the same frozen assignments to use this workflow.

The integrated row manifest assumes one condition and one A2 presentation per
question, matching the current collection configuration. Multiple episodes per
question require a separate episode-level manifest extension. Feedback audit
and partition-selection commands also accept `--layout partitioned`. Selection
returns namespaced master IDs. Partitioned `--exclude` JSON lists are validated
against the complete frozen pool, including other partitions. Exact namespaced
IDs are accepted; an unambiguous legacy MuSiQue ID is normalized to
`musique:<id>`, while unknown/ambiguous IDs report their source file. These IDs are for master-dataset workflows, not direct input to the
legacy MuSiQue Hub `sampling.strategy: listed` loader. Reserved intervention IDs
come from `intervention_question_ids.json`, not collected episode sampling.
