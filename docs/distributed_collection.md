# Distributed episode collection

The unit of work is a **question**, not an activation row. Preparation freezes
the question payload, partition, seed, and ordered chunk membership before any
model outcome is observed. A worker loads its models once, processes one or
more explicitly assigned packages, writes the existing local chunk format, and
uploads that directory as an isolated S3 attempt. S3 is transport only;
`ActivationStore` remains local.

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

## Example commands

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
  --compatibility REPLACE_COMPATIBILITY.json \
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

An interrupted local attempt is retried into a new local directory and uploaded
with a new attempt ID. Acceptance is idempotent for byte-identical results and
rejects a different package for an already accepted logical chunk. AWS access
uses the normal boto3 credential chain; no credential belongs in configuration.

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
