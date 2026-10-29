# Two-EC2 distributed collection smoke test

Run these blocks from the repository root, in Bash, using the implementation
containing candidate-order wiring and partition/downstream integration.
Preparation/finalization can run on EC2 A or another machine. Both GPU workers
need the existing 4B collection environment, model access, and enough GPU memory
for the same Gemma/Qwen pair used by current runs. No worker-to-worker network
connection is required. Use tmux for the GPU work.

This runs real models and uses S3. The instructions have been syntax checked;
they have not been executed against live EC2/S3 by the author.

The test draws **eight 2-hop questions from the historical training pool**.
It assigns disposable smoke-only train/validation/test/intervention partitions:
two questions each. It collects six questions, in six one-question chunks;
two remain reserved. Both workers process one chunk from each collected split.
These are pipeline checks, not research splits or a meaningful probe evaluation.
Do not replace the committed research partitions with the smoke manifest.

## 1. Common setup — every participating machine

Use the **same** `SMOKE_RUN` on every machine. For another test choose a new name,
for example `group3_smoke_002`, rather than overwriting existing results.

```bash
conda activate capstone
export PYTHONPATH=src
export AWS_PROFILE=school
export SMOKE_RUN=group3_smoke_001
export SMOKE_BUCKET=dats-capstone
export SMOKE_PREFIX=shared/group3/data/smoke-tests
export SMOKE_DIR="$PWD/data/distributed-smoke/$SMOKE_RUN"
mkdir -p "$SMOKE_DIR"

# If the school profile is not configured on this machine, do this once:
# aws configure sso --profile school --use-device-code
aws sso login --profile school --use-device-code
aws sts get-caller-identity

# If model access is not already configured on a GPU worker:
# hf auth login

set -euo pipefail

dc() {
  python scripts/distributed_collection.py "$@"
}
```

Run the commands below in that shell so the variables/function are available.
S3 operations use the selected profile. Local collection itself does not need
an active SSO session; authenticate again before uploads if the session expires.

## 2. Prepare — one GPU worker only

This creates a small input file and collection config. It uses the pinned
MuSiQue revision and existing candidates model settings, captures only layer 8,
and keeps the original generation budgets. Dataset/model downloads may be
needed. Contract preparation loads the configured models to record their actual
revisions, precision, and activation sites, then releases them when the command
exits. It performs no inference or episode collection. Use a worker with enough
GPU memory for the configured models; the other preparation commands need no GPU.

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import yaml
from mas_sae.data.musique import load_musique_examples_by_id
from mas_sae.data.production import read_partitions

root = Path(os.environ['SMOKE_DIR']) / 'control'
root.mkdir()  # Refuse to overwrite an earlier preparation.
config = yaml.safe_load(Path('configs/collection/v3/natural_4b_full9_candidates_train.yaml').read_text())
historical = read_partitions(Path('results/behavior/behavior_v011_split80_10_10_20261006/partitions.csv'))
ids = sorted(q for q, r in historical.items() if r['partition'] == 'train' and r['hop_group'] == '2hop')[:8]
assert len(ids) == 8
questions = load_musique_examples_by_id('train', ids, revision=config['dataset']['revision'])
assert len(questions) == 8
(root / 'questions.json').write_text(json.dumps(questions, indent=2) + '\n')
config['dataset'] = {'revision': config['dataset']['revision'], 'source_split': 'train', 'num_questions': 8}
config['collection']['layers'] = [8]
config['collection']['active_conditions'] = ['natural']
config['collection']['a2_presentations'] = ['candidates']
config['output'] = {'run_name': os.environ['SMOKE_RUN'], 'chunk_size': 1}
(root / 'collection.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
inputs = {'inputs': [{'source': 'musique', 'version': config['dataset']['revision'],
                     'path': str(root / 'questions.json'), 'source_split': 'train'}]}
(root / 'inputs.yaml').write_text(yaml.safe_dump(inputs))
partitions = {'seed': 42, 'proportions': {'train': .25, 'validation': .25, 'test': .25, 'intervention': .25}}
(root / 'partition-config.yaml').write_text(yaml.safe_dump(partitions, sort_keys=False))
print('Prepared eight historical training questions for disposable smoke partitions.')
PY

dc master --inputs "$SMOKE_DIR/control/inputs.yaml" \
  --output "$SMOKE_DIR/control/master.json"

dc freeze --master "$SMOKE_DIR/control/master.json" \
  --config "$SMOKE_DIR/control/partition-config.yaml" \
  --output "$SMOKE_DIR/control/partitions.json"

# Record the actual model identities and prompt hashes before freezing the plan.
python scripts/collect_activations.py \
  --config "$SMOKE_DIR/control/collection.yaml" \
  --prepare-compatibility "$SMOKE_DIR/control/experiment-compatibility.json"

dc plan --master "$SMOKE_DIR/control/master.json" \
  --partitions "$SMOKE_DIR/control/partitions.json" \
  --run-id "$SMOKE_RUN" --chunk-size 1 \
  --compatibility "$SMOKE_DIR/control/experiment-compatibility.json" \
  --output "$SMOKE_DIR/control/plan.json" --packages "$SMOKE_DIR/packages"

python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['SMOKE_DIR']) / 'control'
plan = json.loads((root / 'plan.json').read_text())
chunks = [c['chunk_id'] for c in plan['chunks']]
assert len(chunks) == 6
for name, assigned in [('worker-a', chunks[::2]), ('worker-b', chunks[1::2])]:
    assert {c.split('-')[0] for c in assigned} == {'train', 'validation', 'test'}
    (root / f'{name}.txt').write_text('\n'.join(assigned) + '\n')
(root / 'all-chunks.txt').write_text('\n'.join(chunks) + '\n')
print('Worker A:', chunks[::2])
print('Worker B:', chunks[1::2])
PY

aws s3 cp "$SMOKE_DIR/control/" \
  "s3://$SMOKE_BUCKET/$SMOKE_PREFIX/control/$SMOKE_RUN/" --recursive

while IFS= read -r chunk; do
  dc upload-work --bucket "$SMOKE_BUCKET" --prefix "$SMOKE_PREFIX" \
    --run-id "$SMOKE_RUN" --chunk-id "$chunk" \
    --package "$SMOKE_DIR/packages/$chunk.json"
done < "$SMOKE_DIR/control/all-chunks.txt"
```

Expected: six work packages in S3 and two assignment files, three chunks each.
The separate `control/` upload shares the configuration/plan/assignments; work
packages use the existing transport's `inputs/<run>/<chunk>.json` layout.

## 3. Collect — both EC2 instances, different assignments

On EC2 A select `worker-a`. On EC2 B select `worker-b`. Everything else is the
same. Do not run the same assignment on both instances.

```bash
export SMOKE_WORKER=worker-a   # EC2 B: use worker-b instead.

aws s3 cp "s3://$SMOKE_BUCKET/$SMOKE_PREFIX/control/$SMOKE_RUN/" \
  "$SMOKE_DIR/control/" --recursive
mkdir -p "$SMOKE_DIR/work" "$SMOKE_DIR/local-results" "$SMOKE_DIR/logs"

while IFS= read -r chunk; do
  package="$SMOKE_DIR/work/$chunk.json"
  result="$SMOKE_DIR/local-results/$chunk"
  marker="$SMOKE_DIR/local-results/$chunk.upload.json"

  if [[ ! -f "$package" ]]; then
    dc download-work --bucket "$SMOKE_BUCKET" --prefix "$SMOKE_PREFIX" \
      --run-id "$SMOKE_RUN" --chunk-id "$chunk" --output "$package"
  fi

  if [[ ! -d "$result" ]]; then
    python scripts/collect_activations.py \
      --config "$SMOKE_DIR/control/collection.yaml" \
      --work-package "$package" --chunk-output "$result" \
      2>&1 | tee -a "$SMOKE_DIR/logs/$chunk.log"
  fi

  if [[ ! -f "$marker" ]]; then
    dc upload --bucket "$SMOKE_BUCKET" --prefix "$SMOKE_PREFIX" \
      --run-id "$SMOKE_RUN" --chunk-id "$chunk" \
      --chunk-dir "$result" --marker "$marker"
  fi

  dc accept --bucket "$SMOKE_BUCKET" --prefix "$SMOKE_PREFIX" \
    --run-id "$SMOKE_RUN" --chunk-id "$chunk" --marker "$marker"
done < "$SMOKE_DIR/control/$SMOKE_WORKER.txt"
```

Each invocation currently reloads models. These loops are sequential, so one
worker does not try to run several GPU collections concurrently. A collection
exclusion is a recorded outcome, not a failed upload or missing chunk.

If authentication/upload fails, local results remain. Reauthenticate and rerun
this worker block: completed local chunks and existing upload markers are
reused. Do not pass `--resume`; within-chunk resume is not implemented. An
interrupted unfinished one-question chunk starts over on the next invocation.
If acceptance reports conflicting results, stop and inspect the attempts rather
than overwriting the accepted result.

## 4. Download and merge — one machine, after both workers finish

Run the common setup first if this is a new shell. This machine needs Python
and the project environment, but no GPU for merging.

```bash
aws s3 cp "s3://$SMOKE_BUCKET/$SMOKE_PREFIX/control/$SMOKE_RUN/" \
  "$SMOKE_DIR/control/" --recursive
mkdir -p "$SMOKE_DIR/accepted"

while IFS= read -r chunk; do
  destination="$SMOKE_DIR/accepted/$chunk"
  if [[ ! -f "$destination/download_receipt.json" ]]; then
    if [[ -e "$destination" ]]; then
      echo "Partial download exists at $destination; move it aside and rerun this block." >&2
      exit 1
    fi
    dc download --bucket "$SMOKE_BUCKET" --prefix "$SMOKE_PREFIX" \
      --run-id "$SMOKE_RUN" --chunk-id "$chunk" --output "$destination"
  fi
done < "$SMOKE_DIR/control/all-chunks.txt"

python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['SMOKE_DIR'])
metadata = json.loads(next((root / 'accepted').glob('*/metadata.json')).read_text())
(root / 'control' / 'resolved-config.json').write_text(json.dumps(metadata['resolved_config'], indent=2) + '\n')
(root / 'control' / 'sites.txt').write_text(','.join(metadata['candidate_sites']) + '\n')
PY

dc finalize --plan "$SMOKE_DIR/control/plan.json" --chunks "$SMOKE_DIR/accepted" \
  --sites "$(cat "$SMOKE_DIR/control/sites.txt")" \
  --resolved-config "$SMOKE_DIR/control/resolved-config.json" \
  --activation-root "$SMOKE_DIR/final/activations" \
  --result-root "$SMOKE_DIR/final/results" --run-name "$SMOKE_RUN"
```

Expected: final `train`, `validation`, and `test` records and manifests under
`$SMOKE_DIR/final/results/$SMOKE_RUN`, and layer-8 tensors under
`$SMOKE_DIR/final/activations/$SMOKE_RUN/layer_08` when those splits have
successful questions. No intervention tensors should exist. Finalization is
not overwriteable. Ordinary failures while building staging are cleaned up, so
rerun the same command after resolving their cause. For a killed build or a
failure during publication, rerun the identical command with
`--recover-publication`: it rebuilds matching unfinished staging or completes
publication of ready outputs. It refuses unrelated output directories. Result,
SAE, and probe readers reject distributed outputs without valid completion markers.

## 5. Verify merge contents and the actual activation loaders

This compares every merged A1/A2 vector to its original chunk vector, checks
candidate orders/seeds/partitions, and loads tensors through the production and
SAE readers. It does not require every question to produce disagreement.

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import torch
from mas_sae.data.activation_store import ActivationStore
from mas_sae.data.production import read_manifest, read_run_interactions, load_production_activations
from mas_sae.experiments.records import read_jsonl
from mas_sae.sae.dataloader import create_sae_dataloader

root = Path(os.environ['SMOKE_DIR'])
run = os.environ['SMOKE_RUN']
results = root / 'final' / 'results' / run
activations = root / 'final' / 'activations' / run
plan = json.loads((root / 'control' / 'plan.json').read_text())
manifest = read_manifest(results / 'split_manifest.csv')
status = json.loads((results / 'collection_status.json').read_text())
rows = read_run_interactions(results, layout='partitioned')
by_episode = {r['episode_id']: r for r in rows}
assert len(by_episode) == len(rows)
collected = set(status['collected'])
excluded = {r['question_id'] for r in status['excluded']}
reserved = set(status['reserved'])
assert not collected & excluded and not collected & reserved and not excluded & reserved
assert collected | excluded | reserved == {r['question_id'] for r in plan['questions']}
assert len(reserved) == 2 and not (results / 'intervention').exists()
site = (root / 'control' / 'sites.txt').read_text().strip()
assert site == 'model.language_model.layers.8'
store = ActivationStore(activations / 'layer_08')
tensors = {}
for split, purpose in [('train', 'fit'), ('validation', 'tune'), ('test', 'evaluate')]:
    group = [r for r in rows if r['source_split'] == split]
    assert group, f'No successful collection in {split}; use a larger disposable sample to exercise its loaders.'
    a1, a2, sae = [store.load_activations(name) for name in (split+'_attempt1', split+'_attempt2', split)]
    assert a1.shape == a2.shape and a1.shape[0] == len(group)
    assert torch.equal(sae, torch.cat([a1, a2]))
    for attempt, expected in [(1, a1), (2, a2)]:
        loaded = load_production_activations(group, manifest, activations, purpose=purpose, layer=8, attempt=attempt)
        assert torch.equal(loaded, expected)
    batch = next(iter(create_sae_dataloader(2, split, 0, location=activations / 'layer_08')))
    assert torch.equal(batch, sae[:len(batch)])
    tensors[split] = (a1, a2, sae)
for chunk in plan['chunks']:
    source = root / 'accepted' / chunk['chunk_id']
    originals = read_jsonl(source / 'records.jsonl')
    state = torch.load(source / 'activations.pt', map_location='cpu', weights_only=True)
    assigned = {r['question_id']: r for r in chunk['questions']}
    successful = list(dict.fromkeys(r['question_id'] for r in originals))
    for local_a2, original in enumerate(originals):
        row = by_episode[original['episode_id']]
        qid = row['question_id']
        assert row['seed'] == assigned[qid]['collection_seed']
        assert row['candidate_order'] == assigned[qid]['candidate_order']
        assert row['source_split'] == row['partition'] == chunk['partition']
        assert manifest[qid]['full_index'] == row['attempt2_activation_index']
        a1, a2, sae = tensors[chunk['partition']]
        assert torch.equal(a1[row['attempt1_activation_index']], state['attempt1'][site][successful.index(qid)])
        assert torch.equal(a2[row['attempt2_activation_index']], state['attempt2'][site][local_a2])
        assert torch.equal(sae[row['sae_attempt1_index']], a1[row['attempt1_activation_index']])
        assert torch.equal(sae[row['sae_attempt2_index']], a2[row['attempt2_activation_index']])
print(f'PASS: {len(rows)} collected, {len(excluded)} excluded, {len(reserved)} reserved; merge and activation loaders verified.')
PY
```

## 6. Exercise label-package and probe loading — optional but recommended

This uses the **existing rule-based candidates labeler**, not Claude. It labels
saved text and does not rerun the models or alter the frozen assignments. With
six questions, there may be too few eligible examples or classes for meaningful
probe fitting. The check below tests alignment/loading without fitting a probe.

```bash
python scripts/prepare_behavior_v01.py \
  --full-results "$SMOKE_DIR/final/results/$SMOKE_RUN" --layout partitioned \
  --label-policy v0.2.0-candidate --output "$SMOKE_DIR/labels"

python - <<'PY'
import os
from pathlib import Path
import pandas as pd
from mas_sae.probe import behavior_data as bd
root = Path(os.environ['SMOKE_DIR'])
run = os.environ['SMOKE_RUN']
results = root / 'final' / 'results' / run
labels = root / 'labels'
cols = bd.Columns()
episodes = bd.build_episode_table(results, ['train', 'validation', 'test'], [])
aligned, diagnostics = bd.align_episodes(episodes, pd.read_csv(labels / 'labels.csv'),
    pd.read_csv(labels / 'split_manifest.csv'), cols, 'full_index')
matrix = bd.load_layer_matrix(aligned, root / 'final' / 'activations' / run / 'layer_08')
assert matrix.shape[0] == len(aligned) == len(episodes)
print('PASS: probe labels, partitions, and activation rows align.')
print('Eligible examples:', int(bd.as_bool(aligned[cols.eligible]).sum()))
PY
```

Keep the smoke outputs and worker logs for review. Do not treat a passing smoke
test as a statistical evaluation or as validation of a future synthetic release.
