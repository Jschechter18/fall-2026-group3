# Two-EC2 smoke run

This is a disposable operational check, not an evaluation. The automated tests
use deterministic fixture activations and a tiny CPU model; this procedure adds
real collection models and real S3 transfer. It has not been executed by Codex
against live EC2/S3. Use the team's existing GPU instances and model access.

## 1. Common shell setup

On the coordinator and both workers, from the repo root in the existing
`capstone` environment:

```bash
export PYTHONPATH=src
export AWS_PROFILE=school
export COLLECTION_RUN=distributed_smoke_001
export RUN_DIR="$PWD/artifacts/collection/$COLLECTION_RUN"
export S3_RUN="s3://YOUR_BUCKET/YOUR_PREFIX/$COLLECTION_RUN"
mkdir -p "$RUN_DIR"
# Authenticate using your team's normal AWS SSO and Hugging Face setup.
```

Choose a new run name for a fresh test. Do not reuse a completed destination.
Both workers must run the implementation containing this workflow, with the
same experiment configuration. Use tmux and durable storage for checkpoints.

## 2. Prepare on the coordinator

This creates a 32-question real MuSiQue sample using the pinned revision and
uses the collection example's existing model settings. Preparation itself does
not load collection models. A dataset download may be needed.

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import yaml
from mas_sae.data.musique import load_musique_examples

root = Path(os.environ['RUN_DIR'])
revision = 'c8f4f8c9465fb69d31a8eae894c3fd509c4ca321'
questions = list(load_musique_examples('train', num_questions=32, revision=revision))
(root/'questions.json').write_text(json.dumps(questions))
(root/'intervention_ids.json').write_text(json.dumps(['musique:'+str(q['id']) for q in questions[:4]]))
cfg = yaml.safe_load(Path('configs/collection/distributed.yaml').read_text())
cfg['dataset'] = {
    'inputs': [{'source': 'musique', 'version': revision, 'source_split': 'train',
                'path': str(root/'questions.json')}],
    'intervention': {'question_ids_file': str(root/'intervention_ids.json')},
    'split': {'proportions': {'train': .6, 'validation': .2, 'test': .2}, 'seed': 43}}
cfg['collection']['layers'] = [8]
cfg['output'] = {'run_name': os.environ['COLLECTION_RUN'], 'chunk_size': 4}
(root/'collection.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

python scripts/distributed_collection.py prepare \
  --config "$RUN_DIR/collection.yaml" --num-chunks 2 --output "$RUN_DIR/manifest.json"
aws s3 cp "$RUN_DIR/manifest.json" "$S3_RUN/manifest.json"
aws s3 cp "$RUN_DIR/collection.yaml" "$S3_RUN/collection.yaml"
```

All 32 questions, including the four intervention questions, are assigned work.
No train/validation/test membership exists yet. No synthetic producer data is
claimed by this smoke run; mixed-source behavior is covered by the fixtures.

## 3. Collect on EC2 A and B

On both workers:

```bash
aws s3 cp "$S3_RUN/manifest.json" "$RUN_DIR/manifest.json"
aws s3 cp "$S3_RUN/collection.yaml" "$RUN_DIR/collection.yaml"
```

EC2 A owns chunk 0:

```bash
python scripts/collect_activations.py --config "$RUN_DIR/collection.yaml" \
  --manifest "$RUN_DIR/manifest.json" --chunk 0
```

EC2 B owns chunk 1:

```bash
python scripts/collect_activations.py --config "$RUN_DIR/collection.yaml" \
  --manifest "$RUN_DIR/manifest.json" --chunk 1
```

On one worker, test interruption after at least one `chunk_*/metadata.json`
checkpoint exists under its assigned work directory. Stop the process, then
rerun its same command with `--resume`. Previously committed batches should
remain; only the unfinished batch repeats. A matching completed work chunk is
skipped by `--resume`. Never run competing owners for the same work chunk.

## 4. Upload assigned results, completion last

EC2 A:

```bash
aws s3 sync "$RUN_DIR/chunks/000000/" "$S3_RUN/chunks/000000/" \
  --exclude complete.json --exclude '*.tmp/*'
aws s3 cp "$RUN_DIR/chunks/000000/complete.json" "$S3_RUN/chunks/000000/complete.json"
```

EC2 B:

```bash
aws s3 sync "$RUN_DIR/chunks/000001/" "$S3_RUN/chunks/000001/" \
  --exclude complete.json --exclude '*.tmp/*'
aws s3 cp "$RUN_DIR/chunks/000001/complete.json" "$S3_RUN/chunks/000001/complete.json"
```

A failed upload can be repeated after the worker is stopped. S3 ETags are not
used as data checksums; merge verifies the saved completion file's SHA-256 facts.

## 5. Merge, split, and label on the coordinator

After both workers finish:

```bash
aws s3 sync "$S3_RUN/chunks/" "$RUN_DIR/chunks/"
python scripts/distributed_collection.py merge --manifest "$RUN_DIR/manifest.json"
python scripts/distributed_collection.py split \
  --manifest "$RUN_DIR/manifest.json" --config "$RUN_DIR/collection.yaml"

python scripts/prepare_behavior_v01.py \
  --full-results "$RUN_DIR/prepared/results/$COLLECTION_RUN" \
  --layout partitioned --label-policy v0.2.0-candidate --output "$RUN_DIR/labels"
python scripts/prepare_behavior_v01.py \
  --full-results "$RUN_DIR/prepared/results/${COLLECTION_RUN}_intervention" \
  --layout partitioned --label-policy v0.2.0-candidate --output "$RUN_DIR/intervention-labels"
```

Inspect `merged/collection.json` and both final `collection_status.json` files.
Every question is either successful or explicitly excluded. Scientific exclusions
can reduce the 32 collected successes; an absent chunk is not an exclusion.

## 6. Exercise real downstream loaders

```bash
python - <<'PY'
import os
from pathlib import Path
import pandas as pd
from mas_sae.causal import replay
from mas_sae.data import production
from mas_sae.probe import behavior_data as bd
from mas_sae.sae.dataloader import create_sae_dataloader

root = Path(os.environ['RUN_DIR'])
name = os.environ['COLLECTION_RUN']
results = root/'prepared/results'/name
acts = root/'prepared/activations'/name
rows, manifest = production.load_labeled_rows(results, root/'labels', layout='partitioned')
for split, purpose in [('train', 'fit'), ('validation', 'tune'), ('test', 'evaluate')]:
    group = [r for r in rows if r['partition'] == split]
    assert group, f'No successes in {split}; choose a larger smoke sample'
    values = production.load_production_activations(group, manifest, acts, purpose=purpose, layer=8)
    assert len(values) == len(group)
    batch = next(iter(create_sae_dataloader(128, split, 0, location=acts/'layer_08')))
    assert batch.shape[0] == 2*len(group)
episodes = bd.build_episode_table(results, ['train', 'validation'], [])
cols = bd.Columns(target='strict_target', eligible='eligible_strict')
aligned, _ = bd.align_episodes(episodes, pd.read_csv(root/'labels/labels.csv'),
    pd.read_csv(root/'labels/split_manifest.csv'), cols, 'full_index')
selected, _ = bd.select_partitions(aligned, cols, ['train', 'validation'])
assert not set(selected.partition) & {'test', 'intervention'}
intervention = root/'prepared/results'/f'{name}_intervention'
cfg = replay.ReplayConfig(capstone_data_root=root,
    interactions_jsonl=str(intervention/'intervention/interactions.jsonl'),
    contexts_jsonl=str(intervention/'contexts.jsonl'),
    labels_csv=str(root/'intervention-labels/labels.csv'), label_policy='v0.2.0-candidate',
    feature_selection_summary='unused-by-input-check.json', layer=8)
held_out = replay.load_episodes(cfg)
contexts = replay.load_paragraphs(held_out, None, Path(cfg.contexts_jsonl))
assert set(contexts) == set(held_out.question_id)
print('SAE, probe alignment/selection, and causal context loading passed.')
PY
```

The causal input check needs eligible baseline episodes; if the tiny reserved
pool has none, enlarge the disposable sample rather than weakening eligibility.
It does not run feature interventions or claim that a four-question cohort is
an evaluation. With the actual chosen SAE/feature artifacts, configure causal
replay as described in the [workflow guide](distributed_collection.md) and run
`--preflight-only` before interventions. Do not lower the baseline reproduction
threshold to make the smoke pass.
