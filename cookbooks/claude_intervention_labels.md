# Claude intervention labels

The existing Claude validator accepts `--partition intervention` explicitly.
The default remains `train`; the run never mixes partitions or controlled
episodes. It reuses the same candidate-aware prompt, model defaults, retries,
concurrency, provenance and resume behavior as train/validation/test labeling.

Preview the stored intervention episodes without credentials, API calls or
output files:

```bash
PYTHONPATH=src python scripts/labels/run_claude_validation.py \
  --episodes artifacts/collection/mixed_final_001/prepared/results/mixed_final_001_intervention/intervention/interactions.jsonl \
  --partition intervention \
  --output results/claude_validation/mixed_final_001/intervention \
  --concurrency 64 \
  --dry-run
```

Remove `--dry-run` to make the Claude API calls through Bedrock. To continue an
interrupted real run, repeat the same command with `--resume`. Existing outputs
are otherwise refused. `--limit N` is available for a pilot; use a separate
output directory for a pilot and the full run.

Outputs include `raw_results.jsonl`, `binary_labels.csv`, `exclusions.jsonl`,
`excluded_episodes.json`, `run_settings.json`, and `summary.json`. Adoption and
retention enter the binary table; agreement/out and technical failures retain
their audit records. The existing explicit noncommittal-critic shortcut is
recorded with its rule source, rather than represented as a Claude judgment.

The final intervention (stage 6 of the [README pipeline](../README.md#6-intervention)) reads this
run folder directly as `inputs.stored_labels` in `configs/intervention/intervention.yaml`. It needs
nothing below this point.

## Earlier pipeline: causal replay

The rest of this page belongs to the earlier causal replay pipeline, which the stage-6 intervention
replaced.

After labeling, package all four semantic classes with frozen membership:

```bash
PYTHONPATH=src python scripts/labels/export_final_labels_for_probes.py \
  --claude-runs results/claude_validation/mixed_final_001/intervention \
  --full-results artifacts/collection/mixed_final_001/prepared/results/mixed_final_001_intervention \
  --layout partitioned \
  --output results/behavior/mixed_final_001_intervention_claude
```

This labels **stored** intervention episodes. The mixed replay presets now use
`label_policy: claude` and package this run through `--prepare-inputs`.
After GPU generation and any distributed merge, score changed answers on the
coordinator with `scripts/causal_replay.py --label-only RUN` (add `--dry-run`
to preview without API calls). Model/prompt settings come from the frozen
stored-label run. Successful judgments resume from the audit cache; technical
failures stop analysis by default and are retried on the next invocation.
See `cookbooks/causal_replay/distributed_causal_replay.md` for the complete workflow.

The previously generated `mixed_final_001_intervention_candidate` package is
rule-based, is no longer referenced by these presets, and is not a substitute
for semantic labels.
