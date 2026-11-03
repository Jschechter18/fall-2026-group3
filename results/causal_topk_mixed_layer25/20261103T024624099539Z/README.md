# TopK SAE layer-25 causal replay, 500 episodes, Claude labels (#216)

Generated on main 5525bf6 with the #214 speedups. The work was split into 50 shards of 10 episodes over 5 A10G workers, and generation finished on 2026-11-03.

- **Size:** 500 episodes × 916 conditions = 458,000 generations.
- **Replay check:** 500/500 plain replays exactly match the stored episodes (`distributed/merged_exclude_empty6/replay_check.json`).
- **Speedup check:** the rerun is bitwise identical to the pre-#214 run (b08b8a4) on all 39,863 rows they share, comparing every field except `seconds`.
- **Labels:** semantic labels from Claude via Bedrock.
  - 454,248 rows reproduce the stored answer exactly and reuse its Claude label.
  - The 3,752 changed rows reduce to 202 distinct answers. Claude labeled 196 of them; the other 6 were empty.

## The folder in git and the full run in S3
The tables, summary and figure are in `distributed/merged_exclude_empty6/`. The per-row data is not in git. It is in S3, along with the preflight, the manifest, all 50 shards, the merge before labeling (`distributed/merged`) and the Claude labeling audit:

`s3://dats-capstone/shared/group3/results/causal_topk_mixed_layer25/20261103T024624099539Z/`

## Exclusions: 6 episodes
In 6 synthetic episodes, the targeted ablation (all target features, or f1666 alone) made the Solver stop at its first token, so it gave an empty answer. That is 10 rows. Claude cannot label an empty answer, so these episodes are excluded entirely (`generated_errors: exclude_episode`) and **494 episodes are analysed**.

None of the 150,000 random ablations produced an empty answer. `derived_from.json` records the single config change from `distributed/merged`, and the question ids.

## Headline results: 494 episodes, baseline adoption 51.2%
From `targeted_vs_random.csv`. The control is the contribution-matched random features; p-values are Holm-adjusted.

| Edit | Targeted − random | 95% CI | p (Holm) |
|---|---|---|---|
| Ablate all target features | −5.1 pp | [−7.5, −2.7] | 0.0006 |
| Ablate f1666 alone | −3.6 pp | [−6.0, −1.2] | 0.009 |
| Ablate the targets except f1666 | +0.3 pp | [−0.3, 1.0] | 0.34 |
| Suppress ≤1% or ≤2% of the activation (every subset, and the random-direction controls) | −0.2 to +0.1 pp | includes 0 | 1.0 |

## Caveats
- **Edit size is not matched in the ablation arm.** Ablating f1666 removes 0.86 of |x| on average, while the random ablations remove 0.01–0.03 (`diagnostics.csv`, "Mean edit norm"). The comparisons that are size-matched are the suppress arms and the ablation without f1666, and they show no effect.
- **Part of the drop is disruption.** Under the targeted ablation of all features:
  - 26 answers (5.3%) become "out", compared with 0.05% for random ablations;
  - the answer text changes in 18.8% of cases, compared with 1.6% for random;
  - between adoption and retention, 22 flip from adopt to retain and 7 the other way (`transitions.csv`).
- **Claude labels are not perfectly repeatable.** When the labeling was redone, the new Claude calls judged 2 of 458,000 rows differently from the first, interrupted attempt.

## Follow-ups
- A size-matched random control for the ablation arm.
- For the paper, keep the 6 empty answers as "no answer" (counted as non-adoption) and report this exclusion version as the sensitivity check.
