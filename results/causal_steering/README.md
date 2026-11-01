# Preliminary steering runs (not feature-specific causal evidence)

These are Raye's feature-steering runs from #139, preserved from PRs #145 (layer 25) and #144 (layer 33) under #204.
They are the runs that exposed the edit-size problem in Instructor Review #6 (#150). Their effects track how large
each edit was, not which features were edited. Cite them only as preliminary results that motivated the corrected
design in #146, never as evidence that the selected features cause adoption.

| Run | Layer | Episodes | Edits |
|---|---|---|---|
| `20261020T031535823052Z/` | 25 | 450 intervention episodes, free-form `natural_4b_full9` collection | TopK, L1 and pretrained SAEs; reconstruction and error-preserving splices; steering at s ∈ {−2, +2, +5} × ref; 7 frequency-matched random draws |
| `20261020T031903225060Z/` | 33 | same | same |

Both replays reproduced collection exactly: 450 of 450 A2s matched (`replay_check.json`).

## Why the effects are not feature-specific

The four problems Instructor Review #6 names, with the numbers from the files in each run folder:

1. **Edit sizes differ.** Mean edit norm relative to |x| (`diagnostics.csv`), targeted vs random:

   | SAE | s = −2 | s = +2 | s = +5 |
   |---|---|---|---|
   | L1, layer 25 | 2.83 vs 0.08 (37×) | 0.98 vs 0.07 (14×) | 3.82 vs 0.18 (22×) |
   | L1, layer 33 | 2.68 vs 0.15 (18×) | 1.07 vs 0.14 (8×) | 3.88 vs 0.35 (11×) |
   | TopK, layer 25 | 0.11 vs 0.82 | 0.05 vs 0.28 | 0.16 vs 1.10 |

   - Answers almost never changed below an edit of about 0.4·|x| in any condition (#152).
   - The large L1 effects (e.g. +73 pp targeted minus random at layer 25, s = 2) therefore come from edits 8–37× larger than their controls.
   - The TopK effect (−25 pp) is the reverse: there the random edits are larger.
2. **One always-on feature dominates the L1 set.**
   - Layer 25: feature 7025 fires on 100% of fit-partition rows (`results/feature_selection/20261018T025217187826Z/top_features_L1 SAE.csv`) and makes up 86% of the set's summed reference activation (`replay_config.json`).
   - Layer 33: feature 8023 fires on 99.8% of rows and makes up 91%.
3. **Too few random controls.** 7 random draws per condition; a confirmatory comparison needs at least 100.
4. **Negative clamps are out of range.** s = −2 sets latents to −2 × ref, a value these SAEs never produce:
   - L1: as low as −93.4 (layer 25) and −91.1 (layer 33);
   - pretrained SAE: as low as −5,450.6 (layer 25) and −1,866.0 (layer 33).

The targeted features were also selected on adopted vs not adopted in the free-form collection, where "not adopted" mixes Retention with third answers (#191).

## What else is here

- **Files:** each run folder has every table its replay produced. `per_row.jsonl` is too large for git and is at `s3://dats-capstone/shared/group3/results/causal_steering/<run>/per_row.jsonl`.
- **Not for the paper:** `20261020T031535823052Z/steering_table.png` is a raster image, and the paper needs vector figures.
- **Reference activations:** the `feature_reference_activations.json` files under `results/feature_selection/20261016T003252665105Z/` and `20261018T025217187826Z/` are the ones these runs read. #146's runs recompute them with per-latent statistics, which leaves these per-feature values unchanged.
- **Reproduction check (#204):** the current `replay.analyse` on each run's committed rows reproduces `causal_table.csv`, `random_draws.csv` and `transitions.csv` exactly. It also reproduces every original column of `targeted_vs_random.csv`, `dose_response.csv`, `steering_bidirectional.csv` and `diagnostics.csv`; #152 only added columns.

## Corrected design

#146 replaces these comparisons with:
- norm-matched edits, so targeted and random edits have the same size;
- contribution-matched random draws and random directions;
- 7025 and 8023 run alone and without the rest of the set;
- no negative clamps;
- at least 100 draws for the confirmatory comparison.
