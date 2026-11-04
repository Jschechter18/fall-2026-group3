<!-- PROBE CHANGES -->

- RERUN PROBES TO SELECT TOP 10 RETENTION AND TOP 10 ADOPTION FOR EACH SAE AND CONTROL

PROBLEM:

- Right now, we are only picking the 10 features that the probes determine to be most correlated to adoption
- We want the probe to also pick the 10 features that are most correlated to retention
- We want this done for the 4 SAE checkpoints we will perform the expeirment on
- It should also do this for the raw activations since we are going to use those as a control

<!-- Tasks for intervention -->

<!-- IMPORTANT NOTES: -->

1. Intervention pipeline implementation in run_intervention.py
2. EXTREMELY readable code. MOST IMPORTANT NOTE - WE WANT TO CLEARLY BE ABLE TO SEE THE EXPERIMENT. We want the file script/interventions/run_intervention.py
3. Reuse what can clearly be reused. That means artifacts, existing simple, reusable modularized code
4. Cache and make it extremely computationally efficient -> anything that can be cached, should be cached
5. no op checks - before the run, i want you to check for every checkpoint, label, mdoel, etc that may be needed to run. it is important that runtime is not wasted if the machine does not have everything it needs to run the entirety of the script
6. resumability

<!-- Intervention Tasks: -->

1. Prepare (administrator)
   - This prepares a config file to help us run on N machines
   - This should create shards for n machines to run

2. Run
   - This performs actual experiment.
   - This needs to be the cleanest, clearest code we have ever seen

   BASELINE:
   - FOR EACH EPISODE
   - RAW ACTIVATION ABLATIONS
     - IF EPISODE.LABEL = ADOPTION
       - FIRST, INCREASE FEATURES ASSOCIATED WITH RETENTION
       - SECOND, DECREASE FEATURES ASSOCIATED WITH ADOPTION
       - THIRD, DO BOTH ? - SHOULD WE REALLY INCLUDE THIS?
   - IF EPISODE.LABEL = RETENTION
     - FIRST, INCREASE FEATURES ASSOCIATED WITH ADOPTION
     - SECOND, DECREASE FEATURES ASSOCIATED WITH RETENTION
     - THIRD, DO BOTH ? - SHOULD WE REALLY INCLUDE THIS?

   Main experiment:
   - FOR EACH EPISODE
     - FOR EACH SAE
       - IF EPISODE.LABEL = ADOPTION
         - FIRST, INCREASE FEATURES ASSOCIATED WITH RETENTION
         - SECOND, DECREASE FEATURES ASSOCIATED WITH ADOPTION
         - THIRD, DO BOTH ? - SHOULD WE REALLY INCLUDE THIS?
       - IF EPISODE.LABEL = RETENTION
         - FIRST, INCREASE FEATURES ASSOCIATED WITH ADOPTION
         - SECOND, DECREASE FEATURES ASSOCIATED WITH RETENTION
         - THIRD, DO BOTH ? - SHOULD WE REALLY INCLUDE THIS?

   Controls:
   - FOR EACH EPISODE
     - FOR EACH EPISODE
       - FOR EACH SAE
         - NO ABLATION SAE
         - RANDOM FEATURE EDITS - MEAN_DRAW(RANDOM ABLATIONS) (3 DRAWS)

   WHAT WE WANT TO SEE:
   - are our flip rates for the SAE steering, better than flip rates for the controls that really don't do any steering
   - if they are better, meaning that our interventions successfully flip labels, and they do it at a statsically signifcant higher rate than the controls, then our sae features are causalaly related to the labels

3. Merge (administrator)
   - Once finished, merge all shards
   - TODO: QUESTION - DO WE NEED CLAUDE API TO LABEL HERE

   - Labeling: yes, the regenerated A2s must be labeled by the SAME labeler that produced the
     stored labels (Claude API). Do it at merge time so calls can be batched. A labeling failure
     is never counted as retention: retry, or drop the whole episode and record why.

## Final output tables

All rows use the single edit size (e.g. 2% of |x|). "Flip" = the regenerated A2's label is the
opposite of the episode's baseline label.

### Table 1: Main results (the paper table)

Targeted vs. matched random edits, paired by question. **Bold** = the pre-specified primary
endpoint. † = exploratory (Exp 4 uses a different feature set).

| Representation     | Stratum (baseline → target) | Arm                         | N   | Targeted flip % | Random flip % (mean of 3) | Δ (pp) [95% CI] | p   | p (Holm) |
| ------------------ | --------------------------- | --------------------------- | --- | --------------- | ------------------------- | --------------- | --- | -------- |
| Raw activations    | adoption → retention        | decrease adoption features  |     |                 |                           |                 |     |          |
| Raw activations    | adoption → retention        | increase retention features |     |                 |                           |                 |     |          |
| Raw activations    | adoption → retention        | both                        |     |                 |                           |                 |     |          |
| Raw activations    | retention → adoption        | decrease retention features |     |                 |                           |                 |     |          |
| Raw activations    | retention → adoption        | increase adoption features  |     |                 |                           |                 |     |          |
| Raw activations    | retention → adoption        | both                        |     |                 |                           |                 |     |          |
| L1 SAE             | … same 6 rows …             |                             |     |                 |                           |                 |     |          |
| Pretrained SAE     | … same 6 rows …             |                             |     |                 |                           |                 |     |          |
| Fine-tuned decoder | … same 6 rows …             |                             |     |                 |                           |                 |     |          |
| Fully fine-tuned † | … same 6 rows …             |                             |     |                 |                           |                 |     |          |

Reference rows (no edit), one per stratum:

| Condition                                 | Stratum              | N   | Flip % | Meaning                                                       |
| ----------------------------------------- | -------------------- | --- | ------ | ------------------------------------------------------------- |
| Baseline (no SAE, no edit)                | adoption → retention |     |        | should be ~0%: proves the replay reproduces the stored labels |
| Baseline (no SAE, no edit)                | retention → adoption |     |        | same                                                          |
| SAE spliced in, no edit (one row per SAE) | each stratum         |     |        | flips caused only by the SAE's reconstruction error           |

Column definitions:

- **N**: episodes in that stratum (differs by stratum).
- **Targeted flip %**: share of episodes whose label flipped under the targeted edit.
- **Random flip %**: average over the 3 random draws within each episode first, then across episodes.
- **Δ**: targeted − random, paired by question. This is the "are these features causal?" number.
- **95% CI**: bootstrap over questions.
- **p**: paired permutation test (swap targeted/random within each question).
- **p (Holm)**: adjusted within each pre-specified family.

### Table 2: Between-SAE contrasts (secondary)

The only table that supports "SAE A's features are more causal than SAE B's".

| Contrast (A vs. B)                    | Stratum × arm | Δ_A − Δ_B (pp) [95% CI] | p (Holm) |
| ------------------------------------- | ------------- | ----------------------- | -------- |
| L1 SAE vs. Pretrained SAE             |               |                         |          |
| Fine-tuned decoder vs. Pretrained SAE |               |                         |          |
| L1 SAE vs. Fine-tuned decoder         |               |                         |          |
| Fully fine-tuned † vs. Pretrained SAE |               |                         |          |

### Table 3: Diagnostics (appendix, required)

Shows the edits actually did what we intended. Without it, a null result can't be interpreted.

| Representation | Feature set          | Applied ‖Δ‖/‖x‖ | % no-op (no edited feature firing) | % capped | Recon error | Random-draw match quality |
| -------------- | -------------------- | --------------- | ---------------------------------- | -------- | ----------- | ------------------------- |
|                | adoption (targeted)  |                 |                                    |          |             |                           |
|                | retention (targeted) |                 |                                    |          |             |                           |
|                | random draws         |                 |                                    |          |             |                           |

Plus: baseline label match rate, and the number of episodes excluded for labeling/scoring failures.

### Optional extra column for Table 1

Mean change in log p of the baseline's first token (targeted − random). More sensitive than flip
rate: it catches edits that move the model without flipping the label. Already recorded in
telemetry as `baseline_first_token_logprob`.
