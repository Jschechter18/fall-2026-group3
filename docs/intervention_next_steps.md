# Intervention (#224): where things stand and what to do next

*Written 2026-11-06, after the two choose-r pilot runs. Nothing has been prepared yet.*

## Where things stand

Two choose-r runs, both in `results/intervention_layer25/` (versioned run folders, see `run.json` in each):

| Run | Sizes | Result |
|---|---|---|
| `20261106T000808809026Z-02871d677287` | 0.02, 0.05, 0.10, 0.20 | One row meets the suggested rule: Pretrained SAE, increase, r = 0.2 (+4.7 pp, CI [+0.7, +9.3]). Everything else is null. |
| `20261106T030649156970Z-7064c87d114f` | 0.30, 0.50 (exploratory) | No row meets the rule. "Out" answers climb fast: L1 SAE reaches 16–21% at 0.3 and 24–36% at 0.5. |

- **r stays 0.2.** It is the smallest size that met the rule, and the larger sizes were recorded as exploratory before they ran. Set `experiment.edit_size: 0.2` before `prepare`.
- **Both machines produce identical baselines** for all 50 choose-r episodes, so results from either instance can be compared or combined.
- **The overall picture is null so far:** targeted edits change answers no more than matched random edits. 3 episodes account for 70% of the first run's flips.
- **L1 SAE "both" looked promising at 0.2** (+6.0 pp, steady across sizes, not from the unstable episodes) **but fell to +2.2 at 0.3**, with 21% out answers. Treat it as noise until the full study.
- **Pretrained SAE and Fine-tuned decoder "increase" are positive at 0.2, 0.3 and 0.5** (+1.5 to +5.2 pp), with no CI clearly above 0. That's the most consistent weak pattern.
- Plots: `notebooks/intervention/choose_edit_size_pilot_analysis.ipynb` (first run only).

## Why the effect is small

The edit happens **once**, at layer 25, at the **final prompt token** of the prefill ([solver.py:82-95](../src/mas_sae/intervention/generation/solver.py)). Generated tokens are never edited. The edit reaches the answer only through that token's cached keys and values, and the model can still copy the Critic's answer from the unedited prompt tokens. The model is also very sure: 45 of 50 episodes have a first answer token with probability above 0.99.

This position is a deliberate choice: activations, SAEs and features were all built at the final prompt token ([capture.py:61-64](../src/mas_sae/activations/capture.py)). **Nothing upstream (collection, SAE training, feature selection, probes) needs redoing** to try another steering strategy. Only the intervention changes.

## Next steps, in order

### 1. Fix: labeling failures exclude whole episodes (do first)

The 0.3/0.5 run excluded 5 of 50 episodes for labeling failures. The failure reasons in its `choose_r/labeling/attempts/*/raw_results.jsonl` are `missing_field:solver_attempt_2` (10) and `no_conflict` (7). `missing_field:solver_attempt_2` is most likely an **empty edited answer**. That's a degenerate answer and should count as "out", not drop the episode from every condition. Larger edits produce more empty answers, so this will bias any strong steering strategy. Check what `no_conflict` means too, then decide how the labeler should handle both, before the full study.

### 2. Decide: add a positive control (difference of means)

If a new strategy is also null, we can't tell "the SAE features aren't causal" apart from "steering at layer 25 can't change adoption at all". Add a **difference-of-means direction**: the mean layer-25 activation of adoption episodes minus the mean of retention episodes, from the fit partition, as a pre-specified control representation next to Raw activations.
- If it pulls away from random: steering works here, and the SAE question becomes meaningful.
- If even it doesn't: adoption likely isn't steerable at layer 25 at these positions. Look at another layer, or at the Critic's answer tokens (that *would* need new collection).

Work: compute one vector from existing activations and labels, add a representation kind, include it in feature-set and checkpoint provenance.

### 3. Decide: steer at every generated token too

Add a `steering: final_token | all_generated` option to the edit hook, applying Δ at each new token's layer-25 activation as well. It's stronger, but random edits get stronger too: it's informative only if targeted pulls away from random. Plan:
- Keep the final-token edit as the primary family; add every-step steering as a separate, pre-specified family.
- Needs its **own choose-r**: repeated edits are stronger, so its r is probably below 0.2.
- The prefill cache still works (layers 0–25 of the prefill are unchanged).

Caveat for the write-up: features were learned at the final prompt token only, so using them at answer tokens is an extrapolation (standard in steering work, but say so).

### 4. Then: the full study

`prepare` → `run --shard 0..3` → `merge`, with `run_id` set to the run folder that holds the choose-r report. Expect about 9–10 h on one machine:

| Stage | Time |
|---|---|
| `prepare` | ~15–20 min |
| `run`, each shard (~250 episodes) | ~2 h |
| `merge` (Claude labeling) | ~15–30 min |

Two shards at once on the A10G may help (GPU is ~50% busy); check memory with `nvidia-smi` first. Both machines are interchangeable.

## Open questions to bring to the team

- Adopt the out-rate cap as written (whole row), or apply it to targeted answers only? Decide before any new choose-r.
- `prepare` will run from a folder whose choose-r report doesn't contain r = 0.2 (it's in the first run). Record which run justified r (in the config comment, or have `prepare` record it).
- Report flip rates among answers that aren't "out" too: out answers can't flip, which lowers random's flip rate when random edits garble more (e.g., L1 "both" at 0.2: 8% of random answers out, 0% of targeted).
- Check the one suspicious L1 "both" flip at 0.2: a synthetic aqueduct question where "not mentioned... no engineer can be identified" was labeled adoption.
