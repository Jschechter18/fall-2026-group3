# Implement the intervention experiment in `src/mas_sae/intervention/`

<!-- Before sending: fill in the feature_sets.json path (line ~43) after the probe run. -->

## Goal

Write the causal intervention experiment from scratch as a new, self-contained module,
`src/mas_sae/intervention/`, driven by `scripts/intervention/run_intervention.py`.
Replace the pseudocode in that file. Keep its design rationale as the docstring, updated to
match this prompt.
**Readability comes first.** A grad student who has never seen this repo should be able to
read `run_intervention.py` top to bottom and know exactly what the experiment does.

Read first:

1. `scripts/intervention/pre-prompt.md`: requirements and the final output tables. **Authoritative**,
   except that the analysis plan below replaces its notes on the primary endpoint and Holm families.
2. `scripts/intervention/run_intervention.py`: pseudocode and design rationale (where the edit
   happens, splices, provenance, baseline check, sharding, resumability). Use it for intent and
   structure. Where it conflicts with this prompt or pre-prompt.md, they win. Its stale parts:
   only one edit size (chosen by the pilot, below); 3 random draws, not 5; no random-direction experiment; no TopK SAE;
   the one-direction "suppress pro-adoption features" design is replaced by the label-dependent
   arms below.

### What to reuse, and what not to

- **Reuse artifacts**: SAE checkpoints, raw-activation centers, stored episodes, contexts, stored
  Claude labels and the Claude source-run settings, and `feature_sets.json`.
- **Reuse small, clean, general utilities** only when importing them is clearer than writing them:
  SAE loading (`mas_sae.probe.feature_selection.load_sae`), the Claude labeler
  (`mas_sae.evaluation.claude_validator`), and the Solver agent (`mas_sae.agents.solver.Solver`).
- **Do not import the old causal pipeline** (`src/mas_sae/causal/*`, `scripts/causal_*.py`,
  `scripts/distributed_causal_replay.py`). Read it to understand the details that must stay the
  same: how the A2 prompt is built (`replay._solve`), the prefill-only hook, how JSONL tails are
  repaired, the provenance fields, and how the Claude labels are produced. Then reimplement those
  pieces plainly in the new module. Do not modify or delete the old pipeline.
- **A2 prompt:** every intervention episode has `a2_presentation: "candidates"`, so A2 is produced
  by `Solver.choose(question, paragraphs, solver_attempt_1, critic_advocated_answer,
candidate_order)`, not `revise`. Refuse any episode with another presentation.
- The new module must reproduce the old pipeline's prompt and plain replay exactly. The baseline
  check enforces this.

## Input: feature sets (produced by a separate, earlier step)

Feature selection has already been rerun (see `scripts/intervention/probe_prompt_draft.md`). It
wrote `feature_sets.json`: for each representation, its checkpoint path and sha256, its 10
adoption features and 10 retention features (with coefficients), and 3 matched random draws for
each set. Path: `/results/feature_selection_intervention_layer25/20261105T025730344698Z/feature_sets.json` 

Treat this file as read-only input. Do not reselect features. The preflight verifies that each
checkpoint's sha256 matches the hash recorded in the file.

Representations: Raw activations (control), L1 SAE, Pretrained SAE, Fine-tuned decoder, and Fully
fine-tuned † (exploratory).

## The experiment

Per episode, the arms depend on the episode's stored label. Every arm is one greedy generation of
A2. The edit is applied at layer 25, at the final prompt token of the prefill only.

| Stored label (stratum) | Arms                                          |
| ---------------------- | --------------------------------------------- |
| adoption → retention   | decrease adoption · increase retention · both |
| retention → adoption   | decrease retention · increase adoption · both |

For each representation (Raw + 4 SAEs), each arm runs once with the targeted features and once
per random draw (3 draws). For "both", random draw k pairs adoption draw k with retention draw k
from `feature_sets.json`. Each half is matched to the targeted set it replaces, and no new draws are
needed.

Per episode there is also one baseline (no SAE, no edit), and one reconstruction splice with no
edit for each SAE.

Generations per episode: 1 + 4 + 5 reps × 3 arms × (1 + 3) = 65. N = 1000 → 65,000.

### The edit (new code: `intervention/edits.py`)

Everything is error-preserving: x' = x + Δ, where x is the layer-25 activation at the final prompt
token. Every Δ has size |Δ| = r·|x|, where r is the single edit size in the config, chosen by
the pilot (see Commands) before the main run. Implement and unit-test these edits:

- **Feature direction** dᵢ:
  - SAE: the unit decoder row of feature i.
  - Raw activations: the unit basis vector eᵢ, multiplied by the sign of neuron i's probe
    coefficient toward its set's label. This way +dᵢ always means "more of that label", for
    neurons too.
- **Set direction** for a feature set S: u_S = normalize(Σᵢ∈S dᵢ).
- **Increase S**: Δ = +r·|x|·u_S.
- **Decrease S**: Δ = −r·|x|·u_S, the mirror of increase.
  - This is a steering edit, not an ablation. It moves x against the direction the features
    write, whether or not they are firing, so it is never a no-op.
  - It is defined the same way for every representation, raw neurons included.
  - Do not implement an ablation-based decrease (zeroing features in z). That was considered and
    rejected: no-op edits and capping would make targeted and random edits differ in size.
- **Both**: Δ = r·|x|·normalize(u_increase − u_decrease), one combined edit of size r in total.
  - All three arms are then the same size, so differences between arms come from direction alone.
  - Do not apply r to each half. That would make "both" a bigger push (up to 2r·|x|), and a
    higher flip rate could come from size rather than from combining the two sets.
- **Reconstruction splice** (SAEs, no edit): x' = decode(encode(x)).
- Each edit returns Δ together with its diagnostics:
  - applied |Δ|/|x|, which should always be r;
  - how many of the edited features were active before the edit. Report the share of episodes
    where at least one targeted feature fired, so a reader can tell whether a steered direction
    was "live" at that token;
  - reconstruction error |x − decode(encode(x))|/|x|.
- Per representation, once per run: the cosine between u_adoption and u_retention, which shows
  whether "decrease one" and "increase the other" are nearly the same edit.

Random draws use exactly the same edit, with the random features in place of the targeted ones.

## Outcome and statistics

"Flip" means Claude labels the regenerated A2 as the opposite of the episode's stored label.

Labeling happens at merge, batched, resumable and audited. It must use the exact Claude labeler
that produced the stored labels: Claude via AWS Bedrock (`mas_sae.models.claude.ClaudeClient`,
AWS profile `school`, us-east-1), with the settings recorded in
`results/claude_validation/mixed_final_001/intervention/run_settings.json`:
- model `us.anthropic.claude-sonnet-4-5-20250929-v1:0`;
- prompt `claude_binary_v3_candidates`, sha256 `b589c557…9162`;
- temperature 0.0, max_tokens 512, 3 attempts.

Read these values from that file and fail if the current prompt's sha256 differs. Do not switch
to a newer model: a different labeler would make "flips" partly labeler disagreement.

The labeling prompt compares A2 with A1 and C1 by meaning: paraphrases match, refusals are
positions, and correctness is never judged. Use it unchanged, through
`claude_validator.build_labeling_prompt`. Give it:
- the same question, A1 and C1 as the stored episode;
- the candidates display (`candidates_shown(a1, c1, candidate_order)`);
- the regenerated A2 in place of the stored A2.

Rules for sending the requests:
- **One A2 per request.** Send requests concurrently, but never put several answers in one prompt.
- **Blind the labeler.** Never tell Claude the condition, that the answer came from an
  intervention, the stored label, or the stored A2.
- Save Claude's `reason` with every label, so every flip can be audited.

- An A2 identical to the stored A2 reuses its stored label.
- Identical answers share one request.
- A labeling failure is never counted as retention. Retry it; if it still fails, drop the whole
  episode from every condition and record why.
- Agreement and out labels count as "not flipped", and are also reported as their own counts.

### Analysis plan (fixed before any outcome is seen; put it in the config and the docstring)

Every comparison is computed the same way:

- Δ = targeted flip − mean flip over the 3 random draws, paired by question.
- 95% bootstrap CI, resampling questions.
- Paired permutation p-value: swap targeted and random within each question.

The primary claim is "an SAE's features are causal", not one particular SAE or direction.

**Primary (Table P): "some SAE works"**

- Tests: the 3 pre-specified SAEs (L1 SAE, Pretrained SAE, Fine-tuned decoder) × the 3 arms
  (decrease, increase, both) = 9 tests.
- Strata pooled: an arm's adoption-stratum and retention-stratum versions are mirror images.
  "Decrease" pools decrease-adoption on adoption episodes with decrease-retention on retention
  episodes, and likewise for the other arms. N ≈ 1000.
- One-sided: the alternative is targeted flips more than random.
- Holm across the 9 tests. The primary claim holds if at least one test survives.
- Excluded from this family: Fully fine-tuned (exploratory) and Raw activations (the control).
- Which SAE and direction are highlighted in the write-up may be decided after the run. The
  family above may not.

**Secondary (each its own Holm family; two-sided unless stated)**

- **Table 1** (per stratum, from pre-prompt.md): its "p (Holm)" is corrected within each
  representation (6 rows). Fully fine-tuned † forms its own family.
  No row is bold: the primary result is Table P.
- **Table 2, between-SAE contrasts** (from pre-prompt.md), strata pooled, one row per arm:
  - Pre-specified family: L1 vs Pretrained, Fine-tuned decoder vs Pretrained, L1 vs Fine-tuned
    decoder.
  - Exploratory family: Fully fine-tuned † vs Pretrained.
- **Table D, direction contrasts (new):** which direction works best. Within each representation,
  strata pooled, paired by question, compare the arms' Δs: both − decrease, both − increase,
  increase − decrease. Holm across the 3 contrasts within each representation.
  As with SAEs, one arm significant and another not is NOT evidence that they differ; only these
  contrasts support "direction X is best".
- **Reference rows and Table 3 diagnostics** (from pre-prompt.md) are descriptive: no tests.
  Table 3's "% no-op" and "% capped" columns are replaced by:
  - "% with a targeted feature firing";
  - the cosine between u_adoption and u_retention.

Write every table as CSV and markdown, plus `summary.json`. `summary.json` leads with Table P and
lists everything else under "secondary" or "exploratory".

## Commands

```
python scripts/intervention/run_intervention.py pilot   --config configs/intervention.yaml
python scripts/intervention/run_intervention.py prepare --config configs/intervention.yaml
python scripts/intervention/run_intervention.py run     --config configs/intervention.yaml --shard K
python scripts/intervention/run_intervention.py merge   --config configs/intervention.yaml
```

Every command also takes `--check-only`, which runs the preflight and exits.

- **pilot** (one machine with a GPU and Bedrock access): choose the edit size r.
  - **Why:** the probes can't calibrate r. In the existing matched-edits run
    (`results/probe_mixed_final_001_layer25/matched_edits`), the probe's prediction flips at about
    0.0007% of |x|, and even random directions swing it at 1-2.5%. So r must come from real
    generations.
  - **Episodes:** apply the same eligibility filter and seeded shuffle as prepare. The study takes
    positions 1-1000; the pilot takes positions 1001-1050. The two sets never overlap. Record the
    pilot episode IDs, and have prepare refuse if any of them is in the study cohort.
  - **Conditions per pilot episode:**
    - one baseline;
    - for each of the 3 primary SAEs (L1, Pretrained, Fine-tuned decoder), each size in
      `pilot_edit_sizes` (default [0.02, 0.05, 0.10, 0.20]), and each of the 3 arms: one targeted
      edit and the 3 random draws.
    - That is about 50 × (1 + 3 × 4 × 3 × 4) ≈ 7,250 generations, roughly an hour on the A10G.
  - **Same machinery:** use the same generation, edit, results-file and labeling code as the main
    run, so the pilot also tests the pipeline end to end. It is resumable like the other stages.
  - **Output:** `pilot_report.md` and `pilot_report.csv`. For each size, SAE and arm, report:
    - targeted flip %, random flip % and their difference;
    - % of answers labeled "out";
    - % of degenerate answers (empty, repeated tokens, or the token budget hit without an answer);
    - 5 example answers per size.
  - **Then stop.** I choose r and write it into the config as `edit_size`. A suggested rule, which I
    may override: the smallest size where the targeted flip rate clearly exceeds the random flip
    rate for at least one SAE, while "out" and degenerate answers stay under about 5%.
  - The pilot uses no study episodes, and its results are never part of the study tables.
- **prepare** (one machine with a GPU)
  1. Run the preflight. Refuse unless `edit_size` is set and `pilot_report.csv` exists. Record
     both in `frozen_run.json`.
  2. Select the episodes: eligible adoption/retention episodes in the intervention partition, one
     per question, shuffled with seed 42; take the first N = 1000. Refuse if fewer are eligible.
  3. Freeze `frozen_run.json`: config, episode order, feature sets, provenance, and a hash of each.
  4. Generate the baseline for every episode and check it reproduces the stored A2. The label
     match rate must be at least 0.98, with zero prompt differences. If the check fails, stop.
  5. Write the shards.
- **run** (one GPU per shard)
  1. Run the preflight.
  2. Check that `frozen_run.json` is unchanged and that this worker reproduces the baseline for
     its shard.
  3. Generate every condition. Append and flush each row as soon as it is generated.
  4. On restart, skip finished (episode, condition) rows.
  5. When all rows are done, write a completion marker with the row count and the file hash.
- **merge** (one machine; needs AWS profile `school` for Bedrock)
  1. Verify every shard marker, and that every condition appears exactly once.
  2. Label the answers.
  3. Compute the tables.

## Suggested module layout (adjust if something reads better)

```
src/mas_sae/intervention/
  config.py        typed config + validation (refuses unknown keys and missing required fields)
  preflight.py     every check, one combined error message
  episodes.py      load + select + freeze the cohort
  solver.py        load Gemma, build the A2 prompt, generate with an optional edit hook
  representations.py  raw basis + SAE wrappers with a common encode/decode/direction interface
  edits.py         increase / decrease / both / reconstruction, with diagnostics
  conditions.py    the list of (representation, arm, targeted|random draw) for one episode
  results_file.py  append-only, resumable JSONL
  shards.py        split, completion markers, merge checks
  labeling.py      Claude labeling at merge, cached and audited
  stats.py         paired bootstrap, permutation test, Holm
  tables.py        Tables 1–3
  pilot.py         the edit-size pilot and its report
scripts/intervention/run_intervention.py   pilot | prepare | run | merge, reads like the experiment
configs/intervention.yaml
```

## Non-negotiables (from pre-prompt.md)

1. **Readable.**
   - `run_intervention.py` reads like the experiment description.
   - One function per concept, with descriptive names.
   - No abstraction that is used once, and no clever tricks.
   - Short docstrings that say _why_.
2. **Reuse** artifacts and clearly reusable small utilities. Do not reuse the old causal
   pipeline's code (see above).
3. **Cache everything that can be cached**:
   - Within an episode: build and tokenize the prompt once, and run layers 0–25 once (they are
     identical across all 65 conditions).
   - Encode x once per SAE per episode. The Pretrained and Fine-tuned decoder SAEs share an
     encoder, so they share z.
   - Precompute the set directions u_S once per run.
   - Cache Claude labels by answer text.
   - Batch an episode's conditions together if it stays readable. Any batching or caching must
     give rows identical to the plain one-at-a-time path, and a parity test must prove it.
4. **Preflight before any GPU work, in every stage.** One error message must list everything that
   is missing. Check:
   - config valid, with no missing required fields;
   - every input file present;
   - every checkpoint present, with the sha256 recorded in `feature_sets.json`;
   - feature sets valid: 10 unique ints within width, adoption and retention sets disjoint,
     and every random draw disjoint from both targeted sets;
   - solver weights cached at the pinned revision;
   - CUDA available, with enough GPU memory and disk space;
   - output directory writable;
   - for merge: Bedrock credentials work.
5. **Resumable.** Any stage can be killed at any point and rerun with the same command. It never
   redoes finished work. It never mixes runs: the `frozen_run.json` hash is checked on resume.

## Deliverables and how to work

- The module, the script, `configs/intervention.yaml`, and pytest tests in `src/tests/unit/`.
  Tests run on CPU with a tiny fake model and SAE, and cover:
  - the edit math (size exactly r, signs, raw-neuron signs, "both");
  - resume after a truncated JSONL line;
  - shard completeness;
  - each preflight failure;
  - the stats on synthetic rows.
- A smoke run on this A10G: `prepare` and `run` on 3 episodes with 1 shard, then `merge` with
  labeling stubbed. Use a separate smoke config with a placeholder `edit_size` and its own output
  directory. Smoke results are never used. Report the wall time per generation and the projected total.
- Stop for review after each of:
  1. `edits.py` and its tests;
  2. the full pipeline and the smoke run;
  3. the pilot run and `pilot_report.md`. Do not run prepare until I have set `edit_size`.
- Ask me rather than guess on anything this prompt does not settle.
