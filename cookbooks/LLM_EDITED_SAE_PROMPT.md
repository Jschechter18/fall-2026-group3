# Prompt: build and run the LLM-edited SAE experiment (exploratory)

Paste this whole file to Claude Code on the EC2 machine, from the repository root. It is written
for a session that has none of the conversation in which this was designed, so everything needed
is here. Facts marked **verified** were checked against this repository on 2026-11-11. Items
marked **verify** were not checked yet; check them before relying on them.

**Status (2026-11-11): Phases 0–2 are done, and Phase 3 is built and smoke-tested.** The open
decisions were settled with grill-me and are written into this file (Section 3, Section 6, and
Phases 2, 3, 4 and 7). Phase 1 wrote `artifacts/llm_edit/train_contexts.jsonl`; Phase 2 wrote
`artifacts/llm_edit/sandbox_inputs/` and the sandbox (`src/mas_sae/llm_edit/`); Phase 3's harness is
`scripts/sae/llm_edit_sae.py`. Next: the real editing run (ask first: it costs money).

**Deadline:** the stage-6 labeler is Claude Sonnet 4.5 (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`,
read from the stored labels' `run_settings.json`). It is deprecated and retires on the Claude API on
2026-11-30, and Bedrock already lists it as LEGACY. The new stage-6 run's `merge` must finish before
it is retired. Don't switch labelers: the new groups must be labeled exactly like the study run.

---

## 0. How to work

**Skills.** Use these if they are installed on this machine:

- **grill-me**: Phase 0 is done (decisions below). Use it again only if a new open decision comes
  up. Ask one question at a time, with your recommendation.
- **recycle-code**: use it for every piece of new code. Most of this experiment reuses existing
  code (Section 4). Do not rewrite what exists.
- **explain-to-me**: use it whenever you explain a concept, a result or a design choice to me.
- **readable-code**: use it (installed on the EC2 machine). Also follow the readable-script
  rules below.

**Readable-script rules** (my standing preference for experiment scripts):

- Put the pipeline first: a short `run()` that reads as a table of contents of named steps. Step
  functions go below it, then helpers, then the command line.
- Keep one script in one file unless reuse genuinely needs a module.
- Use descriptive names and explain non-obvious math in comments.
- Match the existing scripts' style. `scripts/intervention/run_intervention.py` and
  `scripts/intervention/interpret_features.py` are good models.

**House rules:**

- **No commits.** Never run `git commit`, `git tag` or `git push`. Leave changes uncommitted. At
  the end of each phase, tell me what changed and what passed, and suggest a commit message.
- **Don't touch the main study's results** (Section 2), and don't edit existing configs. Make
  new configs and new output folders.
- **Don't remove** `src/component/`, `src/docs/`, `src/shellscripts/`, `demo/` or `documents/`.
  The professor wants them there.
- **Stop and ask before spending money:** any Bedrock run beyond a smoke test, and any stage-6
  run. Give me an estimate first.
- **Don't guess.** If a fact in this file turns out wrong, stop, tell me, and propose a fix.
- **Environment:** `conda activate capstone` and `export PYTHONPATH=src`, run from the
  repository root. Data and SAEs come from S3; see `cookbooks/setup.md`.

---

## 1. What we are adding, in one paragraph

The main study steers the Solver (Gemma-3-4B-it) at layer 25 along SAE features and counts
adopt ↔ retain flips (stage 6). Here we ask: **can an LLM improve the pretrained SAE by hand, with
no training, so that its features steer more effectively?** Claude, through the Bedrock API,
writes code in a locked-down sandbox that measures things on **train** data and edits the
decoder rows of the 20 selected features. The edited SAE then goes through stages 5 and 6
unchanged. It is compared with the original pretrained SAE and with a mechanical edit that uses
a fixed formula and no LLM. This result is **exploratory and post-hoc**: it is reported in its own
section, never in the pre-registered Table P.

Motivation, to cite at most once in the discussion: Takehi, Teoh, Zhang and Khattab, "Sparks of
Weight-Space Programming" (blog post, MIT CSAIL, 2026-11-02),
https://rikiyat.github.io/blog/weight-space-programming/. In its Experiment 8, an LLM with a
Python kernel edited 131,072 output weights of Qwen3.5-0.8B (one direction removed from MLP
output columns at two layers) and changed the model's behavior with no gradient training. It is
not peer-reviewed. Cite the steering methods themselves (difference of means, contrastive
activation addition) for the mechanical control.

---

## 2. The main study (verified, from `README.md` and the results)

- **Question:** when a Critic disagrees, are SAE features in the Solver's layer-25 residual stream
  causal for adopting the Critic's answer vs. keeping its own?
- **Stages:** 1 collection · 2 Claude labels · 3 layer selection (layer 25, validation balanced
  accuracy 0.843) · 4 SAEs · 5 feature selection · 6 intervention · 7 analysis.
- **Stage 5:** `scripts/feature_selection/feature_selection.py --config
configs/feature_selection/feature_selection_intervention_layer25.yaml`, then
  `scripts/intervention/build_feature_set_pool.py <run> <run>_pool`. Final:
  `results/feature_selection_intervention_layer25/20261105T025730344698Z_pool/feature_sets.json`.
- **Stage 6:** `scripts/intervention/run_intervention.py` with `choose-r`, `prepare`,
  `run --shard K` and `merge`, config `configs/intervention/intervention.yaml`. The study run is
  `results/intervention_layer25/20261106T000808809026Z-02871d677287/` (r = 0.2, 1,000 episodes,
  merged results in `merge/`, including `labeled_rows.jsonl` and `summary.json`).
- **Pretrained SAE results** (Table P, targeted minus random flip rate, N = 983): decrease +1.7 pp
  [+0.4, +2.9], increase +1.2 pp [+0.2, +2.1], both +2.5 pp [+1.1, +3.9].
- **Pretrained SAE:** Gemma Scope 2,
  `results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big`
  (`config.json` + `params.safetensors`, width 16,384, JumpReLU).

---

## 3. Design decisions already made

1. **One edited SAE, both directions.** Stage 6 uses the adoption set and the retention set on
   every episode (arms increase, decrease, both). So Claude edits the 10 adoption rows toward
   adoption and the 10 retention rows toward retention, in one SAE.
2. **Decoder-only edits.** The encoder, `b_enc` and `threshold` stay read-only. Stage 5 picks
   features from the encoder (verified: a logistic probe on train SAE features, ranked by mean
   |SHAP| on train; `src/mas_sae/probe/feature_selection.py`, module docstring), so stage 5 should
   pick the same 10 + 10 features as for the pretrained SAE. The comparison then isolates one
   question: did Claude point those features in better directions? This parallels the existing
   "Fine-tuned decoder" group (same encoder, new decoder).
3. **Only the 20 selected rows may change.** Every other row stays identical.
4. **Keep each edited row's original length.** Stage 6 normalizes each decoder row before
   steering (verified: `sae_feature_directions` in `src/mas_sae/intervention/generation/edits.py`),
   so scaling a row has no effect on steering; keeping lengths just keeps the edit clean. Stage 5's
   random pools are matched on training firing frequency only (verified:
   `frequency_matched_controls`, `src/mas_sae/probe/feature_selection.py` lines 372-375,
   recorded as "matched on training firing frequency"). Firing frequency depends only on the
   encoder, which is unchanged, so stage 5 should produce the same random pools as for the
   pretrained SAE. The new run's random-control results should then reproduce the old run's.
   That's a free sanity check.
5. **Rotation cap (decided in Phase 0): each edited row keeps cosine ≥ 0.95 with its original
   row** (at most about 18°). Why not 0.7: steering uses the normalized sum of 10 rows, and the
   rows' parts perpendicular to the mean-difference direction `d` mostly cancel in the sum. So
   per-row caps barely limit how close the set direction gets to `d`. Measured on the stage-5 train
   rows (2026-11-11), the mechanical edit's "both"-arm direction has cosine 0.29 with `d` unedited,
   0.88 at a 0.95 cap, 0.93 at 0.9 and 0.97 at 0.7. For scale, the trained decoder fine-tune moved
   these 20 rows by cosine 0.93–0.996. `apply_edit` and the audit report each set direction's
   cosine with `d` as a diagnostic, so the paper can say how close each edited SAE came to
   mean-difference steering.
6. **No training.** No optimizer chooses any weight. Claude may measure (means, differences,
   norms, cosines, counts, quantiles, what a feature fires on) and run a few test steerings. It
   may not fit models (no logistic regression, no gradient descent, no automated search over
   edit values). Every change goes through `apply_edit(...)` with numbers Claude writes out.
7. **Claude sees train data only.** Never validation, test, intervention, or any stage-6 result.
8. **Three groups, all `role: exploratory`:**
   - **LLM-edited SAE**: the experiment.
   - **Mechanical edit**: the essential control. The same edit by formula (Section 6).
   - **Random edit** (included, decided in Phase 0): the same 20 rows rotated by the same angles
     as Claude's edits, in random directions. It rules out "any rotation of this size helps".
9. **Nothing is recollected.** Saved activations, labels and episodes are reused.
10. **Compare with the pretrained SAE across runs.** The new stage-6 run contains only the new
    groups. The pretrained SAE's rows come from the existing study run, paired by episode and
    condition. That's valid because generation is greedy and every episode must reproduce its
    stored baseline. The random-control check in decision 4 confirms it. The check is exact:
    random draws come from the same seed and question ID and exclude both targeted sets, so every
    random-control feature is an unedited row. The new groups' random-control generations should
    therefore be identical, text for text, to the study run's Pretrained SAE random rows.

---

## 4. What to reuse (verified locations)

| Need                                                 | Existing code                                                                                                                                                                                                                           |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Load an SAE folder                                   | `load_sae` in `src/mas_sae/probe/feature_selection.py`; `load_gemma_scope_sae` and `JumpReLUSAE` in `src/mas_sae/sae/pretrained.py` (parameter names `w_enc`, `b_enc`, `w_dec`, `b_dec`, `threshold`; `w_dec` has shape latent × input) |
| Save an SAE in Gemma Scope format                    | `save_model(model, sae_config, destination)` in `src/mas_sae/sae/finetuning/loop.py` (writes `params.safetensors` + `config.json`, loadable as `kind: gemma_scope`)                                                                     |
| Steering directions and edits                        | `sae_feature_directions`, `set_direction`, `edit_delta` in `src/mas_sae/intervention/generation/edits.py`                                                                                                                               |
| Apply one condition to one episode                   | `apply_condition`, `encode_once` in `src/mas_sae/intervention/generation/episode.py`                                                                                                                                                    |
| Generate A2 (baseline and edited)                    | `load_solver`, `generate_baseline`, `generate_with_activation` in `src/mas_sae/intervention/generation/solver.py`                                                                                                                       |
| Build episodes from interactions + contexts + labels | `eligible_episodes` in `src/mas_sae/intervention/experiment/episodes.py` (natural critic, candidates presentation, labels adoption/retention only)                                                                                      |
| Feature sets                                         | `read_feature_sets`, `build_representation`, `load_representations` in `src/mas_sae/intervention/experiment/representations.py`                                                                                                         |
| Paired statistics                                    | `bootstrap_ci`, `permutation_p`, `paired_difference`, `holm` in `src/mas_sae/intervention/analysis/stats.py`                                                                                                                            |
| Bedrock Claude client                                | `AdaptiveThinkingClaudeClient` in `src/mas_sae/evaluation/claude_adaptive_client.py` (single-turn `generate`; see Phase 3)                                                                                                              |
| Train data and labels                                | `mas_sae.probe.behavior_data`, used by stage 5 with the paths in `configs/feature_selection/feature_selection_intervention_layer25.yaml`                                                                                                |
| What a feature fires on                              | `scripts/intervention/interpret_features.py` (profiles and example episodes per feature)                                                                                                                                                |

---

## 5. Facts about the data (verified)

- **Train labels:** `results/claude_validation/mixed_final_001/train` (the stage-2 labels;
  `eligible_episodes` reads labels from such a folder). Stage 5 reads the derived
  `results/behavior/mixed_final_001_claude_probe/labels.csv` instead. The two agree on all
  72,810 train questions (checked on EC2, 2026-11-11). Use the stage-5 file for `y_train`, so the
  sandbox's labels match the ones the features were selected with.
- **Train episodes:**
  `artifacts/collection/mixed_final_001/prepared/results/mixed_final_001/train/interactions.jsonl`.
  The train folder has only `interactions.jsonl` and `exclusions.jsonl`, with **no
  `contexts.jsonl`**.
- **Passages for train episodes** are needed to regenerate A2 in `steer_test`. They're in the
  collection manifest, `artifacts/collection/mixed_final_001/manifest.json`, where
  `questions[*]` has `question_id` and `payload`, and the payload holds `paragraphs` and
  `question`. Collection wrote the intervention partition's contexts in this shape
  (`src/mas_sae/experiments/distributed_collection/core.py`, around line 215):
  `{"question_id": ..., "payload": ...}`. Write the same format for the train question IDs.
  **No model runs are needed.** In the manifest, non-intervention questions have
  `partition: null` because the split was assigned later, so select by the question IDs in the
  train `interactions.jsonl`.
- **Train activations:** `.../activations/mixed_final_001/layer_25/train_attempt2.pt` (the stage-5
  config uses `activation_file: "{split}_attempt2.pt"`). Load them through the same loader stage 5
  uses, so rows align with labels.

---

## 6. The mechanical control (the formula)

1. `d = normalize(mean(train adoption activations) − mean(train retention activations))`, using
   the same layer-25 rows and labels stage 5 uses.
2. For each selected **adoption** feature with original row `r`, and `u = r / |r|`:
   `p = normalize(d − (d·u) u)` (the part of `d` perpendicular to `u`), then
   `new = |r| · (0.95 u + sqrt(1 − 0.95²) p)`. That's cosine 0.95 with the original (the cap in
   decision 5), as far toward `d` as the cap allows, with the original length.
3. For each selected **retention** feature, do the same with `−d`.
4. Leave every other row unchanged. Save with `save_model`.

The random edit uses the same rotation angles as Claude's final edit, row by row, toward random
unit directions perpendicular to each row, with a fixed seed.

---

## 7. Phases

Work phase by phase. End each one with a short report: what was built, what was checked, and
what's next.

### Phase 0: Grill me, and check the machine (done 2026-11-11)

**Decisions:**

| # | Decision | Value |
|---|---|---|
| 1 | Rotation cap | Per-row cosine ≥ 0.95 (decision 5) |
| 2 | Random-edit group | Included (decision 8) |
| 3 | Editing model | `us.anthropic.claude-opus-5`, effort `xhigh` (Phase 3; Fable 5.2 and Opus 5.5 can't be called on the school account) |
| 4 | Budget | 40 snippets; 4 `steer_test` calls × 200 train episodes; a dollar cap only as a runaway guard ($5,000; was $60) (Phase 3) |
| 5 | Feature hints | Given, with caveats; `description_test_spearman` dropped (Phase 2) |
| 6 | Sandbox | Docker, stock `ubuntu:26.04` image with the host conda env mounted read-only, `--memory 8g` (Phase 2) |
| 7 | Stage-6 route | `role: primary` plumbing in a separate config, no pipeline changes (Phase 7) |
| 8 | Claude's brief | Approved as written in Phase 3 |

**Machine check (EC2, 2026-11-11):**

- Conda env `capstone` imports `mas_sae`; CUDA works with 1 GPU (A10G, 23 GB); 4 CPUs; 15.4 GiB
  RAM and no swap; 54 GB of disk free.
- Docker 29.1.3 works, and `docker run --network none` blocks network access. The user is in the
  `docker` group; a shell started before that change needs `sg docker -c "docker ..."`.
- `aws sts get-caller-identity --profile school` works (the SSO session expires; renew with
  `aws sso login --profile school --use-device-code`). Every current Claude model is active on
  the account's Bedrock.
- Every input in Sections 2, 4 and 5 is present, including the Gemma snapshot. The read-only
  parts of stage 6's preflight pass. Don't run `run_intervention.py ... --check-only` as a
  read-only check: it creates and deletes a temporary file inside the run folder, and for
  choose-r and merge it calls AWS STS.
- The 9 intervention tests pass
  (`python -m pytest -p no:cacheprovider src/tests/unit/causal/test_intervention*.py`).

### Phase 1: Data prep (no model runs)

- Write the train contexts file (Section 5) to a new, clearly named location, e.g.
  `artifacts/llm_edit/train_contexts.jsonl` (artifacts are gitignored).
- Check that `eligible_episodes(train interactions, train contexts, train labels, "train")`
  loads with no "missing or mismatched collected context" error.
- Read the 10 adoption and 10 retention feature IDs for "Pretrained SAE" from the final
  `feature_sets.json` through the existing reader. Print them.

### Phase 2: The sandbox

Claude's code runs here. Snippets need CPU only; Gemma runs outside (Phase 3).

- **Docker (decided in Phase 0):**
  - **Image:** the stock `ubuntu:26.04` image (it matches the host's glibc 2.43), with the host's
    `capstone` conda env mounted read-only at its host path, so package versions are identical
    to the host's. No Dockerfile and no pip downloads; the only download is pulling the image
    once.
  - **Flags:** `--network none`, `--read-only` root filesystem, a `--tmpfs /tmp`,
    `--cap-drop ALL`, `--security-opt no-new-privileges`, `--pids-limit 256`, a non-root user,
    `--cpus 3`, `--memory 8g --memory-swap 8g`.
  - **Why 8 GB:** the machine has 15.4 GiB and no swap, and the kernel stays alive while
    `steer_test` holds Gemma in a host process. 16 GB is not possible.
  - **Mounts:** one read-only sandbox input folder written by the harness (SAE params, `X_train`,
    `y_train`, `meta_train`, `Z_train`, `hints`, `selected`), and one writable run folder. Don't
    mount the repository, validation, test or intervention data.
  - **`mas_sae` is not importable inside**, by design: it is an editable install pointing at the
    repo's `src/`, which isn't mounted. So `encode`, `decode` and the other helpers are short
    self-contained functions, and a host-side test checks them against `mas_sae`'s
    `JumpReLUSAE`.
- **A persistent kernel** (`jupyter_client`), so variables survive between snippets, like a
  notebook. **Verified:** `jupyter_client` 8.10.0 and `ipykernel` 7.4.0 are in the host env, so
  they're in the sandbox. With `--network none`, the host can't reach ports inside the
  container (the container still has its own loopback). So the kernel **and** its
  `jupyter_client` manager both run inside the container, in a small driver. The host talks to
  the driver over the container's stdin/stdout (`docker run -i`): one JSON request per snippet,
  one JSON reply.
- **Guards at kernel start:**
  - `torch.set_grad_enabled(False)`
  - an import blocker (a `sys.meta_path` finder) for `torch.optim`, `scipy.optimize` and
    `sklearn`
  - `torch.Tensor.backward` replaced with a function that raises
- A time limit of 30 s on each snippet, enforced with `interrupt_kernel()`, so variables
  survive a timeout.
- **Preloaded variables:**
  - `W_enc`, `b_enc`, `threshold` (read-only)
  - `W_dec`, `b_dec` (the decoder, editable only through `apply_edit`)
  - `X_train`, `y_train` (1 = adoption, 0 = retention), `meta_train` (question, A1, the Critic's
    answer, dataset source), from the stage-5 rows and labels (Section 5)
  - `Z_train`: the train codes `encode(X_train)`, computed once on the host and stored sparse.
    The encoder never changes, so they never change. A dense copy would be 25,556 × 16,384
    floats ≈ 1.7 GB.
  - `selected = {"adoption": [...], "retention": [...]}`
  - `hints` (decided in Phase 0): for the 20 features, from
    `results/feature_interpretation_layer25/20261106T221912476834Z/feature_summary.csv`, the
    columns `claude_description`, `claude_confidence`, `strongest_property`,
    `top_promoted_tokens`, `fires_in_pct_of_episodes`, `mean_value_adoption` and
    `mean_value_retention`. All come from train or from the weights alone. Drop
    `description_test_spearman`, which used validation.
- **Preloaded helpers** (all work in chunks of 2,048 rows, to stay well inside 8 GB):
  - `encode`, `decode`
  - `separation(i)`: how well feature _i_ separates adoption from retention on train, with no
    fitting
  - `top_examples(i, k)`
  - `apply_edit(statement)`: applies one edit and appends it to `edits.py`. It rejects edits
    that touch rows outside `selected`, break the 0.95 cosine cap (decision 5), or change a
    row's length. It reports each edited row's cosine with its original, and each set
    direction's cosine with its original and with `d`.
- **Honest limit:** Python guards stop accidental training, not every possible hand-written
  optimization loop. The time limit, the `steer_test` budget and the transcript audit (Phase 5)
  cover the rest.

### Phase 3: The harness (`scripts/sae/llm_edit_sae.py`)

- **Model (decided in Phase 3): `us.anthropic.claude-opus-5`, effort `xhigh`.** Being listed on
  Bedrock does not mean a model can be called. **Verified (2026-11-11), on the school account:**
  - Fable 5.2 is refused: "data retention mode 'default' is not available for this model". It needs
    30-day data retention, which the account would have to enable.
  - Opus 5.5 and Sonnet 5.5 are refused: there's no AWS Marketplace subscription, and the student
    role can't create one.
  - Opus 5 works. The team used it for synthetic data (`configs/data/opus5_v3_full.yaml`).
  - Price at Anthropic's list rates: $5 / $25 per million tokens (Bedrock bills separately).
- **Multi-turn Bedrock calls:** `ConversationClient` in `src/mas_sae/llm_edit/conversation.py`, a
  subclass of `AdaptiveThinkingClaudeClient` that reuses its `_client.invoke_model` and request
  body.
  - It sends `thinking: {"type": "adaptive", "display": "summarized"}`, so the transcript keeps a
    readable summary of Claude's reasoning.
  - The conversation is strictly append-only: each response's content blocks go back unchanged.
    Truncating a snippet's output *before* appending it is fine; editing an earlier message is not.
  - **Verified** with a tiny two-turn call on Opus 5: the thinking blocks were resent and the
    summarized reasoning came back.
- **Budget (decided in Phase 0):**
  - 40 Python snippets and 4 `steer_test` calls.
  - Snippet output is truncated to 4,000 characters before it goes back to Claude.
  - **A dollar cap, only as a guard against a runaway loop: $5,000** (raised from $60 after
    Phase 3; money is not a constraint, and the snippet and `steer_test` budgets are what shape the
    run). Cost is tracked from each response's `usage`. At 80%, tell Claude to wrap up; at 100%,
    stop and save the last valid edit. Estimated cost of a full run: about $15–35 at Anthropic list
    prices for Opus 5 (Bedrock prices separately). The smoke run's three turns cost $0.70.
- **Protocol:** Claude answers each turn with either one fenced `python` block (run in the
  sandbox; the output goes back, truncated), or `steer_test()` (run by the harness; the result
  goes back), or `DONE`. Enforce the budgets, and tell Claude what's left after each turn. A reply
  with no action, or more than one, is not run; after 5 such replies the run stops.
- **The first message** (added after the smoke run, approved; the brief is unchanged) also says:
  "`apply_edit(feature, new_row, why)` takes one feature id, a 2560-number row and a reason. Send
  one action, then stop and wait: the output comes in the next message. Never write an output
  yourself." In the smoke run, Claude guessed `apply_edit`'s signature (its first snippet crashed),
  and wrote an edit, an invented output and `steer_test()` in one reply (refused).
- **`steer_test()` runs outside the sandbox, on the GPU, in a host process that loads Gemma once:**
  - It loads the current edited `W_dec` from the run folder and builds directions with
    `sae_feature_directions`.
  - It runs the stage-6 "both" arm at r = 0.2 on **200 train episodes**, the same 200 every call
    (the first 200 of a fixed seeded order), and regenerates with the existing generation code.
    Generations with the unedited SAE are cached after the first call. Each call takes about 2–4
    GPU-minutes at 0.644 s per generation (measured in the study run).
  - Why 200 and not 20: flips are rare, so at n = 20 a call sees about one flip and can't tell
    edits apart. Don't tell Claude any stage-6 flip rate (decision 7).
  - It classifies each answer cheaply against the episode's A1 and the Critic's answer with
    `answer_matches` / `normalize_answer` in `src/mas_sae/evaluation/scoring.py`. No Bedrock
    calls.
  - It returns, for the edited vs. the unedited SAE on the same episodes: flip counts in both
    directions, unclear answers (matching neither A1 nor the Critic's answer), and the mean change
    in the **candidate margin**: log p(the other answer's first token) − log p(the original
    answer's first token).
    - The candidate first tokens come from collection's `candidate_first_token_ids`. The margin is
      defined only where the two differ (`decision_token_distinct`, about half the episodes).
    - The margin replaced an earlier signal (changed after the smoke run, approved): the change in
      log p of the baseline's first token alone. That signal rewarded wording changes: the smoke
      edit made answers start with "The" ("The Veldra River" instead of "Veldra River"), which
      cost the original first token 7 nats without changing behavior.
    - To keep each generation's first-token log-probabilities, `steer_test` runs
      `generate_episode`'s steps itself, with the same functions. The shared stage-6 code is
      unchanged.
  - **Verified:** on the matcher, the stored train A2s never contradict a stored Claude label,
    and 6% are unclear.
- **Log everything to `results/sae/llm_edited/<run>/`:**
  - `transcript.jsonl`: every message, snippet and output
  - `edits.py`
  - the edited SAE (`params.safetensors`, `config.json`), saved with `save_model`
  - `run.json`: model, effort, budgets, rules text, git commit, input hashes, and the sandbox's
    memory limit, `ubuntu:26.04` image digest and package lists (`conda list --explicit` and
    `pip freeze`)
- **First a smoke run:** 3 turns, 1 `steer_test` with `n = 2`. Then stop and show me the
  transcript before the real run (it costs money; see the house rules).

**The brief Claude receives** (approved in Phase 0). Claude is not told about the mechanical or
random controls, so it can't aim at or away from one formula, and it sees no stage-6 numbers:

> You are editing a sparse autoencoder (SAE) by hand. **Setting:** a Solver model answers a
> question, a Critic disagrees, and the Solver writes a second answer that either adopts the
> Critic's answer or keeps its own. **The test your edit will face:** at layer 25, once, at the
> final prompt token before the second answer, we replace the Solver's activation x with
> x' = x + 0.2·|x|·u. Here u is built from two feature sets of this SAE. The adoption set A and
> the retention set R each have one direction: the normalized sum of their features' unit decoder
> rows (u_A, u_R). In the "both" arm, an episode the Solver originally adopted gets
> u = normalize(u_R − u_A), and one it originally retained gets u = normalize(u_A − u_R). We count
> how often the second answer flips to the opposite behavior, compared with steering along random
> features of similar firing frequency.
>
> **Your goal:** edit the decoder rows of the 10 adoption features and the 10 retention features
> in `selected`, so that this steering flips more answers in both directions.
>
> **Rules:**
> - Change only those 20 rows, and only through `apply_edit(...)`.
> - Keep each row's length.
> - Keep cosine ≥ 0.95 with its original row; `apply_edit` reports each row's cosine, and how the
>   set directions changed.
> - Measure anything you like on the train data.
> - Do not fit models or optimize: no gradients, no regression, no automated search over values.
>   Every number in an edit must be one you choose and can explain.
>
> **What you have:**
> - `X_train`, `y_train` (1 = adoption, 0 = retention), `meta_train`, and the SAE.
> - `Z_train`, the train codes. They're fixed, since the encoder never changes.
> - `encode`, `decode`, `separation(i)` and `top_examples(i, k)`.
> - `hints`: auto-generated descriptions of the 20 features, from train examples and the
>   decoder's logit lens. They may be wrong; check them. Many contrast "fictional vs. real"
>   entities, which largely tracks the dataset source (synthetic vs. MuSiQue), not behavior.
>
> **`steer_test()`** runs the real test on 200 fixed train episodes. It returns flips and the mean
> change in the candidate margin, log p(the other answer's first token) − log p(the original
> answer's first token), on the episodes whose two answers start with different tokens, for your
> current edit vs. the unedited SAE. Flips are noisy at this size; weigh the margin too.
>
> **Budget:** 40 code snippets, 4 `steer_test` calls and a token cap; you'll be told what's left
> after each turn. Each turn, reply with exactly one of: one fenced `python` block,
> `steer_test()`, or `DONE`. You will never see validation or test results.
>
> Explain your reasoning before each edit. Finish with DONE and a summary: what you changed, why,
> and what you'd expect.

### Phase 4: Controls

- `scripts/sae/mechanical_edit_sae.py`: the formula in Section 6, written to
  `results/sae/mechanical_edit/<run>/`.
- `scripts/sae/random_edit_sae.py` (included, decided in Phase 0): same angles as Claude's edit,
  random directions, fixed seed.

### Phase 5: Audit (`scripts/sae/audit_llm_edit.py`)

- Rebuild the SAE by applying `edits.py` to the original pretrained SAE, and confirm the
  `params.safetensors` hash matches the saved one.
- Confirm that only the 20 selected rows differ, that each kept its length, and that each has
  cosine ≥ 0.95 with its original. Report each set direction's cosine with `d`, for all three
  edited SAEs.
- Report the run's token use and cost.
- Scan the transcript for blocked imports, gradient use and suspicious loops, and report counts
  of snippets and `steer_test` calls. Show me `edits.py`.

### Phase 6: Stage 5 for the new groups

- New config: copy `configs/feature_selection/feature_selection_intervention_layer25.yaml` to a
  new file. List only the new groups under `saes` (`kind: gemma_scope`), with a new `output_dir`.
  Run `feature_selection.py`, then `build_feature_set_pool.py`.
- **Check:** each new group's selected 10 + 10 features equal the Pretrained SAE's, and its
  random pool equals the Pretrained SAE's (decision 4). If either differs, stop and tell me.
- **Done (2026-11-12):** config `configs/feature_selection/feature_selection_llm_edited_layer25.yaml`,
  run `results/feature_selection_llm_edited_layer25/20261112T002233925900Z`. The selected features
  match exactly, in order. The random pools do **not**: only 76 of 100 draws are identical, and the
  coefficients differ in the fifth digit. The unedited Pretrained SAE gives the same new pool on this
  machine, so the cause is the machine (tiny numerical differences in the encodings, which move
  firing frequencies), not the edits. A stable sort does not restore the old pool.
  - `build_feature_set_pool.py` isn't needed: current stage 5 already writes all 100 draws.
  - **Decided:** use the study's pool. `scripts/sae/llm_edit_feature_sets.py` writes
    `..._study_pool/feature_sets.json`: each group gets the study's Pretrained SAE entry (features,
    coefficients, 100 draws) with its own path and sha256s. That keeps decision 10's exact check.

### Phase 7: Stage 6 for the new groups

- **Why the groups can't simply be `role: exploratory` here** (found on EC2, 2026-11-11):
  - With no primary groups, Table P is empty and `merge` crashes with a KeyError on
    "Survives Holm (α = 0.05)" (`run_intervention.py`, around line 359).
  - `prepare` needs choose-r's outputs: `choose_r/choose_r_report.csv` (`preflight.py`, around
    line 145) and `choose_r/episodes.json` (`run_intervention.py`, around line 417). It also hashes
    the report into `frozen_run.json`. And choose-r only runs the primary groups
    (`choose_r_conditions`).
- **The route (decided in Phase 0), with no pipeline code changes:** in this separate config only, give the
  new groups `role: primary`. This is plumbing, not a claim. The run lives in its own
  `output_root`, and everything it produces is reported as exploratory.
  - `primary_representations`: the new groups.
  - `between_sae_contrasts`: e.g. `[LLM-edited SAE, Mechanical edit]`.
  - `edit_size: 0.2` and `candidate_edit_sizes: [0.2]`, so choose-r runs only at the fixed size.
  - The same inputs, solver and seed as the study config, the new `feature_sets`, and a new
    `output_root`.
  - Then run `choose-r` (creates the run folder and the files prepare needs), then `prepare`,
    `run --shard K` and `merge`.
  - Note: `summary.json`'s "primary" wording is hard-coded ("3 SAEs × 3 arms", "pre-specified").
    Ignore it for this run, and label everything exploratory in the notebook and the paper.
  - **Verified (2026-11-11):** choose-r accepts a single candidate size
    (`choose_r_conditions` in `src/mas_sae/intervention/experiment/conditions.py`), and the
    validator accepts a config with no raw-activations control group.
- Rejected alternatives: a pipeline change to support no primary groups, and one to copy the
  random rows from the study run instead of regenerating them. Don't change pipeline code.
- **The random rows are duplicates, on purpose.** Each new group's random-control rows reproduce
  the study run's Pretrained SAE random rows (decision 10). That's 9 of every 13 generations,
  and about $95 of the labeling. It's accepted: it buys an exact reproduction check.
- **Cost (measured from the study run: 0.644 s per generation; 567 input and 102 output tokens
  per labeling request):**
  - **choose-r:** 50 episodes × 3 groups at r = 0.2; about 1,850 labels, roughly $6.
  - **The study:** per group, 1,000 episodes × 13 generations (1 reconstruction + 3 arms ×
    (targeted + 3 random draws)) ≈ 2.3 A10G-hours, plus about 13,000 labels ≈ $45.
  - **Total for 3 groups:** about 7 A10G-hours and about $140. This machine has one GPU (A10G), so
    shards run one after another.
  - Ask before starting, and finish `merge` before the Sonnet 4.5 labeler is retired (see the
    deadline at the top).

### Phase 8: Analysis (a new notebook in `notebooks/intervention/`)

- Load the new run's `merge/labeled_rows.jsonl` and the study run's rows for "Pretrained SAE".
  Pair them by episode, arm and condition.
- **Sanity check first:** the new run's random-control rows should be identical to the study
  run's Pretrained SAE random rows, with the same generated text and labels, so the flip rates
  match exactly (decisions 4 and 10). Report any row that differs. A difference means the
  cross-run pairing is not valid, so stop and tell me.
- Report, for each group and arm: targeted flip %, random flip %, the difference with CI
  (`stats.paired_difference`), and raw flip rates. Then the paired contrasts: LLM-edited vs.
  pretrained, LLM-edited vs. mechanical, and LLM-edited vs. random edit. All exploratory, with no
  Holm correction claims. The new run's `summary.json` says "primary" and "pre-specified"
  (hard-coded wording); ignore that.
- Show `edits.py` and a short summary of Claude's reasoning from the transcript.

### Phase 9: Documentation

- Add a short "Exploratory: LLM-edited SAE" entry to `README.md` under earlier and side work,
  with commands, configs and output folders. Keep the main-study sections unchanged.
- Write a one-page summary of the results for the paper's exploratory section. Cover the
  method, the controls, the results and the limits (post-hoc; label-guided; the guards are not
  airtight; the blog is not peer-reviewed).

---

## 8. What success and failure mean

- **Success:** the LLM-edited SAE flips more than the pretrained SAE, in both directions, and
  more than the mechanical edit. Then an LLM improved model weights by hand, without training,
  as measured by our own pre-built test.
- **Matches the mechanical edit:** the LLM's reasoning added nothing beyond the obvious formula.
- **No better than the pretrained SAE:** small adjustments can't improve these features. That
  supports, rather than contradicts, the main result.

All three are reportable. Report whichever happens, plainly.
