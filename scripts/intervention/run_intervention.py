"""
THE INTERVENTION BENCHMARK
==========================

STATUS: pseudocode. Every undefined name in this file is listed in the IMPLEMENTATION CHECKLIST
at the bottom, with what it must do and which existing code to reuse.


QUESTION
--------
Do the targeted features cause the Solver to adopt the Critic's answer?

We edit those features inside the Solver while it writes its second answer (A2), and ask whether
adoption changes more than when we make the same edit to matched random features.


OUTCOME
-------
adopted (1 / 0): the regenerated A2 adopts the Critic's answer, judged by the same labeler that
produced the stored labels. Measured only on eligible disagreement episodes (stored label:
adoption or retention). Says nothing about whether the answer is correct.


WHERE THE EDIT HAPPENS
----------------------
Solver layer 25, at the final prompt token of the A2 prefill, and nowhere else: earlier positions
and decode steps are not edited. (Below the last layer, the edited token's keys and values still
reach every decode step, so the answer can change even when the first token does not.)
A null result means "these features don't matter at this position", not "they don't matter".


EXPERIMENT DESIGN
-----------------
Baseline 1            No SAE, no edit: the plain replay. Every edit is compared to it.

Raw activations       No SAE: the "features" are the layer's neurons, each centered on its mean
                      over the feature-selection fit rows (so ablating a neuron sets it to its mean).
    Control 1         random neurons, edited                       (matched random draws)
    Control 2         targeted neurons, edited

Exp 1  L1 SAE                                (custom trained)


# something to consider - the SAE features will be the same for both of these, that means we can cache the activations.
Exp 2  Pretrained SAE                        (Gemma Scope 2, layer 25, 16k)
           results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big
Exp 3  Pretrained SAE, fine-tuned decoder    (same encoder; decoder fine-tuned on our data)
           results/sae/decoder_finetuning/20261104T164310458065Z/best
           (params.safetensors sha256 a93c7d01…4de6; s3://dats-capstone/shared/group3/results/sae/...)
           Chosen BEFORE any intervention outcome; fixed for the experiment:
             - decoder only, 10 epochs, Adam lr 1e-5, fitted on mixed_final_001 train (attempt 2);
               encoder, b_enc and thresholds identical to Exp 2, so the features, their firing
               frequencies and the selected targeted features are the same as Exp 2's.
             - validation variance explained 0.225 -> 0.794 (probe cohort 0.238 -> 0.770).
             - the 14 targeted features keep decoder cosine 0.928-0.996 with Exp 2 (row norms
               0.87-1.02): edits keep their meaning, fitted to what each feature writes here.
             - not used: the 5-epoch run 20261102T205853702830Z (superseded), the 15-epoch run
               20261104T165208970599Z (+0.02 variance explained for targeted cosine down to 0.88),
               and every full (encoder + decoder) fine-tune: none passed the stability limits, and a
               new encoder would change the features themselves.
           Under the error-preserving splice b_dec cancels, so Exp 3 differs from Exp 2 only in the
           decoder rows of the edited features. Exp 3 vs. Exp 2 therefore asks: does fitting those
           directions to our data change the effect? Control 1 (reconstruction splice) is where
           the better reconstruction itself shows up.
    each with:
    Control 1         SAE spliced in, nothing edited               (reconstruction-damage check)
    Control 2         random sparse features, edited               (matched random draws)
    Experiment        targeted sparse features, edited

Random direction      No SAE: a random unit direction added at each size-matched edit size.
                      The floor: does ANY edit this big move adoption?


"EDITED" MEANS EACH OF THESE (config.edits), ONE GENERATION PER EDIT
--------------------------------------------------------------------
    full_ablation     Set the features to 0 (raw: to their mean). The edit's size depends on the
                      set (pretrained: targeted 2.2% of |x| vs. random 9%), so targeted vs. random
                      is confounded by size here. Reported, but descriptive.
    suppress_1pct     The full-ablation edit rescaled so that |Δ| = 1% of |x|.
    suppress_2pct     The same at 2%.
                      Same size for every set, so targeted vs. random compares DIRECTION, not size.
                      Never pushed past full ablation (cap_at_ablation): a set whose full ablation
                      is smaller than the target size is capped and reported as "≤ r".
                      Sizes come from the edit-norm audit and are fixed BEFORE any outcome is seen.


HOW THE ACTIVATION x IS REPLACED (THE "SPLICE")
-----------------------------------------------
    Control 1 (SAE, no edit)    x' = decode(encode(x))          reconstruction splice
    every edit                  x' = x + Δ                      error-preserving splice
                                Δ  = decode(z edited) − decode(z)
    Every edit keeps x apart from Δ (the SAE's reconstruction error stays in), so every edit's
    control is Baseline 1. Control 1 is a separate check on the SAE itself.


COMPARISONS (paired by question; random draws averaged per question first)
--------------------------------------------------------------------------
    PRIMARY    ONE representation × edit: targeted − mean(random).  See PRE-SPECIFIED ANALYSIS.
    secondary  between-SAE contrasts: (targeted − mean(random)) of one SAE minus another's
               targeted − mean(random) for every other representation × edit
                   (Holm-adjusted across representations, separately for each edit)
               each edited condition vs. Baseline 1
               targeted vs. random direction, at the same size
               Control 1 vs. Baseline 1: reconstruction damage, reported separately


PRE-SPECIFIED ANALYSIS
----------------------
Fixed before any intervention outcome is seen. Every DECIDE below must be filled in (here and in
the config) before the baseline stage runs. Changing one after outcomes are seen makes the result
exploratory.

Where the targeted features come from
    For each representation: the top 10 features by mean |SHAP| on the probe's TRAIN partition,
    restricted to features whose coefficient points toward adoption. So every targeted feature
    is PRO-ADOPTION. The intervention episodes are a separate partition, so the features were
    not selected on the episodes they are tested on.

Hypothesis and direction
    H1: suppressing the targeted (pro-adoption) features LOWERS adoption more than suppressing
        matched random features:      targeted − mean(random)  <  0
    DECIDE: one-sided (tests H1's direction only) or two-sided (any difference).
        Two-sided is the conservative default. One-sided is valid only because the direction is
        stated here, before the run; it must not be switched after seeing results.

Primary endpoint: the ONE confirmatory test of H1
    DECIDE: representation × edit, e.g.  L1 SAE × suppress_2pct
        targeted − mean(random) adoption, paired by question,
        bootstrap 95% CI over questions, permutation p-value, sidedness as above.
    No multiplicity correction: it is a single test. Every other result is secondary.

Secondary: between-SAE contrasts (what the design is built to compare)
    For representations A and B, at the primary edit, paired by question:
        [ targeted − mean(random) ]_A  −  [ targeted − mean(random) ]_B
    DECIDE which contrasts, e.g.
        L1 SAE                 vs. Pretrained SAE
        Fine-tuned decoder     vs. Pretrained SAE
        L1 SAE                 vs. Fine-tuned decoder
    Holm-adjusted across the contrasts. Only this test supports a claim like "L1 features are
    more causal than pretrained features": one representation significant and another not is
    NOT evidence that they differ.

Secondary: everything else
    The other comparisons listed above. Reported with CIs and Holm-adjusted p-values where
    stated; none of them is used to confirm H1.

Sample size
    N = DECIDE eligible episodes, the first N of the seeded shuffle. (Pool ≈ 3,500; Run B used
    250.) N is required in the config; there is no default.
    Chosen by a power estimate for the primary endpoint, done before the run:
        smallest effect of interest   DECIDE pp of adoption (e.g. 3 pp)
        alpha                         0.05, sidedness as above
        power                         0.80
        paired variance               from EXISTING runs (e.g. the TopK run's per-question
                                      targeted − random differences), never from this cohort
    Record the estimate and its inputs here once done. Cost: N × 82 generations.


GENERATIONS PER EPISODE (5 random draws, 3 random directions)
-------------------------------------------------------------
      1   Baseline 1
    + 3   Control 1                       (3 SAEs)
    + 72  4 representations × 3 edits × (1 targeted + 5 random)
    + 6   3 random directions × 2 size-matched edits
    = 82                                  × N episodes


HOW TO RUN
----------
    python scripts/intervention/run_intervention.py baseline  --config configs/intervention.yaml
    python scripts/intervention/run_intervention.py intervene --config configs/intervention.yaml --shard 0
    python scripts/intervention/run_intervention.py merge     --config configs/intervention.yaml

1. baseline    Regenerate every episode's A2 with no edit, and check that it reproduces the
               stored answer. If it does not, no intervention result means anything: stop.
               Then freeze the run (episodes, feature sets, provenance) and split it into shards.

2. intervene   One GPU worker, one shard. Check this worker reproduces the baseline, then run
               every condition above for every episode in the shard. Each answer is saved the
               moment it is generated, so an interrupted worker resumes where it stopped.

3. merge       Combine the shards, score every answer, and run the comparisons above.

Everything except the intervention stays identical across conditions:
same question, context, critic feedback, prompt, and generation settings.
"""


# ---------------------------------------------------------------------
# READ THE COMMAND AND THE CONFIG
# ---------------------------------------------------------------------

args = parse_args()
#   args.command   baseline | intervene | merge
#   args.config    path to the experiment's YAML
#   args.shard     which shard to run (intervene only)

config = load_config(args.config)
#   config.seed
#   config.layer                         25
#   config.partition                     "intervention"
#   config.number_of_episodes            N, REQUIRED (see Sample size); first N of the seeded shuffle
#
#   config.interactions_path             stored episodes (question, A1, critic feedback, stored A2)
#   config.contexts_path                 exact collected context for each question
#   config.labels_path                   stored labels: eligibility and the stored outcome
#   config.label_policy                  the labeler that produced the stored labels
#
#   config.solver                        model id + revision, dtype, greedy decoding, max_new_tokens
#   config.representations               Raw activations, L1 SAE, Pretrained SAE,
#                                        Pretrained SAE (fine-tuned decoder): name, kind, path
#                                        fine-tuned decoder = results/sae/decoder_finetuning/
#                                        20261104T164310458065Z/best (kind: gemma_scope; see Exp 3)
#   config.feature_selection_summary     targeted features + matched random draws per representation
#                                        must be a run that used the checkpoints above: its recorded
#                                        sae_sha256 is checked (replay.feature_sets). The current
#                                        20261102T210433855083Z run used the 5-epoch decoder, so rerun
#                                        configs/feature_selection/feature_selection_pretrained_adapted_layer25.yaml
#   config.number_of_random_draws        5
#
#   config.edits                         full_ablation, suppress_1pct (-0.01), suppress_2pct (-0.02)
#   config.number_of_random_directions   3
#
#   config.primary_endpoint              representation + edit, e.g. L1 SAE × suppress_2pct
#   config.test_sidedness                one_sided_lower | two_sided
#   config.between_representation_contrasts   pairs of representations, e.g. [L1 SAE, Pretrained SAE]
#
#   config.min_baseline_match            0.98: share of baseline LABELS that must match stored
#   config.number_of_shards
#   config.scorer
#   config.on_scoring_failure            stop | exclude_whole_episode
#   config.n_bootstrap, config.n_permutations
#   config.output_dir

set_random_seed(config.seed)

run_dir = config.output_dir

episodes = load_intervention_episodes(config)


# =====================================================================
# STAGE 1: BASELINE — NO EDIT, EVERY EPISODE
# =====================================================================

def run_baseline_stage():

    solver = load_solver_model(config.solver)
    record_provenance(run_dir / "provenance.json", solver)

    # appends one line per answer; on a rerun, episodes already in the file are skipped
    results = ResultsFile(run_dir / "baseline.jsonl")

    for episode in episodes:

        if results.has_episode(episode.id):
            continue

        prompt = build_solver_a2_prompt(episode)

        generation = generate_answer(
            solver=solver,
            prompt=prompt,
            intervention=None,
        )

        results.append({
            "episode_id": episode.id,
            "experiment": "baseline",
            "representation": "none",
            "condition": "no_edit",
            "edit": None,
            "random_draw": None,
            "edited_features": [],
            "answer": generation.answer,
            **generation.telemetry,
        })


    # ----------------------------------------------------------------
    # BASELINE CHECK: does the plain replay reproduce the stored answers?
    # ----------------------------------------------------------------

    check = check_baseline_reproduces_stored(results.rows(), episodes, config.min_baseline_match)
    write_json(check, run_dir / "baseline_check.json")
    write_csv(check.mismatches, run_dir / "baseline_mismatches.csv")

    if not check.passed:
        raise RuntimeError(
            f"Baseline does not reproduce the stored answers ({check.failures}). "
            "Stopping before any intervention."
        )


    # ----------------------------------------------------------------
    # FREEZE THE RUN, THEN SPLIT THE EPISODES INTO SHARDS
    # ----------------------------------------------------------------

    representations = load_representations(config)
    write_json(
        freeze_run(config, episodes, representations, run_dir / "provenance.json"),
        run_dir / "frozen_run.json",
    )

    shards = split_into_shards(episodes, config.number_of_shards)
    write_json(shards, run_dir / "shards.json")


# =====================================================================
# STAGE 2: INTERVENE — ONE SHARD OF EPISODES, ON ONE GPU WORKER
# =====================================================================

def run_intervention_stage(shard_id):

    # require_passed_baseline_check(run_dir / "baseline_check.json")
    # TODO: restore before real runs. Without it, a worker can edit on top of a baseline that
    #   failed (or never finished) the reproduction check.

    shard_episodes = load_shard(run_dir / "shards.json", shard_id, episodes)

    solver = load_solver_model(config.solver)
    require_same_provenance(run_dir / "provenance.json", solver)

    # A representation is a basis whose features we can edit: raw activations, or an SAE.
    # Its feature sets are fixed once, and every episode uses the same ones:
    #   .targeted_features       the features feature selection picked
    #   .random_feature_draws    config.number_of_random_draws sets, same size as the targeted
    #                            set, matched to it, never including a targeted feature
    representations = load_representations(config)
    require_matches_frozen_run(run_dir / "frozen_run.json", shard_episodes, representations)

    # Before any edit: this worker must regenerate the baseline exactly (prompt, answer, first
    # token). Otherwise differences below could come from the machine, not the intervention.
    baseline = read_rows_by_episode(run_dir / "baseline.jsonl")
    require_worker_reproduces_baseline(solver, shard_episodes, baseline)

    # appends one line per answer; on a rerun, finished episodes are skipped
    # (a partially finished episode is redone from the start)
    results = ResultsFile(run_dir / f"shard_{shard_id}.jsonl")

    for episode in shard_episodes:

        if results.has_complete_episode(episode.id):
            continue

        prompt = build_solver_a2_prompt(episode)

        run_random_directions(episode, prompt, baseline[episode.id], solver, results)

        for representation in representations:
            run_representation(representation, episode, prompt, baseline[episode.id], solver, results)

    mark_shard_complete(results, shard_episodes, run_dir / f"shard_{shard_id}.complete")


def run_representation(representation, episode, prompt, baseline, solver, results):

    # ------------------------------------------------------------
    # CONTROL 1: SAE SPLICED IN, NOTHING EDITED   (SAEs only)
    # Reconstruction-damage check: does swapping x for the SAE's reconstruction, by itself,
    # change the answer? Compared to Baseline 1 and reported separately.
    # Raw activations have no reconstruction, so they skip this.
    # ------------------------------------------------------------

    if representation.is_sae:

        generation = generate_answer(
            solver=solver,
            prompt=prompt,
            compare_to=baseline,
            intervention={
                "layer": config.layer,
                "position": "final_prompt_token_prefill_only",
                "representation": representation,
                "splice": "reconstruction",
                "features": [],
                "edit": None,
            },
        )

        results.append({
            "episode_id": episode.id,
            "experiment": representation.name,
            "representation": representation.kind,
            "condition": "no_edit",
            "edit": None,
            "random_draw": None,
            "edited_features": [],
            "answer": generation.answer,
            **generation.telemetry,
        })


    for edit in config.edits:

        # ------------------------------------------------------------
        # CONTROL 2: RANDOM FEATURES, THIS EDIT
        # ------------------------------------------------------------

        for random_draw, random_features in enumerate(representation.random_feature_draws):

            generation = generate_answer(
                solver=solver,
                prompt=prompt,
                compare_to=baseline,
                intervention={
                    "layer": config.layer,
                    "position": "final_prompt_token_prefill_only",
                    "representation": representation,
                    "splice": "error_preserving",
                    "features": random_features,
                    "edit": edit,
                },
            )

            results.append({
                "episode_id": episode.id,
                "experiment": representation.name,
                "representation": representation.kind,
                "condition": "random",
                "edit": edit.name,
                "random_draw": random_draw,
                "edited_features": random_features,
                "answer": generation.answer,
                **generation.telemetry,
            })


        # ------------------------------------------------------------
        # EXPERIMENT: TARGETED FEATURES, THIS EDIT
        # ------------------------------------------------------------

        generation = generate_answer(
            solver=solver,
            prompt=prompt,
            compare_to=baseline,
            intervention={
                "layer": config.layer,
                "position": "final_prompt_token_prefill_only",
                "representation": representation,
                "splice": "error_preserving",
                "features": representation.targeted_features,
                "edit": edit,
            },
        )

        results.append({
            "episode_id": episode.id,
            "experiment": representation.name,
            "representation": representation.kind,
            "condition": "targeted",
            "edit": edit.name,
            "random_draw": None,
            "edited_features": representation.targeted_features,
            "answer": generation.answer,
            **generation.telemetry,
        })


def run_random_directions(episode, prompt, baseline, solver, results):

    # ------------------------------------------------------------
    # RANDOM DIRECTION: NO SAE, A RANDOM DIRECTION OF THE SAME SIZE
    # The floor for the size-matched edits: does any edit this big move adoption?
    # Each direction is a fixed, seeded unit vector, the same for every episode.
    # Full ablation has no single size to match, so it gets no random direction.
    # ------------------------------------------------------------

    for edit in config.edits:

        if not edit.is_size_matched:
            continue

        for random_draw in range(config.number_of_random_directions):
            
            generation = generate_answer(
                solver=solver,
                prompt=prompt,
                compare_to=baseline,
                intervention={
                    "layer": config.layer,
                    "position": "final_prompt_token_prefill_only",
                    "representation": None,
                    "splice": "error_preserving",
                    "direction_seed": random_direction_seed(config.seed, random_draw),
                    "edit": edit,
                },
            )

            results.append({
                "episode_id": episode.id,
                "experiment": "random_direction",
                "representation": "none",
                "condition": "random",
                "edit": edit.name,
                "random_draw": random_draw,
                "edited_features": [],
                "answer": generation.answer,
                **generation.telemetry,
            })


# =====================================================================
# STAGE 3: MERGE — COMBINE, SCORE, COMPARE
# =====================================================================

def run_merge_stage():

    shards = read_json(run_dir / "shards.json")

    results = read_rows(run_dir / "baseline.jsonl")
    for shard_id in range(len(shards)):
        require_shard_complete(run_dir / f"shard_{shard_id}.complete")
        results += read_rows(run_dir / f"shard_{shard_id}.jsonl")

    require_every_condition_present(results, episodes, run_dir / "frozen_run.json")


    # ----------------------------------------------------------------
    # SCORE EVERY GENERATED ANSWER
    # (done here, not on the GPU workers, so workers never need scorer credentials)
    # ----------------------------------------------------------------

    scorer = load_scorer(config.scorer, config.label_policy)
    episodes_by_id = {episode.id: episode for episode in episodes}

    for row in results:
        row["score"] = score_answer(
            scorer=scorer,
            episode=episodes_by_id[row["episode_id"]],
            answer=row["answer"],
        )

    # A scoring failure is never counted as retention: stop, or drop the WHOLE episode from
    # every condition (so comparisons stay paired) and record which episodes and why.
    results, excluded = handle_scoring_failures(results, policy=config.on_scoring_failure)

    write_csv(results, run_dir / "results.csv")
    write_json(excluded, run_dir / "excluded_episodes.json")


    # ----------------------------------------------------------------
    # COMPARE
    # Every comparison is paired by question, with random draws averaged per question first.
    # Bootstrap CIs resample questions; p-values are permutation tests.
    # ----------------------------------------------------------------

    # PRIMARY: the ONE confirmatory test of H1 (see PRE-SPECIFIED ANALYSIS).
    # targeted − mean(random) at config.primary_endpoint, with config.test_sidedness.
    primary = primary_endpoint(
        results,
        representation=config.primary_endpoint.representation,
        edit=config.primary_endpoint.edit,
        sidedness=config.test_sidedness,
    )
    write_json(primary, run_dir / "primary_endpoint.json")

    # SECONDARY: does one representation's targeted effect beat another's?
    # [targeted − mean(random)]_A − [targeted − mean(random)]_B at the primary edit,
    # paired by question, Holm-adjusted across the contrasts.
    contrasts = between_representation_contrasts(
        results,
        pairs=config.between_representation_contrasts,
        edit=config.primary_endpoint.edit,
    )
    write_csv(contrasts, run_dir / "between_representation_contrasts.csv")

    # SECONDARY: targeted − mean(random), for every representation × edit.
    # Holm-adjusted across representations, separately for each edit.
    targeted_vs_random = targeted_minus_random(results)
    write_csv(targeted_vs_random, run_dir / "targeted_vs_random.csv")

    # Each edited condition (targeted, random) vs. Baseline 1.
    effects = each_condition_vs_baseline(results)
    write_csv(effects, run_dir / "effects_vs_baseline.csv")

    # Targeted vs. a random direction of the same size (size-matched edits only).
    vs_direction = targeted_minus_random_direction(results)
    write_csv(vs_direction, run_dir / "targeted_vs_random_direction.csv")

    # Control 1 vs. Baseline 1: does splicing the SAE in, by itself, change answers?
    damage = reconstruction_damage(results)
    write_csv(damage, run_dir / "reconstruction_damage.csv")

    # What a reader needs to trust the above: edit sizes, capped and no-op edits, per-draw
    # effects, label transitions, changed answers, and how well random draws were matched.
    write_diagnostics(results, run_dir)

    summary = write_summary(
        run_dir / "summary.json",
        primary=primary,
        contrasts=contrasts,
        targeted_vs_random=targeted_vs_random,
        effects=effects,
        vs_direction=vs_direction,
        damage=damage,
        baseline_check=read_json(run_dir / "baseline_check.json"),
        excluded=excluded,
    )
    print(summary)


# =====================================================================
# RUN THE REQUESTED STAGE
# =====================================================================


def main():
    if args.command == "baseline":
        run_baseline_stage()

    elif args.command == "intervene":
        run_intervention_stage(args.shard)

    elif args.command == "merge":
        run_merge_stage()

    else:
        print("Unknown command")

if __name__ == "__main__":
    main()


# =====================================================================
# IMPLEMENTATION CHECKLIST
# =====================================================================
#
# Every undefined name above, in the order it is used. "Reuse" points at code that already does
# the job in the old pipeline (src/mas_sae/causal/), so implementing usually means moving it.
# Mark each item [x] when it is implemented and tested.
#
# Sections: SETUP, STAGE 1, STAGE 2, STAGE 3 (required for the benchmark),
#           OUTSIDE THIS SCRIPT (required, done before the baseline stage),
#           OPTIONAL SPEEDUPS (not required; the benchmark is correct without them).
#
#
# ---- SETUP ----------------------------------------------------------
#
# [ ] parse_args()
#     command (baseline | intervene | merge), --config, --shard (intervene only).
#
# [ ] load_config(path)
#     YAML -> typed config with the fields listed at the top. Refuse unknown keys.
#     Assert greedy decoding (do_sample: false) and a pinned solver revision.
#     Edits: each has .name, .size (None for full_ablation, e.g. -0.01), .is_size_matched.
#     Refuse to load while the pre-specified analysis is incomplete: number_of_episodes,
#     primary_endpoint, test_sidedness and between_representation_contrasts are required (no
#     defaults); the primary endpoint must name a configured representation and edit; every
#     contrast must name two configured representations.
#     Reuse: replay.ReplayConfig.from_dict (its validation).
#
# [ ] set_random_seed(seed)
#
# [ ] load_intervention_episodes(config)
#     Keep only eligible adoption/retention episodes of config.partition, Natural critic
#     condition, exactly one stored episode per question, whose stored A2 relabeled with
#     config.label_policy reproduces the stored label. Attach each episode's exact collected
#     context (checked against its question). Shuffle with config.seed, THEN take the first
#     config.number_of_episodes, so the cohort is a random sample. Refuse if N exceeds the pool.
#     Record any excluded episodes and why.
#     Reuse: replay.load_episodes, replay.load_paragraphs.
#
#
# ---- STAGE 1 --------------------------------------------------------
#
# [ ] load_solver_model(solver_spec)
#     Same model, revision, dtype, prompt and greedy settings as collection; resolve the layer's
#     module for the hook.
#     Reuse: the solver half of replay.load_components.
#
# [ ] record_provenance(path, solver)
#     git commit, code tree hash, uncommitted-code hash, package versions, GPU model, solver
#     spec, hashes of every input file and representation file.
#     Reuse: replay.provenance.
#
# [ ] ResultsFile(path)
#     Append-only JSONL: one line per generation, flushed immediately.
#     .append(row), .rows(), .has_episode(id), .has_complete_episode(id)
#     On open: cut off an unfinished last line (interrupted write); any other bad line = corrupt.
#     Reuse: replay.repair_tail, replay.read_rows.
#
# build_solver_a2_prompt(episode)
#     The exact revise prompt from collection (question, context, A1, critic feedback).
#     Reuse: replay._solve.
#
# [ ] generate_answer(solver, prompt, intervention, compare_to=None) -> generation
#     Greedy-generate A2. With an intervention, hook the layer and replace x at the final
#     prompt token of the prefill ONLY (first forward call); every later call is untouched.
#       splice "reconstruction":    x' = decode(encode(x))
#       splice "error_preserving":  x' = x + Δ,  Δ = decode(z edited) − decode(z)
#       edit full_ablation:         edited features set to 0
#       size-matched edit (r < 0):  the full-ablation Δ rescaled to |Δ| = |r|·|x|, never past
#                                   full ablation (cap_at_ablation); zero if no feature fired
#       direction_seed:             Δ = r·|x|·u, u a seeded random unit vector
#     Raise if the hook never ran, or if the prompt hash differs from compare_to's.
#     generation.answer, and generation.telemetry =
#       prompt_token_hash, generated_tokens, reached_token_budget,
#       first_token_id, first_token_logprob,
#       baseline_first_token_logprob   log p(compare_to's first token) under this condition
#       n_active_features              features firing at x before the edit
#       n_edited_active                edited features that were firing; 0 => no-op edit
#       raw_edit_rel_norm              |Δ| / |x| before rescaling
#       applied_edit_rel_norm          |Δ| / |x| as applied
#       edit_scale, edit_capped        rescaling factor; whether cap_at_ablation applied
#       recon_rel_error                |x − decode(encode(x))| / |x|
#     Keep the speed-up that reuses the shared prefill and skips byte-identical edits; rows must
#     be identical to generating each condition separately.
#     Further speedups go behind this function only; see OPTIONAL SPEEDUPS.
#     Reuse: replay.SAESplice, replay.DirectionSplice, replay.FirstTokenLogits,
#            prefix.PrefixCache, replay.generate_episode, replay._row.
#
# [ ] check_baseline_reproduces_stored(rows, episodes, min_match) -> check
#     Passes only if: every episode has a baseline row; ZERO prompt-hash differences from
#     collection; label match rate >= min_match (labels, not exact text). With Claude labels,
#     only an exact stored answer reuses its stored label; any other counts as a mismatch.
#     check.passed, .failures, .match rates, .mismatches (one row per non-exact episode, kind:
#     prompt_differs | label_changed | formatting_only | text_differs_same_label).
#     Reuse: replay.replay_check, replay.replay_mismatches.
#
# [ ] load_representations(config)
#     Load raw activations (neuron centers) and each SAE. For each: targeted features and
#     config.number_of_random_draws matched random draws from the feature-selection summary.
#     Refuse unless: each representation's file hashes equal the ones feature selection used,
#     and it ran on config.layer; every set is non-empty, integer, unique, within width; each
#     random draw is the targeted set's size and shares no targeted feature. Random draws are
#     matched on firing frequency then decoded contribution (raw neurons: on std alone).
#     .name, .kind, .is_sae, .targeted_features, .random_feature_draws, .match_quality
#     Reuse: feature_selection.load_sae, replay.feature_sets, replay.validate_features,
#            offline.contribution_matched_controls, offline.match_quality.
#
# [ ] freeze_run(config, episodes, representations, provenance_path)
#     Everything a worker must reproduce: resolved config, episode order, every feature set,
#     provenance, and a hash of each. Nothing in it may change once workers start.
#
# [ ] split_into_shards(episodes, n)
#     Contiguous, balanced shards of the frozen episode order.
#     Reuse: distributed.prepare_distributed_run.
#
#
# ---- STAGE 2 --------------------------------------------------------
#
# [ ] require_passed_baseline_check(path)
# [ ] load_shard(shards_path, shard_id, episodes)
#     Refuse if the current episodes don't reproduce the frozen order for this shard.
#
# [ ] require_same_provenance(path, solver)
#     This worker's code, packages, GPU model, solver and input hashes equal stage 1's.
#     Reuse: replay.validate_provenance_identity.
#
# [ ] require_matches_frozen_run(path, shard_episodes, representations)
#     Resolved episode order and feature sets equal frozen_run.json. Also checked on resume.
#
# [ ] read_rows_by_episode(path)
#     {episode_id: baseline row}.
#
# [ ] require_worker_reproduces_baseline(solver, shard_episodes, baseline)
#     Regenerate this shard's baseline on this worker; prompt hash, answer and first token must
#     equal stage 1's for every episode.
#     Reuse: distributed._validate_worker_baseline.
#
# [ ] random_direction_seed(seed, draw)
#     Fixed per draw, identical for every episode and worker (old pipeline: 1000·seed + draw).
#
# [ ] mark_shard_complete(results, shard_episodes, marker_path)
#     Every (episode, experiment, condition, edit, draw) row present exactly once; then write
#     the marker with row count and file hashes. Merge accepts only marked shards.
#     Reuse: distributed._validate_row_keys, the COMPLETE marker in distributed.run_shard.
#
#
# ---- STAGE 3 --------------------------------------------------------
#
# [ ] read_json(path), read_rows(path), write_json(obj, path), write_csv(rows, path)
#
# [ ] require_shard_complete(marker_path)
#     Marker exists and its file hashes match the shard file.
#
# [ ] require_every_condition_present(results, episodes, frozen_run_path)
#
# [ ] load_scorer(scorer_spec, label_policy), score_answer(scorer, episode, answer)
#     The SAME labeler as the stored labels. Rules: same rule version. Claude: the source run's
#     exact model id, prompt version/hash, sampling settings, token limit and retries. An answer
#     identical to the stored A2 reuses its stored judgment; identical answers share one request.
#     Keep the requests/responses as an audit; resume without re-paying for finished requests.
#     score = adopted (1/0). Keep agreement / out labels as their own counts.
#     Reuse: semantic.label_run, semantic.generation_label, behavior_labels.classify_candidate.
#
# [ ] handle_scoring_failures(results, policy)
#     stop: raise. exclude_whole_episode: drop that episode from EVERY condition, return which
#     and why. Never count a failure as retention.
#
# [ ] primary_endpoint(results, representation, edit, sidedness)
#     The ONE confirmatory test. At that representation × edit: per question, targeted minus
#     the mean over random draws; mean difference in pp, bootstrap 95% CI over questions,
#     permutation p-value (one-sided toward "lower", or two-sided, as configured). No Holm.
#     Also record N, the no-op and capped rates on both sides, and the excluded episodes.
#     Reuse: offline.targeted_vs_random (add the one-sided option).
#
# [ ] between_representation_contrasts(results, pairs, edit)
#     For each pair (A, B), per question:
#         d_A = targeted_A − mean(random_A),   d_B = targeted_B − mean(random_B)
#     then d_A − d_B: mean in pp, bootstrap CI over questions, permutation p (two-sided).
#     Holm across the pairs. Paired works because every representation runs on the same episodes.
#     New: no equivalent in the old pipeline; build on offline.paired_effect.
#
# [ ] targeted_minus_random(results)
#     SECONDARY. Per representation × edit: targeted − mean over draws of random. Paired by question;
#     bootstrap CI over questions, and a second CI that also resamples draws; permutation p;
#     Holm across representations within each edit. Also on edited rows only (no-op edits
#     dropped), with targeted and random no-op rates beside it.
#     Reuse: offline.targeted_vs_random, offline.holm, the compare() block in replay.analyse.
#
# [ ] each_condition_vs_baseline(results)
#     Acceptance per condition; Δ vs. Baseline 1, paired by question, bootstrap CI, permutation p.
#     Reuse: offline.paired_effect.
#
# [ ] targeted_minus_random_direction(results)
#     Size-matched edits only: targeted − mean over random directions at the same size.
#     Reuse: offline.targeted_vs_random.
#
# [ ] reconstruction_damage(results)
#     Per SAE, Control 1 vs. Baseline 1: acceptance Δ (CI, p), % answers / labels / first
#     tokens changed, Δ log p(baseline's first token), mean reconstruction error.
#     Reuse: the damage block in replay.analyse, replay._diagnostics.
#
# [ ] write_diagnostics(results, run_dir)
#     edit sizes before/after rescaling and % capped (per representation × edit × side),
#     no-op rates, per-draw effects and targeted − each draw, label transitions
#     (baseline -> condition), every changed answer side by side, random-draw match quality.
#     Reuse: replay._pairs, replay._diagnostics, the random_draws / transitions /
#            changed_answers / control_match blocks of replay.analyse.
#
# [ ] write_summary(path, ...)
#     One JSON with every table above, the baseline check, exclusions, provenance, and the
#     written definitions of the outcome, the edit position and the edits. If the baseline
#     check failed, mark the whole summary UNVALIDATED.
#     Lead with the primary endpoint and its pre-specified sidedness; list every other result
#     under "secondary".
#
#
# ---- OUTSIDE THIS SCRIPT, BEFORE THE BASELINE STAGE ---------------------
#
# [ ] Power estimate (fills in "Sample size" above)
#     From EXISTING runs only (e.g. the TopK run's per-question targeted − random differences):
#     estimate the paired variance, then the N that detects the smallest effect of interest at
#     alpha 0.05 and power 0.80 with the chosen sidedness. Record the inputs and the result in
#     the docstring and set config.number_of_episodes.
#
# [ ] Edit-norm audit for L1 (fills in config.edits)
#     python scripts/causal_edit_norm_audit.py on the L1 SAE: full-ablation edit sizes of the
#     targeted set and every random draw. Choose suppress sizes below them so that neither side
#     is capped.
#
#
# ---- OPTIONAL SPEEDUPS ----------------------------------------------
#
# Not needed for correctness. Current cost: ~0.43 s per generation (TopK run), so 82 × N
# generations, e.g. ~34 GPU-hours for the whole ~3,500-episode pool. Build these only if runtime
# becomes the bottleneck. Each one lives inside generate_answer, so the benchmark above does not
# change, and each must pass the parity check before any real run uses it.
#
# [ ] Batch one episode's conditions together
#     Today every condition is generated alone (batch size 1), which leaves the GPU mostly idle
#     during token-by-token decoding. All of an episode's conditions share the same prompt and
#     length, so they can run as one batch with no padding. Plausibly several times faster.
#     [ ] Layers 0–25 captured once per episode and broadcast across the batch
#         The edit is at layer 25, so layers 0–25 are identical for every condition.
#         Reuse: prefix.PrefixCache (capture/replay); lift its batch-size-1 check.
#     [ ] Batched splice hook: a different edit per batch row
#         Reuse: replay.SAESplice._edit / DirectionSplice._edit (the math already works on a
#         batch of rows); make the edit stats per row (they currently sum over everything
#         and index row 0); lift the batch-size-1 check in SAESplice._hook.
#     [ ] Per-row first-token log-probs
#         Reuse: replay.FirstTokenLogits, indexed per row.
#     [ ] Split the batched output back into one row per condition
#         Reuse: the loop and duplicate-edit handling in replay.generate_episode.
#     [ ] Batch the baseline the same way
#         Batch size can change floating-point results; the baseline must be generated the
#         same way as the conditions, and must still pass the baseline check.
#
# [ ] Parity check (required before any speedup is used)
#     Regenerate a few hundred finished episodes with the speedup and compare every row with
#     the reference run, field by field (except seconds). Require identical answers, or show
#     that every difference is a near-tie between the top two tokens. Record the result in the
#     docstring.
#     Reuse: scripts/check_replay_reuse.py; reference rows: the TopK run's per-row data in S3.
