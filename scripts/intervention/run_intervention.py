"""
Run the intervention experiment.

For every intervention episode:

1. Generate a baseline answer without modifying activations.
2. Test raw activation-feature ablations.
3. Test each SAE using:
   a. no ablation
   b. random sparse-feature ablation
   c. targeted sparse-feature ablation
4. Score every generated answer.
5. Save one result row per episode × condition.
"""

# ---------------------------------------------------------------------
# SET EXPERIMENT CONSTANTS
# ---------------------------------------------------------------------

SEED = 42
LAYER = 25
NUMBER_OF_RANDOM_DRAWS = 3

INTERVENTION_EPISODES_PATH = "..."
TARGET_FEATURES_PATH = "..."
OUTPUT_PATH = "results/intervention/results.csv"

CUSTOM_SAE_PATH = "..."
PRETRAINED_SAE_PATH = "..."
FINETUNED_SAE_PATH = "..."

COMMAND = "prepare" # worker, merge

config = {
    'seed': SEED,
    'layer': LAYER,
    'number_of_random_draws': NUMBER_OF_RANDOM_DRAWS,
    'command': COMMAND
}


# ---------------------------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------------------------

set_random_seed(SEED)

episodes = load_intervention_episodes(INTERVENTION_EPISODES_PATH)

solver = load_solver_model()
critic = load_critic_or_scorer()

custom_sae = load_custom_sae(CUSTOM_SAE_PATH)
pretrained_sae = load_pretrained_sae(PRETRAINED_SAE_PATH)
finetuned_sae = load_custom_sae(FINETUNED_SAE_PATH)

targeted_raw_features = load_targeted_raw_features(
    TARGET_FEATURES_PATH
)

targeted_sae_features = {
    "custom_sae": load_targeted_features(
        TARGET_FEATURES_PATH,
        sae_name="custom_sae",
    ),
    "pretrained_sae": load_targeted_features(
        TARGET_FEATURES_PATH,
        sae_name="pretrained_sae",
    ),
    "finetuned_sae": load_targeted_features(
        TARGET_FEATURES_PATH,
        sae_name="finetuned_sae",
    ),
}

saes = {
    "custom_sae": custom_sae,
    "pretrained_sae": pretrained_sae,
    "finetuned_sae": finetuned_sae,
}

results = []


# ---------------------------------------------------------------------
# RUN THE EXPERIMENT
# ---------------------------------------------------------------------

for episode in episodes:
    
    if config['command'] == "prepare":
        continue
    
    if config['command'] == "worker":
        continue
    
    

    # Everything except the intervention should stay identical:
    # same question, context, critic feedback, prompt, and generation settings.
    prompt = build_solver_a2_prompt(episode)


    # ================================================================
    # BASELINE: NO ACTIVATION INTERVENTION
    # ================================================================

    baseline_answer = generate_answer(
        solver=solver,
        prompt=prompt,
        intervention=None,
    )

    baseline_score = score_answer(
        critic=critic,
        episode=episode,
        answer=baseline_answer,
    )

    results.append({
        "episode_id": episode.id,
        "experiment": "baseline",
        "representation": "none",
        "condition": "no_intervention",
        "random_draw": None,
        "answer": baseline_answer,
        "score": baseline_score,
    })


    # ================================================================
    # RAW-ACTIVATION CONTROL 1:
    # RANDOM RAW-FEATURE ABLATIONS
    # ================================================================

    for random_draw in range(NUMBER_OF_RANDOM_DRAWS):

        random_raw_features = sample_matched_random_features(
            candidate_features=all_raw_features,
            number_to_select=len(targeted_raw_features),
            exclude=targeted_raw_features,
            random_seed=SEED + random_draw,
        )

        answer = generate_answer(
            solver=solver,
            prompt=prompt,
            intervention={
                "layer": LAYER,
                "representation": "raw_activations",
                "operation": "ablate",
                "features": random_raw_features,
            },
        )

        score = score_answer(
            critic=critic,
            episode=episode,
            answer=answer,
        )

        results.append({
            "episode_id": episode.id,
            "experiment": "raw_activation_control",
            "representation": "raw_activations",
            "condition": "random_ablation",
            "random_draw": random_draw,
            "ablated_features": random_raw_features,
            "answer": answer,
            "score": score,
        })


    # ================================================================
    # RAW-ACTIVATION CONTROL 2:
    # TARGETED RAW-FEATURE ABLATION
    # ================================================================

    answer = generate_answer(
        solver=solver,
        prompt=prompt,
        intervention={
            "layer": LAYER,
            "representation": "raw_activations",
            "operation": "ablate",
            "features": targeted_raw_features,
        },
    )

    score = score_answer(
        critic=critic,
        episode=episode,
        answer=answer,
    )

    results.append({
        "episode_id": episode.id,
        "experiment": "raw_activation_control",
        "representation": "raw_activations",
        "condition": "targeted_ablation",
        "random_draw": None,
        "ablated_features": targeted_raw_features,
        "answer": answer,
        "score": score,
    })


    # ================================================================
    # SAE EXPERIMENTS
    # ================================================================

    for sae_name, sae in saes.items():

        targeted_features = targeted_sae_features[sae_name]


        # ------------------------------------------------------------
        # SAE CONTROL 1: SAE PRESENT, NO FEATURES ABLATED
        # ------------------------------------------------------------

        answer = generate_answer(
            solver=solver,
            prompt=prompt,
            intervention={
                "layer": LAYER,
                "sae": sae,
                "operation": "reconstruct_only",
                "features": [],
            },
        )

        score = score_answer(
            critic=critic,
            episode=episode,
            answer=answer,
        )

        results.append({
            "episode_id": episode.id,
            "experiment": sae_name,
            "representation": "sae",
            "condition": "no_ablation",
            "random_draw": None,
            "ablated_features": [],
            "answer": answer,
            "score": score,
        })


        # ------------------------------------------------------------
        # SAE CONTROL 2: RANDOM SPARSE FEATURES ABLATED
        # ------------------------------------------------------------

        for random_draw in range(NUMBER_OF_RANDOM_DRAWS):

            random_features = sample_matched_random_features(
                candidate_features=all_features_for(sae),
                number_to_select=len(targeted_features),
                exclude=targeted_features,
                random_seed=SEED + random_draw,
            )

            answer = generate_answer(
                solver=solver,
                prompt=prompt,
                intervention={
                    "layer": LAYER,
                    "sae": sae,
                    "operation": "ablate",
                    "features": random_features,
                },
            )

            score = score_answer(
                critic=critic,
                episode=episode,
                answer=answer,
            )

            results.append({
                "episode_id": episode.id,
                "experiment": sae_name,
                "representation": "sae",
                "condition": "random_ablation",
                "random_draw": random_draw,
                "ablated_features": random_features,
                "answer": answer,
                "score": score,
            })


        # ------------------------------------------------------------
        # SAE EXPERIMENT: TARGETED SPARSE FEATURES ABLATED
        # ------------------------------------------------------------

        answer = generate_answer(
            solver=solver,
            prompt=prompt,
            intervention={
                "layer": LAYER,
                "sae": sae,
                "operation": "ablate",
                "features": targeted_features,
            },
        )

        score = score_answer(
            critic=critic,
            episode=episode,
            answer=answer,
        )

        results.append({
            "episode_id": episode.id,
            "experiment": sae_name,
            "representation": "sae",
            "condition": "targeted_ablation",
            "random_draw": None,
            "ablated_features": targeted_features,
            "answer": answer,
            "score": score,
        })
        
        
    if config['command'] == "merge":
        continue


# ---------------------------------------------------------------------
# SAVE RAW RESULTS
# ---------------------------------------------------------------------

write_csv(results, OUTPUT_PATH)


# ---------------------------------------------------------------------
# SUMMARIZE THE EXPERIMENT
# ---------------------------------------------------------------------

summary = summarize_results(
    results,
    group_by=[
        "experiment",
        "condition",
    ],
    compare_each_ablation_to_its_control=True,
)

write_csv(summary, "results/intervention/summary.csv")
print(summary)