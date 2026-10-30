from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import torch

from mas_sae.agents.critic import Critic, CriticBlindAnswerError
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.data.musique import (
    MuSiQueExample,
    assign_experiment_splits,
    describe_sampled_questions,
    load_musique_examples,
    load_musique_examples_by_id,
    sample_musique_examples,
)
from mas_sae.experiments.controlled_targets import ControlledTargetError
from mas_sae.experiments.pipeline import CriticRecommendationError, run_question
from mas_sae.experiments.conditions import (
    OMITTED,
    A2Presentation,
    assign_candidate_orders,
    resolve_conditions,
    resolve_presentations,
)


logger = logging.getLogger(__name__)


class CollectionResult(TypedDict):
    """Activation rows and metadata produced by one collection run."""

    records: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    attempt1_by_site: dict[str, list[torch.Tensor]]
    attempt2_by_site: dict[str, list[torch.Tensor]]


class QuestionSelection(TypedDict):
    """Questions chosen for one run plus their split and manifest."""

    examples: list[MuSiQueExample]
    experiment_splits: dict[str, str] | None
    sampled_questions: list[dict[str, Any]] | None
    candidate_orders: dict[str, str] | None


def select_questions(
    dataset_config: dict[str, Any],
    *,
    default_seed: int,
    a2_presentations=OMITTED,
) -> QuestionSelection:
    """Select questions and assign experiment splits from ``config.dataset``.

    Without a ``sampling`` section, or with ``strategy: first_n``, this is
    the original behaviour: the first N answerable questions in dataset
    order and no experiment split. Otherwise a seeded (optionally
    hop-stratified) sample is drawn, and when ``experiment_split`` is
    configured each question id is assigned exactly one split.
    ``strategy: listed`` collects exactly the questions in
    ``sampling.question_ids_file`` (a JSON list), in that order.
    When ``a2_presentations`` includes candidates, every question is given
    a candidate order before anything is generated, and it is recorded in
    the manifest.
    ``sampled_questions`` is the ordered reproducibility manifest, or
    ``None`` on the pure dataset-order path.
    Seeds default to ``default_seed`` (``collection.seed``).
    """
    source_split = dataset_config["source_split"]
    num_questions = dataset_config["num_questions"]
    sampling = dataset_config.get("sampling") or {}
    strategy = sampling.get("strategy", "first_n")
    split_config = dataset_config.get("experiment_split")
    revision = dataset_config.get("revision")
    local_input = {"path": dataset_config["path"]} if "path" in dataset_config else {}

    if strategy == "first_n":
        examples = load_musique_examples(
            source_split=source_split,
            num_questions=num_questions,
            revision=revision,
            **local_input,
        )
    elif strategy == "listed":
        question_ids = json.loads(
            Path(sampling["question_ids_file"]).read_text(encoding="utf-8")
        )
        if len(question_ids) != num_questions:
            raise ValueError(
                f"{sampling['question_ids_file']} lists {len(question_ids)} "
                f"questions but dataset.num_questions is {num_questions}."
            )
        examples = load_musique_examples_by_id(
            source_split, question_ids, revision=revision, **local_input
        )
    else:
        examples = sample_musique_examples(
            source_split=source_split,
            num_questions=num_questions,
            seed=sampling.get("seed", default_seed),
            hop_proportions=(
                sampling.get("hop_proportions")
                if strategy == "stratified"
                else None
            ),
            revision=revision,
            **local_input,
        )

    experiment_splits = None
    if split_config is not None:
        experiment_splits = assign_experiment_splits(
            [str(example["id"]) for example in examples],
            proportions=split_config["proportions"],
            seed=split_config.get("seed", default_seed),
        )

    candidate_orders = None
    if A2Presentation.CANDIDATES in resolve_presentations(a2_presentations):
        candidate_orders = assign_candidate_orders(
            [str(example["id"]) for example in examples],
            seed=default_seed,
        )

    if strategy == "first_n" and split_config is None and candidate_orders is None:
        sampled_questions = None
    else:
        sampled_questions = describe_sampled_questions(
            examples, experiment_splits, candidate_orders
        )

    return {
        "examples": examples,
        "experiment_splits": experiment_splits,
        "sampled_questions": sampled_questions,
        "candidate_orders": candidate_orders,
    }


def collect_examples(
    *,
    examples: list[MuSiQueExample],
    source_split: str,
    model: torch.nn.Module,
    solver: Solver,
    critic: Critic,
    validator: Validator,
    candidate_sites: list[str],
    base_seed: int,
    experiment_splits: dict[str, str] | None = None,
    type_checked_target: bool = False,
    active_conditions=OMITTED,
    question_offset: int = 0,
    a2_presentations=OMITTED,
    candidate_orders: dict[str, str] | None = None,
) -> CollectionResult:
    """Collect paired Solver-Critic episodes and activation-row mappings.

    Each question is passed once to ``run_question``. That call generates one
    Solver Attempt 1 and reuses it across natural, controlled-correct, and
    controlled-incorrect critic conditions. The three resulting records share
    one Attempt 1 activation index, while each Attempt 2 has its own index.
    When ``experiment_splits`` is given, all three records of a question also
    share that question's ``experiment_split``.

    A question whose treatment cannot be constructed (``run_question`` raises
    ``CriticBlindAnswerError`` or ``ControlledTargetError``) is skipped with
    a warning naming the question id, contributing no records and no
    activation rows, and collection continues with the next question. Any
    other exception still propagates.

    Parameters
    ----------
    examples
        Answerable MuSiQue examples to collect in the provided order.
    source_split
        Original MuSiQue source split, currently ``train`` or ``validation``.
    model
        Language model whose candidate modules are captured during generation.
    solver
        Solver agent used for the initial answer and revision.
    critic
        Critic agent used to generate each feedback condition.
    validator
        Secondary judge used to evaluate the final Solver answer.
    candidate_sites
        Fully qualified model-module names whose activations are captured.
    base_seed
        Seed for the first question. Question index is added deterministically.
    experiment_splits
        Optional mapping from question id to scientific experiment split
        (``discovery`` / ``validation`` / ``intervention``). Assigned at
        question level, so it is written unchanged to every episode of that
        question. When omitted, records carry no ``experiment_split`` field,
        which preserves the original record schema.
    type_checked_target
        Forwarded to ``run_question`` together with each example's MuSiQue
        ``question_decomposition``. When set, the controlled-incorrect
        target comes from the type-checked generator instead of the
        paragraph-title heuristic.
    question_offset
        Position of ``examples[0]`` in the full ordered question list. A
        chunk of a longer run passes its start position so seeds and
        exclusion ``question_index`` values match an uninterrupted run;
        activation indices stay local to this call.
    a2_presentations
        Forwarded to ``run_question``. Each presentation adds one episode
        and one Attempt 2 activation row per condition.
    candidate_orders
        Question id to candidate order, required for every question when
        the candidates presentation is on (``select_questions`` assigns it).
        A question whose Critic gives no recommendation is excluded with
        stage ``critic_recommendation``.

    Returns
    -------
    CollectionResult
        Episode records plus Attempt 1 and Attempt 2 tensors grouped by site.

    Raises
    ------
    ValueError
        If no examples or no candidate activation sites are provided, or if
        ``experiment_splits`` is given but lacks one of the question ids.
    """
    resolve_conditions(active_conditions)
    presentations = resolve_presentations(a2_presentations)
    if not examples:
        raise ValueError("examples must not be empty.")
    if not candidate_sites:
        raise ValueError("candidate_sites must not be empty.")
    if type(question_offset) is not int or question_offset < 0:
        raise ValueError("question_offset must be a non-negative integer.")

    if experiment_splits is not None:
        missing = [
            str(example["id"])
            for example in examples
            if str(example["id"]) not in experiment_splits
        ]
        if missing:
            raise ValueError(
                "experiment_splits is missing question ids: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
            )

    if A2Presentation.CANDIDATES in presentations:
        missing = [
            str(example["id"])
            for example in examples
            if str(example["id"]) not in (candidate_orders or {})
        ]
        if missing:
            raise ValueError(
                "candidate_orders is missing question ids: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
            )

    attempt1_by_site: dict[str, list[torch.Tensor]] = {
        site: [] for site in candidate_sites
    }
    attempt2_by_site: dict[str, list[torch.Tensor]] = {
        site: [] for site in candidate_sites
    }
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for question_index, example in enumerate(examples, start=question_offset):
        seed = base_seed + question_index
        logger.info(
            "Collecting question %d/%d (%s)",
            question_index - question_offset + 1,
            len(examples),
            example["id"],
        )

        try:
            result = run_question(
                question_id=str(example["id"]),
                question=example["question"],
                paragraphs=example["paragraphs"],
                gold=example["answer"],
                aliases=example.get("answer_aliases", []) or [],
                model=model,
                solver=solver,
                critic=critic,
                validator=validator,
                candidate_sites=candidate_sites,
                seed=seed,
                decomposition=example.get("question_decomposition") or [],
                type_checked_target=type_checked_target,
                active_conditions=active_conditions,
                a2_presentations=a2_presentations,
                candidate_order=(candidate_orders or {}).get(str(example["id"])),
            )
        except (CriticBlindAnswerError, ControlledTargetError,
                CriticRecommendationError) as error:
            logger.warning(
                "Skipping question %s: %s", example["id"], error
            )
            # A1 always exists here (it is generated first). The blind step
            # produced (unusable) critic output; whether the target step
            # called the critic is not known at this level. A missing
            # recommendation stops before that condition's Attempt 2.
            blind_failure = isinstance(error, CriticBlindAnswerError)
            no_recommendation = isinstance(error, CriticRecommendationError)
            exclusions.append({
                "question_id": str(example["id"]),
                "question_index": question_index, "seed": seed,
                "stage": ("critic_blind" if blind_failure
                          else "critic_recommendation" if no_recommendation
                          else "controlled_target"),
                "error_type": type(error).__name__, "reason": str(error),
                "a1_exists": True,
                "critic_exists": True if blind_failure or no_recommendation else None,
                "a2_exists": False,
            })
            continue

        attempt1_index = len(attempt1_by_site[candidate_sites[0]])

        for site in candidate_sites:
            attempt1_by_site[site].append(result["attempt1_activations"][site])

        for episode in result["episodes"]:
            attempt2_index = len(attempt2_by_site[candidate_sites[0]])

            for site in candidate_sites:
                attempt2_by_site[site].append(
                    episode["attempt2_activations"][site]
                )

            record = dict(episode["record"])
            record["source_split"] = source_split
            if experiment_splits is not None:
                record["experiment_split"] = experiment_splits[
                    str(example["id"])
                ]
            record["attempt1_activation_index"] = attempt1_index
            record["attempt2_activation_index"] = attempt2_index
            records.append(record)

    return {
        "records": records,
        "exclusions": exclusions,
        "attempt1_by_site": attempt1_by_site,
        "attempt2_by_site": attempt2_by_site,
    }


def stack_site_activations(
    activations_by_site: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack collected activation rows for each model site."""
    stacked: dict[str, torch.Tensor] = {}

    for site, tensors in activations_by_site.items():
        if not tensors:
            raise ValueError(f"No activations were collected for {site!r}.")
        stacked[site] = torch.cat(tensors, dim=0)

    return stacked
