from __future__ import annotations

import logging
from typing import Any, TypedDict

import torch

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.data.musique import MuSiQueExample
from mas_sae.experiments.pipeline import run_question


logger = logging.getLogger(__name__)


class CollectionResult(TypedDict):
    """Activation rows and metadata produced by one collection run."""

    records: list[dict[str, Any]]
    attempt1_by_site: dict[str, list[torch.Tensor]]
    attempt2_by_site: dict[str, list[torch.Tensor]]


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
) -> CollectionResult:
    """Collect paired Solver-Critic episodes and activation-row mappings.

    Each question is passed once to ``run_question``. That call generates one
    Solver Attempt 1 and reuses it across natural, controlled-correct, and
    controlled-incorrect critic conditions. The three resulting records share
    one Attempt 1 activation index, while each Attempt 2 has its own index.

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

    Returns
    -------
    CollectionResult
        Episode records plus Attempt 1 and Attempt 2 tensors grouped by site.

    Raises
    ------
    ValueError
        If no examples or no candidate activation sites are provided.
    """
    if not examples:
        raise ValueError("examples must not be empty.")
    if not candidate_sites:
        raise ValueError("candidate_sites must not be empty.")

    attempt1_by_site: dict[str, list[torch.Tensor]] = {
        site: [] for site in candidate_sites
    }
    attempt2_by_site: dict[str, list[torch.Tensor]] = {
        site: [] for site in candidate_sites
    }
    records: list[dict[str, Any]] = []

    for question_index, example in enumerate(examples):
        seed = base_seed + question_index
        logger.info(
            "Collecting question %d/%d (%s)",
            question_index + 1,
            len(examples),
            example["id"],
        )

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
        )

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
            record["attempt1_activation_index"] = attempt1_index
            record["attempt2_activation_index"] = attempt2_index
            records.append(record)

    return {
        "records": records,
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
