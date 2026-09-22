from __future__ import annotations

import logging
from typing import Any

import torch

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.experiments.pipeline import run_question


logger = logging.getLogger(__name__)


def collect_examples(
    *,
    examples: list[dict[str, Any]],
    source_split: str,
    model: Any,
    solver: Solver,
    critic: Critic,
    validator: Validator,
    candidate_sites: list[str],
    base_seed: int,
) -> dict[str, Any]:
    """Run Solver-Critic collection for already-loaded examples."""
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
