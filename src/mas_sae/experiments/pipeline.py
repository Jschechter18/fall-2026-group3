from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mas_sae.activations.capture import MultiSiteCapture
from mas_sae.agents.critic import Critic, CriticCondition
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.evaluation.scoring import (
    answer_matches,
    interaction_labels,
)
from mas_sae.experiments.reproducibility import seed_everything


def choose_incorrect_answer(
    *,
    paragraphs: list[dict[str, Any]],
    gold: str,
    aliases: Iterable[str] = (),
) -> str:
    """Choose a deterministic wrong answer for controlled feedback."""

    aliases = tuple(aliases)

    # Prefer a title from a non-supporting paragraph.
    candidates = [
        paragraph
        for paragraph in paragraphs
        if not paragraph.get("is_supporting", False)
    ]

    candidates.extend(
        paragraph
        for paragraph in paragraphs
        if paragraph.get("is_supporting", False)
    )

    for paragraph in candidates:
        candidate = str(
            paragraph.get("title", "")
        ).strip()

        if (
            candidate
            and not answer_matches(
                candidate,
                gold,
                aliases,
            )
        ):
            return candidate

    fallback = f"Not {gold}"

    if answer_matches(
        fallback,
        gold,
        aliases,
    ):
        raise ValueError(
            "Could not construct a controlled incorrect answer."
        )

    return fallback


def run_question(
    *,
    question_id: str,
    question: str,
    paragraphs: list[dict[str, Any]],
    gold: str,
    aliases: Iterable[str],
    model: Any,
    solver: Solver,
    critic: Critic,
    validator: Validator,
    candidate_sites: list[str],
    seed: int,
) -> dict[str, Any]:
    """
    Run one complete paired Solver-Critic experiment.

    Solver Attempt 1 is generated exactly once and reused across
    Natural, Controlled Correct, and Controlled Incorrect feedback.
    """

    aliases = tuple(aliases)

    seed_everything(seed)

    # ---------------------------------------------------------
    # Attempt 1: generate exactly once.
    # ---------------------------------------------------------

    with MultiSiteCapture(
        model,
        candidate_sites,
    ) as capture:
        attempt1 = solver.solve(
            question,
            paragraphs,
        )

    attempt1_activations = capture.activations

    attempt1_correct = answer_matches(
        attempt1,
        gold,
        aliases,
    )

    incorrect_answer = choose_incorrect_answer(
        paragraphs=paragraphs,
        gold=gold,
        aliases=aliases,
    )

    episodes: list[dict[str, Any]] = []

    conditions = (
        CriticCondition.NATURAL,
        CriticCondition.CONTROLLED_CORRECT,
        CriticCondition.CONTROLLED_INCORRECT,
    )

    # ---------------------------------------------------------
    # All three conditions branch from SAME Attempt 1.
    # ---------------------------------------------------------

    for condition in conditions:
        if condition is CriticCondition.NATURAL:
            target_answer = None

        elif condition is CriticCondition.CONTROLLED_CORRECT:
            target_answer = gold

        else:
            target_answer = incorrect_answer

        feedback = critic.critique(
            question=question,
            paragraphs=paragraphs,
            solver_answer=attempt1,
            condition=condition,
            advocated_answer=target_answer,
        )

        feedback_text = feedback.to_solver_text()

        # -----------------------------------------------------
        # Attempt 2 after Critic feedback.
        # -----------------------------------------------------

        with MultiSiteCapture(
            model,
            candidate_sites,
        ) as capture:
            attempt2 = solver.revise(
                question,
                paragraphs,
                attempt1,
                feedback_text,
            )

        attempt2_activations = capture.activations

        labels = interaction_labels(
            attempt1=attempt1,
            attempt2=attempt2,
            advocated_answer=feedback.advocated_answer,
            gold=gold,
            aliases=aliases,
        )

        # Validator judges final system output.
        validator_verdict = validator.validate(
            question,
            gold,
            aliases,
            attempt2,
        )

        record = {
            "episode_id": (
                f"{question_id}__{condition.value}"
            ),
            "question_id": question_id,
            "question": question,
            "ground_truth": gold,
            "answer_aliases": list(aliases),

            "solver_attempt_1": attempt1,
            "solver_attempt_1_correct": attempt1_correct,

            "critic_condition": condition.value,
            "critic_verdict": feedback.verdict,
            "critic_advocated_answer": (
                feedback.advocated_answer
            ),
            "critic_feedback": feedback_text,
            "critic_raw_output": feedback.raw_output,
            "critic_noncommittal": feedback.noncommittal,

            "solver_attempt_2": attempt2,

            **labels,

            "validator_final_correct": (
                validator_verdict.is_correct
            ),
            "validator_raw_output": (
                validator_verdict.raw_output
            ),

            "seed": seed,
        }

        episodes.append(
            {
                "record": record,
                "attempt2_activations": (
                    attempt2_activations
                ),
            }
        )

    return {
        "attempt1": attempt1,
        "attempt1_activations": (
            attempt1_activations
        ),
        "episodes": episodes,
    }
