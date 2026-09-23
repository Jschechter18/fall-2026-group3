from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from mas_sae.activations.capture import MultiSiteCapture
from mas_sae.agents.critic import (
    PROMPT_VERSIONS,
    Critic,
    CriticCondition,
    CriticFeedback,
)
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.evaluation.scoring import (
    answer_matches,
    interaction_labels,
)
from mas_sae.experiments.controlled_targets import (
    ControlledTarget,
    DistractorRequest,
    choose_controlled_incorrect_target,
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


def choose_incorrect_answer_v2(
    *,
    question_id: str,
    question: str,
    paragraphs: list[dict[str, Any]],
    gold: str,
    aliases: tuple[str, ...],
    attempt1: str,
    decomposition: Sequence[dict[str, Any]],
    critic: Critic,
) -> ControlledTarget:
    """V2 wrong target: hop answer, context span, LLM distractor, or raise."""

    def generate_distractor(request: DistractorRequest) -> str:
        return critic.propose_distractor(
            question=request.question,
            paragraphs=request.paragraphs,
            type_description=request.type_description,
            excluded=request.excluded,
        )

    return choose_controlled_incorrect_target(
        question_id=question_id,
        question=question,
        paragraphs=paragraphs,
        gold=gold,
        aliases=aliases,
        attempt1=attempt1,
        decomposition=decomposition,
        distractor_generator=generate_distractor,
    )


def _v2_record_fields(
    condition: CriticCondition,
    feedback: CriticFeedback,
    controlled_target: ControlledTarget | None,
) -> dict[str, Any]:
    """V2-only record fields for one episode (null where not applicable)."""
    if condition is CriticCondition.NATURAL:
        return {
            "critic_blind_answer": feedback.blind_answer,
            "controlled_target_source": None,
            "controlled_target_type_check": None,
        }

    if condition is CriticCondition.CONTROLLED_CORRECT:
        return {
            "critic_blind_answer": None,
            "controlled_target_source": "gold",
            "controlled_target_type_check": None,
        }

    if controlled_target is None:
        raise ValueError(
            "controlled_target is required for the controlled-incorrect "
            "condition."
        )

    return {
        "critic_blind_answer": None,
        "controlled_target_source": controlled_target.source.value,
        "controlled_target_type_check": controlled_target.type_check,
    }


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
    decomposition: Sequence[dict[str, Any]] = (),
    protocol_version: str = "v1",
) -> dict[str, Any]:
    """
    Run one complete paired Solver-Critic experiment.

    Solver Attempt 1 is generated exactly once and reused across
    Natural, Controlled Correct, and Controlled Incorrect feedback.

    ``protocol_version`` ``"v1"`` reproduces the original records exactly.
    ``"v2"`` adds the blind-then-compare natural critic, the type-checked
    controlled-incorrect target, and the fields ``critic_blind_answer``,
    ``controlled_target_source`` and ``controlled_target_type_check``.
    Under V2 a missing blind answer or an unresolvable target raises and
    aborts the run; whole-question exclusion is later QC work.
    """

    if protocol_version not in PROMPT_VERSIONS:
        raise ValueError(
            "protocol_version must be one of "
            f"{list(PROMPT_VERSIONS)}, got {protocol_version!r}."
        )

    if protocol_version != critic.prompt_version:
        raise ValueError(
            f"protocol_version {protocol_version!r} does not match "
            f"critic.prompt_version {critic.prompt_version!r}."
        )

    is_v2 = protocol_version == "v2"
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

    # ---------------------------------------------------------
    # V2: the critic's blind answer is formed once per question,
    # before any prompt that contains Attempt 1.
    # ---------------------------------------------------------

    blind_answer: str | None = None
    controlled_target: ControlledTarget | None = None

    if is_v2:
        blind_answer = critic.answer_blind(question, paragraphs)

        controlled_target = choose_incorrect_answer_v2(
            question_id=question_id,
            question=question,
            paragraphs=paragraphs,
            gold=gold,
            aliases=aliases,
            attempt1=attempt1,
            decomposition=decomposition,
            critic=critic,
        )
        incorrect_answer = controlled_target.answer
    else:
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
            blind_answer=(
                blind_answer
                if is_v2 and condition is CriticCondition.NATURAL
                else None
            ),
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

        if is_v2:
            record.update(
                _v2_record_fields(condition, feedback, controlled_target)
            )

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
