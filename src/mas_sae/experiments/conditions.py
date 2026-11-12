"""Which Critic conditions and Attempt 2 presentations a collection run uses.

``OMITTED`` marks a setting absent from the config; omission reproduces the original V1 run
(all three conditions, feedback presentation) and its record schema. Candidate orders are
assigned per question from the ids and seed alone, before any generation.
"""

from enum import Enum

from mas_sae.agents.critic import CriticCondition
from mas_sae.agents.solver import Solver
from mas_sae.data.musique import assign_experiment_splits


OMITTED = object()


def resolve_conditions(values=OMITTED) -> tuple[CriticCondition, ...]:
    """Omission preserves V1; explicit lists must be nonempty and unique."""
    if values is OMITTED:
        return tuple(CriticCondition)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("active_conditions must be a nonempty list")
    conditions = tuple(CriticCondition(value) for value in values)
    if len(set(conditions)) != len(conditions):
        raise ValueError("active_conditions must be unique")
    return conditions


class A2Presentation(str, Enum):
    """How the Solver's second attempt is prompted."""

    # Solver.REVISE_PROMPT_V1 with the Critic's full feedback text.
    FEEDBACK = "feedback"
    # Solver.CANDIDATE_PROMPT: A1 and the Critic's recommendation as two
    # labeled candidates, in the question's assigned order.
    CANDIDATES = "candidates"


def resolve_presentations(values=OMITTED) -> tuple[A2Presentation, ...]:
    """Omission keeps the original feedback prompt; explicit lists must be nonempty and unique."""
    if values is OMITTED:
        return (A2Presentation.FEEDBACK,)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("a2_presentations must be a nonempty list")
    presentations = tuple(A2Presentation(value) for value in values)
    if len(set(presentations)) != len(presentations):
        raise ValueError("a2_presentations must be unique")
    return presentations


def assign_candidate_orders(question_ids: list[str], *, seed: int,
                            hop_groups: dict[str, str] | None = None) -> dict[str, str]:
    """Seeded candidate order per question, balanced within each hop group.

    Decided from the question ids and the seed alone, before any generation.
    Explicit ``hop_groups`` supports IDs that do not encode a MuSiQue hop type.
    """
    return assign_experiment_splits(
        question_ids,
        proportions={order: 0.5 for order in Solver.CANDIDATE_ORDERS},
        seed=seed,
        supported_splits=Solver.CANDIDATE_ORDERS,
        hop_groups=hop_groups,
    )
