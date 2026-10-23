from enum import Enum

from mas_sae.agents.critic import CriticCondition
from mas_sae.agents.solver import Solver
from mas_sae.data.musique import assign_experiment_splits


OMITTED = object()


def resolve_conditions(values=OMITTED):
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


def resolve_presentations(values=OMITTED):
    """Omission keeps the original feedback prompt; explicit lists must be nonempty and unique."""
    if values is OMITTED:
        return (A2Presentation.FEEDBACK,)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("a2_presentations must be a nonempty list")
    presentations = tuple(A2Presentation(value) for value in values)
    if len(set(presentations)) != len(presentations):
        raise ValueError("a2_presentations must be unique")
    return presentations


def assign_candidate_orders(question_ids, *, seed):
    """Seeded candidate order per question, balanced within each hop group.

    Decided from the question ids and the seed alone, before any generation.
    """
    return assign_experiment_splits(
        question_ids,
        proportions={order: 0.5 for order in Solver.CANDIDATE_ORDERS},
        seed=seed,
        supported_splits=Solver.CANDIDATE_ORDERS,
    )
