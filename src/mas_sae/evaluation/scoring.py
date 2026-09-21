from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


_ARTICLES = re.compile(
    r"\b(?:a|an|the)\b",
    flags=re.IGNORECASE,
)


def normalize_answer(value: object) -> str:
    text = str(value).casefold()

    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )

    text = _ARTICLES.sub(" ", text)

    return " ".join(text.split())


def answer_matches(
    candidate: str | None,
    gold: str,
    aliases: Iterable[str] = (),
) -> bool:
    if candidate is None:
        return False

    if isinstance(aliases, str):
        aliases = (aliases,)

    targets = {normalize_answer(gold)}

    targets.update(
        normalize_answer(alias)
        for alias in aliases
    )

    targets.discard("")

    normalized_candidate = normalize_answer(candidate)

    return (
        bool(normalized_candidate)
        and normalized_candidate in targets
    )


def interaction_labels(
    attempt1: str,
    attempt2: str,
    advocated_answer: str | None,
    gold: str,
    aliases: Iterable[str] = (),
) -> dict[str, bool | None]:
    """Compute behavioral labels for one Critic-Solver episode."""

    if advocated_answer is None:
        solver_accepted_feedback = None
        critic_feedback_correct = None

    else:
        critic_feedback_correct = answer_matches(
            advocated_answer,
            gold,
            aliases,
        )

        if critic_feedback_correct:
            # If the critic advocated a correct answer, any accepted
            # gold alias counts as accepting that feedback.
            solver_accepted_feedback = answer_matches(
                attempt2,
                gold,
                aliases,
            )
        else:
            # If the critic advocated an incorrect answer, accepting
            # the feedback means matching that specific advocated answer.
            solver_accepted_feedback = answer_matches(
                attempt2,
                advocated_answer,
            )

    return {
        "solver_accepted_feedback": solver_accepted_feedback,
        "solver_changed_answer": not answer_matches(
            attempt2,
            attempt1,
        ),
        "critic_feedback_correct": critic_feedback_correct,
        "solver_attempt_2_correct": answer_matches(
            attempt2,
            gold,
            aliases,
        ),
    }
