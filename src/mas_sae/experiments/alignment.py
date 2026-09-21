from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mas_sae.agents.critic import CriticCondition
from mas_sae.data.activation_store import ActivationStore


CONDITIONS = {
    condition.value
    for condition in CriticCondition
}


def validate_pilot_alignment(
    store_root: str | Path,
    interactions: list[dict[str, Any]],
    layers: list[int],
    num_questions: int,
    hidden_dim: int = 2560,
) -> dict[str, Any]:
    """Validate pilot tensors against their interaction metadata."""

    num_episodes = num_questions * 3

    if len(interactions) != num_episodes:
        raise ValueError(
            f"Expected {num_episodes} interactions, "
            f"found {len(interactions)}."
        )

    for row_index, row in enumerate(interactions):
        if row["attempt2_activation_index"] != row_index:
            raise ValueError(
                "Attempt 2 activation index does not match "
                "its interaction row."
            )

    question_groups: dict[str, list[dict[str, Any]]] = {}

    for row in interactions:
        question_groups.setdefault(
            str(row["question_id"]),
            [],
        ).append(row)

    if len(question_groups) != num_questions:
        raise ValueError("Unexpected number of unique questions.")

    seen_attempt1_indices = set()

    for rows in question_groups.values():
        if len(rows) != 3:
            raise ValueError("Each question must have 3 episodes.")

        if {row["critic_condition"] for row in rows} != CONDITIONS:
            raise ValueError("Critic conditions are incomplete.")

        if len({
            row["attempt1_activation_index"]
            for row in rows
        }) != 1:
            raise ValueError(
                "Question does not reuse one Attempt 1 index."
            )

        attempt1_index = rows[0]["attempt1_activation_index"]

        if attempt1_index in seen_attempt1_indices:
            raise ValueError(
                "Attempt 1 index is shared across questions."
            )

        seen_attempt1_indices.add(attempt1_index)

        for row in rows:
            if row["attempt2_activation_index"] // 3 != attempt1_index:
                raise ValueError(
                    "Attempt 1 and Attempt 2 indices are misaligned."
                )

        if len({
            row["solver_attempt_1"]
            for row in rows
        }) != 1:
            raise ValueError(
                "Question does not reuse one Attempt 1 answer."
            )

    if seen_attempt1_indices != set(range(num_questions)):
        raise ValueError(
            "Attempt 1 indices do not cover all question rows."
        )

    layer_report = {}

    for layer in layers:
        store = ActivationStore(
            Path(store_root) / f"layer_{layer:02d}"
        )

        attempt1 = store.load_activations(
            "pilot_attempt1"
        )
        attempt2 = store.load_activations(
            "pilot_attempt2"
        )

        if tuple(attempt1.shape) != (
            num_questions,
            hidden_dim,
        ):
            raise ValueError(
                f"Layer {layer} Attempt 1 shape is invalid."
            )

        if tuple(attempt2.shape) != (
            num_episodes,
            hidden_dim,
        ):
            raise ValueError(
                f"Layer {layer} Attempt 2 shape is invalid."
            )

        if not torch.isfinite(attempt1).all():
            raise ValueError(
                f"Layer {layer} Attempt 1 has non-finite values."
            )

        if not torch.isfinite(attempt2).all():
            raise ValueError(
                f"Layer {layer} Attempt 2 has non-finite values."
            )

        layer_report[str(layer)] = {
            "attempt1_shape": list(attempt1.shape),
            "attempt2_shape": list(attempt2.shape),
        }

    labels = Counter(
        str(row["solver_accepted_feedback"])
        for row in interactions
    )

    return {
        "status": "pass",
        "questions": num_questions,
        "episodes": num_episodes,
        "accepted": labels["True"],
        "rejected": labels["False"],
        "noncommittal": labels["None"],
        "layers": layer_report,
    }
