from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mas_sae.data.activation_store import ActivationStore


def _validate_attempt2_indices(
    interactions: list[dict[str, Any]],
) -> None:
    """Verify Attempt 2 indices uniquely cover all activation rows."""
    indices = sorted(
        int(row["attempt2_activation_index"])
        for row in interactions
    )

    if indices != list(range(len(interactions))):
        raise ValueError(
            "Attempt 2 activation indices do not uniquely cover "
            "all activation rows."
        )


def _group_by_question(
    interactions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group interaction records by question ID."""
    groups: dict[str, list[dict[str, Any]]] = {}

    for row in interactions:
        groups.setdefault(
            str(row["question_id"]),
            [],
        ).append(row)

    return groups


def _validate_question_groups(
    question_groups: dict[str, list[dict[str, Any]]],
    expected_conditions: set[str],
    expected_num_questions: int | None,
) -> int:
    """Validate question-level Attempt 1 reuse and condition coverage."""
    num_questions = len(question_groups)

    if (
        expected_num_questions is not None
        and num_questions != expected_num_questions
    ):
        raise ValueError(
            f"Expected {expected_num_questions} questions, "
            f"found {num_questions}."
        )

    if not expected_conditions:
        raise ValueError("Expected conditions cannot be empty.")

    seen_attempt1_indices: set[int] = set()

    for rows in question_groups.values():
        if len(rows) != len(expected_conditions):
            raise ValueError(
                "Unexpected number of interaction rows for a question."
            )

        actual_conditions = {
            str(row["critic_condition"])
            for row in rows
        }

        if actual_conditions != expected_conditions:
            raise ValueError(
                "Critic conditions are incomplete or unexpected."
            )

        attempt1_indices = {
            int(row["attempt1_activation_index"])
            for row in rows
        }

        if len(attempt1_indices) != 1:
            raise ValueError(
                "Question does not reuse one Attempt 1 index."
            )

        attempt1_index = next(iter(attempt1_indices))

        if attempt1_index in seen_attempt1_indices:
            raise ValueError(
                "Attempt 1 index is shared across questions."
            )

        seen_attempt1_indices.add(attempt1_index)

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

    return num_questions


def _validate_layer_activations(
    store_root: str | Path,
    layers: list[int],
    attempt1_split: str,
    attempt2_split: str,
    num_questions: int,
    num_episodes: int,
    hidden_dim: int,
) -> dict[str, dict[str, list[int]]]:
    """Load and validate activation tensors for every requested layer."""
    layer_report: dict[str, dict[str, list[int]]] = {}

    for layer in layers:
        store = ActivationStore(
            Path(store_root) / f"layer_{layer:02d}"
        )

        attempt1 = store.load_activations(attempt1_split)
        attempt2 = store.load_activations(attempt2_split)

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

    return layer_report


def _summarize_labels(
    interactions: list[dict[str, Any]],
) -> dict[str, int]:
    """Summarize binary and noncommittal feedback-acceptance labels."""
    labels = Counter(
        str(row["solver_accepted_feedback"])
        for row in interactions
    )

    return {
        "accepted": labels["True"],
        "rejected": labels["False"],
        "noncommittal": labels["None"],
    }


def validate_activation_alignment(
    store_root: str | Path,
    interactions: list[dict[str, Any]],
    layers: list[int],
    attempt1_split: str,
    attempt2_split: str,
    expected_conditions: set[str],
    hidden_dim: int,
    expected_num_questions: int | None = None,
) -> dict[str, Any]:
    """Validate interaction metadata against saved activation tensors."""
    if not interactions:
        raise ValueError("No interactions were provided.")

    _validate_attempt2_indices(interactions)

    question_groups = _group_by_question(interactions)

    num_questions = _validate_question_groups(
        question_groups,
        expected_conditions,
        expected_num_questions,
    )

    num_episodes = len(interactions)

    layer_report = _validate_layer_activations(
        store_root=store_root,
        layers=layers,
        attempt1_split=attempt1_split,
        attempt2_split=attempt2_split,
        num_questions=num_questions,
        num_episodes=num_episodes,
        hidden_dim=hidden_dim,
    )

    label_summary = _summarize_labels(interactions)

    return {
        "status": "pass",
        "questions": num_questions,
        "episodes": num_episodes,
        **label_summary,
        "layers": layer_report,
    }
