from unittest.mock import Mock

import pytest
import torch

from mas_sae.experiments import collection


SITES = [
    "model.language_model.layers.8",
    "model.language_model.layers.17",
]


def make_result(question_id: str):
    return {
        "attempt1": "first answer",
        "attempt1_activations": {
            site: torch.ones(1, 4)
            for site in SITES
        },
        "episodes": [
            {
                "record": {
                    "question_id": question_id,
                    "critic_condition": condition,
                    "solver_accepted_feedback": accepted,
                },
                "attempt2_activations": {
                    site: torch.ones(1, 4)
                    for site in SITES
                },
            }
            for condition, accepted in [
                ("natural", True),
                ("controlled_correct", True),
                ("controlled_incorrect", False),
            ]
        ],
    }


def test_collect_examples_preserves_indices_and_source_split(
    monkeypatch,
) -> None:
    mock_run_question = Mock(
        side_effect=[
            make_result("q1"),
            make_result("q2"),
        ]
    )
    monkeypatch.setattr(
        collection,
        "run_question",
        mock_run_question,
    )

    examples = [
        {
            "id": "q1",
            "question": "Question 1",
            "paragraphs": [],
            "answer": "A1",
            "answer_aliases": [],
        },
        {
            "id": "q2",
            "question": "Question 2",
            "paragraphs": [],
            "answer": "A2",
            "answer_aliases": [],
        },
    ]

    result = collection.collect_examples(
        examples=examples,
        source_split="train",
        model=object(),
        solver=Mock(),
        critic=Mock(),
        validator=Mock(),
        candidate_sites=SITES,
        base_seed=42,
    )

    records = result["records"]

    assert len(records) == 6
    assert {
        row["source_split"]
        for row in records
    } == {"train"}

    assert [
        row["attempt1_activation_index"]
        for row in records
    ] == [0, 0, 0, 1, 1, 1]

    assert [
        row["attempt2_activation_index"]
        for row in records
    ] == list(range(6))

    assert [
        call.kwargs["seed"]
        for call in mock_run_question.call_args_list
    ] == [42, 43]


def test_stack_site_activations() -> None:
    activations = {
        site: [
            torch.ones(1, 4),
            torch.zeros(1, 4),
        ]
        for site in SITES
    }

    stacked = collection.stack_site_activations(
        activations
    )

    assert stacked[SITES[0]].shape == (2, 4)
    assert stacked[SITES[1]].shape == (2, 4)


def test_collect_examples_rejects_empty_examples() -> None:
    with pytest.raises(
        ValueError,
        match="examples must not be empty",
    ):
        collection.collect_examples(
            examples=[],
            source_split="validation",
            model=object(),
            solver=Mock(),
            critic=Mock(),
            validator=Mock(),
            candidate_sites=SITES,
            base_seed=42,
        )
