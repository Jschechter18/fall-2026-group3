from unittest.mock import Mock

import pytest

from mas_sae.agents.critic import (
    Critic,
    CriticCondition,
    CriticFeedback,
)


def make_critic():
    model = Mock()
    model.device = "cpu"

    processor = Mock()

    return Critic(model, processor)

def test_controlled_critic_requires_advocated_answer():
    critic = make_critic()

    with pytest.raises(
        ValueError,
        match="advocated_answer is required",
    ):
        critic.critique(
            question="What is the capital of France?",
            paragraphs=[
                {
                    "idx": 0,
                    "title": "France",
                    "paragraph_text": "Paris is the capital of France.",
                }
            ],
            solver_answer="Paris",
            condition=CriticCondition.CONTROLLED_CORRECT,
        )


def test_parse_natural_structured_json():
    raw = """
    ```json
    {
        "verdict": "disagree",
        "advocated_answer": "Paris",
        "explanation": "The paragraph states that Paris is the capital."
    }
    ```
    """

    feedback = Critic.parse_natural(raw)

    assert feedback.condition is CriticCondition.NATURAL
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.noncommittal is False


def test_parse_natural_invalid_output_is_noncommittal():
    feedback = Critic.parse_natural(
        "I am not sure about the answer."
    )

    assert feedback.verdict is None
    assert feedback.advocated_answer is None
    assert feedback.noncommittal is True


def test_noncommittal_feedback_does_not_claim_disagreement():
    feedback = CriticFeedback(
        condition=CriticCondition.NATURAL,
        verdict=None,
        advocated_answer=None,
        explanation="The evidence is ambiguous.",
        raw_output="The evidence is ambiguous.",
        noncommittal=True,
    )

    text = feedback.to_solver_text()

    assert "without a clear agree or disagree verdict" in text
    assert "The reviewer disagrees with your previous answer." not in text


def test_controlled_critic_uses_supplied_advocated_answer():
    critic = make_critic()
    critic._generate = Mock(
        return_value="Paris is supported by the provided paragraph."
    )

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        solver_answer="Lyon",
        condition=CriticCondition.CONTROLLED_CORRECT,
        advocated_answer="Paris",
    )

    assert (
        feedback.condition
        is CriticCondition.CONTROLLED_CORRECT
    )
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.noncommittal is False


def test_natural_critique_builds_prompt_and_parses_response():
    critic = make_critic()

    critic._generate = Mock(
        return_value=(
            '{"verdict": "disagree", '
            '"advocated_answer": "Paris", '
            '"explanation": "Paris is supported by the paragraph."}'
        )
    )

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        solver_answer="Lyon",
        condition=CriticCondition.NATURAL,
    )

    prompt = critic._generate.call_args.args[0]

    assert '{"verdict": "agree" or "disagree"' in prompt
    assert "Proposed answer: Lyon" in prompt
    assert prompt.endswith("JSON:")

    assert feedback.condition is CriticCondition.NATURAL
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.explanation == (
        "Paris is supported by the paragraph."
    )
    assert feedback.noncommittal is False

def test_controlled_incorrect_preserves_condition_and_advocated_answer():
    critic = make_critic()

    critic._generate = Mock(
        return_value="London should be preferred."
    )

    feedback = critic.critique(
        question="What is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        solver_answer="Paris",
        condition=CriticCondition.CONTROLLED_INCORRECT,
        advocated_answer="London",
    )

    assert (
        feedback.condition
        is CriticCondition.CONTROLLED_INCORRECT
    )
    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "London"
    assert feedback.noncommittal is False


def test_parse_natural_ignores_trailing_braced_text():
    raw = (
        '{"verdict": "disagree", '
        '"advocated_answer": "Paris", '
        '"explanation": "Paris is supported."}'
        ' trailing note {"ignored": "value"}'
    )

    feedback = Critic.parse_natural(raw)

    assert feedback.verdict == "disagree"
    assert feedback.advocated_answer == "Paris"
    assert feedback.explanation == "Paris is supported."
    assert feedback.noncommittal is False
