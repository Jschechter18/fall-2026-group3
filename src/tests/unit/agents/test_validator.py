from unittest.mock import Mock

from mas_sae.agents.validator import Validator


def make_validator(output: str):
    model = Mock()
    model.device = "cpu"

    processor = Mock()

    validator = Validator(model, processor)
    validator._generate = Mock(return_value=output)

    return validator


def test_validator_yes_returns_true():
    validator = make_validator("YES")

    verdict = validator.validate(
        "What is the capital of France?",
        "Paris",
        ["Paris, France"],
        "Paris",
    )

    assert verdict.is_correct is True


def test_validator_no_returns_false():
    validator = make_validator("NO")

    verdict = validator.validate(
        "What is the capital of France?",
        "Paris",
        [],
        "London",
    )

    assert verdict.is_correct is False


def test_validator_unexpected_output_returns_none():
    validator = make_validator("UNCERTAIN")

    verdict = validator.validate(
        "Question?",
        "Gold",
        [],
        "Candidate",
    )

    assert verdict.is_correct is None


def test_validator_prompt_contains_gold_aliases_and_candidate():
    validator = make_validator("YES")

    prompt = validator.build_prompt(
        question="Question?",
        gold_answer="United States",
        aliases=["US", "U.S."],
        candidate="U.S.",
    )

    assert "Gold answer: United States" in prompt
    assert "Accepted aliases: US, U.S." in prompt
    assert "Candidate answer: U.S." in prompt
