"""End-to-end tests of ``run_question`` under the V1 and V2 protocols.

Agents are real ``Critic`` objects whose generation is faked, so the
prompts actually built by the pipeline are inspected; activation capture
is replaced by a stub.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mas_sae.agents.critic import Critic, CriticBlindAnswerError
from mas_sae.experiments import pipeline
from mas_sae.experiments.controlled_targets import ControlledTargetError


SITES = ["model.language_model.layers.8"]

QUESTION = "Who is the spouse of the performer of Green?"
GOLD = "Miquette Giraudy"
DECOMPOSITION = [
    {"question": "Green >> performer", "answer": "Steve Hillage"},
    {"question": "#1 >> spouse", "answer": "Miquette Giraudy"},
]
PARAGRAPHS = [
    {
        "idx": 0,
        "title": "Green (album)",
        "paragraph_text": "Green is a 1978 album by Steve Hillage.",
        "is_supporting": True,
    },
    {
        "idx": 1,
        "title": "Steve Hillage",
        "paragraph_text": "Hillage's partner is Miquette Giraudy.",
        "is_supporting": True,
    },
    {
        "idx": 2,
        "title": "Paris",
        "paragraph_text": "Paris is the capital of France.",
        "is_supporting": False,
    },
]

NATURAL_JSON = (
    '{"verdict": "agree", "advocated_answer": "Miquette Giraudy", '
    '"explanation": "Paragraph 1 names her."}'
)


class FakeCapture:
    def __init__(self, model, sites):
        self.activations = {site: torch.zeros(1, 4) for site in sites}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_generation(prompts, *, blind='{"answer": "Steve Hillage"}',
                    distractor='{"answer": "Daevid Allen"}'):
    """Route each critic prompt to a canned output and log the prompt."""

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        if "answer the question independently" in prompt:
            return blind
        if "propose one plausible but incorrect answer" in prompt:
            return distractor
        if "Your own answer:" in prompt or '"verdict": "agree" or' in prompt:
            return NATURAL_JSON
        return "Reviewer text."

    return generate


def make_agents(prompt_version, prompts, **generation):
    critic = Critic(Mock(device="cpu"), Mock(), prompt_version=prompt_version)
    critic._generate = fake_generation(prompts, **generation)

    solver = Mock()
    solver.solve.return_value = GOLD
    solver.revise.return_value = GOLD

    validator = Mock()
    validator.validate.return_value = SimpleNamespace(
        is_correct=True, raw_output="YES"
    )
    return solver, critic, validator


def run(monkeypatch, protocol_version, prompts, decomposition=DECOMPOSITION,
        **generation):
    monkeypatch.setattr(pipeline, "MultiSiteCapture", FakeCapture)
    solver, critic, validator = make_agents(
        protocol_version, prompts, **generation
    )

    return pipeline.run_question(
        question_id="2hop__1_2",
        question=QUESTION,
        paragraphs=PARAGRAPHS,
        gold=GOLD,
        aliases=[],
        model=object(),
        solver=solver,
        critic=critic,
        validator=validator,
        candidate_sites=SITES,
        seed=42,
        decomposition=decomposition,
        protocol_version=protocol_version,
    )


def records(result):
    return {ep["record"]["critic_condition"]: ep["record"] for ep in result["episodes"]}


def test_v2_blind_answer_is_formed_before_the_critic_sees_attempt1(
    monkeypatch,
):
    prompts: list[str] = []
    result = run(monkeypatch, "v2", prompts)

    blind_prompts = [
        index for index, prompt in enumerate(prompts)
        if "answer the question independently" in prompt
    ]
    prompts_with_attempt1 = [
        index for index, prompt in enumerate(prompts)
        if f"Proposed answer: {GOLD}" in prompt
    ]

    assert len(blind_prompts) == 1
    assert GOLD not in prompts[blind_prompts[0]].split("Question:")[1]
    assert all(blind_prompts[0] < index for index in prompts_with_attempt1)
    assert len(prompts_with_attempt1) == 3  # natural compare + 2 controlled

    natural = records(result)["natural"]
    assert natural["critic_blind_answer"] == "Steve Hillage"
    assert "Steve Hillage" not in natural["critic_feedback"]


def test_v2_records_carry_target_fields_per_condition(monkeypatch):
    prompts: list[str] = []
    by_condition = records(run(monkeypatch, "v2", prompts))

    natural = by_condition["natural"]
    correct = by_condition["controlled_correct"]
    incorrect = by_condition["controlled_incorrect"]

    assert natural["controlled_target_source"] is None
    assert natural["controlled_target_type_check"] is None

    assert correct["critic_blind_answer"] is None
    assert correct["controlled_target_source"] == "gold"
    assert correct["controlled_target_type_check"] is None
    assert correct["critic_advocated_answer"] == GOLD

    assert incorrect["critic_blind_answer"] is None
    assert incorrect["controlled_target_source"] == "hop_intermediate"
    assert incorrect["controlled_target_type_check"] == "pass"
    assert incorrect["critic_advocated_answer"] == "Steve Hillage"
    assert incorrect["critic_feedback_correct"] is False

    # hop-1 target found, so no distractor call was made
    assert not any("plausible but incorrect" in prompt for prompt in prompts)


def test_v2_controlled_prompts_present_target_as_own_conclusion(monkeypatch):
    prompts: list[str] = []
    run(monkeypatch, "v2", prompts)

    controlled = [prompt for prompt in prompts if prompt.endswith("Review:")]
    assert len(controlled) == 2
    for prompt in controlled:
        assert "concluded that the correct answer is:" in prompt
        assert "Answer to advocate" not in prompt
        assert "supplied" not in prompt.casefold()


def test_v2_uses_distractor_when_no_hop_candidate(monkeypatch):
    prompts: list[str] = []
    by_condition = records(run(monkeypatch, "v2", prompts, decomposition=[]))

    incorrect = by_condition["controlled_incorrect"]
    assert incorrect["controlled_target_source"] == "llm_distractor"
    # "Who ..." gives a known person subtype, imposed through the prompt
    assert incorrect["controlled_target_type_check"] == "pass"
    assert incorrect["critic_advocated_answer"] == "Daevid Allen"

    distractor_prompt = next(
        prompt for prompt in prompts if "plausible but incorrect" in prompt
    )
    assert "must be the name of a person" in distractor_prompt
    assert GOLD in distractor_prompt  # excluded strings listed


def test_v2_blind_failure_aborts_before_any_feedback(monkeypatch):
    prompts: list[str] = []

    with pytest.raises(CriticBlindAnswerError):
        run(monkeypatch, "v2", prompts, blind="I cannot tell.")

    assert len(prompts) == 1


def test_v2_unresolvable_target_aborts(monkeypatch):
    prompts: list[str] = []

    with pytest.raises(ControlledTargetError):
        run(
            monkeypatch,
            "v2",
            prompts,
            decomposition=[],
            distractor='{"answer": "Miquette Giraudy"}',
        )


def test_v1_path_is_unchanged(monkeypatch):
    prompts: list[str] = []
    result = run(monkeypatch, "v1", prompts)

    assert not any("independently" in prompt for prompt in prompts)
    assert not any("plausible but incorrect" in prompt for prompt in prompts)
    assert any("Answer to advocate:" in prompt for prompt in prompts)

    for record in records(result).values():
        assert "critic_blind_answer" not in record
        assert "controlled_target_source" not in record
        assert "controlled_target_type_check" not in record

    # V1 target: first non-supporting paragraph title
    assert records(result)["controlled_incorrect"]["critic_advocated_answer"] == (
        "Paris"
    )


def test_protocol_version_must_match_critic(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(pipeline, "MultiSiteCapture", FakeCapture)
    solver, critic, validator = make_agents("v1", prompts)

    with pytest.raises(ValueError, match="does not match"):
        pipeline.run_question(
            question_id="q",
            question=QUESTION,
            paragraphs=PARAGRAPHS,
            gold=GOLD,
            aliases=[],
            model=object(),
            solver=solver,
            critic=critic,
            validator=validator,
            candidate_sites=SITES,
            seed=42,
            protocol_version="v2",
        )


def test_unknown_protocol_version_rejected(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(pipeline, "MultiSiteCapture", FakeCapture)
    solver, critic, validator = make_agents("v1", prompts)

    with pytest.raises(ValueError, match="protocol_version"):
        pipeline.run_question(
            question_id="q",
            question=QUESTION,
            paragraphs=PARAGRAPHS,
            gold=GOLD,
            aliases=[],
            model=object(),
            solver=solver,
            critic=critic,
            validator=validator,
            candidate_sites=SITES,
            seed=42,
            protocol_version="v3",
        )
