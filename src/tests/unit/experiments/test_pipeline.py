"""End-to-end tests of ``run_question`` with and without the optional
critic and target capabilities.

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

# All three optional capabilities on (what the collection script maps
# ``protocol_version: v2`` to).
BLIND_TYPED = dict(
    blind_then_compare=True,
    controlled_as_own_conclusion=True,
    type_checked_target=True,
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


def make_agents(prompts, *, blind_then_compare=False,
                controlled_as_own_conclusion=False, **generation):
    critic = Critic(
        Mock(device="cpu"),
        Mock(),
        blind_then_compare=blind_then_compare,
        controlled_as_own_conclusion=controlled_as_own_conclusion,
    )
    critic._generate = fake_generation(prompts, **generation)

    solver = Mock()
    solver.solve.return_value = GOLD
    solver.revise.return_value = GOLD

    validator = Mock()
    validator.validate.return_value = SimpleNamespace(
        is_correct=True, raw_output="YES"
    )
    return solver, critic, validator


def run(monkeypatch, prompts, *, blind_then_compare=False,
        controlled_as_own_conclusion=False, type_checked_target=False,
        decomposition=DECOMPOSITION, **generation):
    monkeypatch.setattr(pipeline, "MultiSiteCapture", FakeCapture)
    solver, critic, validator = make_agents(
        prompts,
        blind_then_compare=blind_then_compare,
        controlled_as_own_conclusion=controlled_as_own_conclusion,
        **generation,
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
        type_checked_target=type_checked_target,
    )


def records(result):
    return {ep["record"]["critic_condition"]: ep["record"] for ep in result["episodes"]}


def test_blind_answer_is_formed_before_the_critic_sees_attempt1(
    monkeypatch,
):
    prompts: list[str] = []
    result = run(monkeypatch, prompts, **BLIND_TYPED)

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


def test_records_carry_target_fields_per_condition(monkeypatch):
    prompts: list[str] = []
    by_condition = records(run(monkeypatch, prompts, **BLIND_TYPED))

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


def test_own_conclusion_controlled_prompts_present_target_as_own(monkeypatch):
    prompts: list[str] = []
    run(monkeypatch, prompts, **BLIND_TYPED)

    controlled = [prompt for prompt in prompts if prompt.endswith("Review:")]
    assert len(controlled) == 2
    for prompt in controlled:
        assert "concluded that the correct answer is:" in prompt
        assert "Answer to advocate" not in prompt
        assert "supplied" not in prompt.casefold()


def test_type_checked_target_uses_distractor_when_no_hop_candidate(
    monkeypatch,
):
    prompts: list[str] = []
    by_condition = records(
        run(monkeypatch, prompts, decomposition=[], **BLIND_TYPED)
    )

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


def test_blind_failure_aborts_before_any_feedback(monkeypatch):
    prompts: list[str] = []

    with pytest.raises(CriticBlindAnswerError):
        run(monkeypatch, prompts, blind="I cannot tell.", **BLIND_TYPED)

    assert len(prompts) == 1


def test_unresolvable_type_checked_target_aborts(monkeypatch):
    prompts: list[str] = []

    with pytest.raises(ControlledTargetError):
        run(
            monkeypatch,
            prompts,
            decomposition=[],
            distractor='{"answer": "Miquette Giraudy"}',
            **BLIND_TYPED,
        )


def test_capabilities_are_independent(monkeypatch):
    """Each flag adds only its own fields; the others stay absent."""
    prompts: list[str] = []
    result = run(monkeypatch, prompts, blind_then_compare=True)

    for record in records(result).values():
        assert "critic_blind_answer" in record
        assert "controlled_target_source" not in record
    assert records(result)["controlled_incorrect"]["critic_advocated_answer"] == (
        "Paris"
    )
    assert any("Answer to advocate:" in prompt for prompt in prompts)

    prompts = []
    result = run(monkeypatch, prompts, type_checked_target=True)

    assert not any("independently" in prompt for prompt in prompts)
    for record in records(result).values():
        assert "critic_blind_answer" not in record
        assert "controlled_target_source" in record
    assert records(result)["controlled_incorrect"]["critic_advocated_answer"] == (
        "Steve Hillage"
    )


def test_default_path_is_unchanged(monkeypatch):
    prompts: list[str] = []
    result = run(monkeypatch, prompts)

    assert not any("independently" in prompt for prompt in prompts)
    assert not any("plausible but incorrect" in prompt for prompt in prompts)
    assert any("Answer to advocate:" in prompt for prompt in prompts)

    for record in records(result).values():
        assert "critic_blind_answer" not in record
        assert "controlled_target_source" not in record
        assert "controlled_target_type_check" not in record

    # original target: first non-supporting paragraph title
    assert records(result)["controlled_incorrect"]["critic_advocated_answer"] == (
        "Paris"
    )


def run_presentations(monkeypatch, prompts, *, a2_presentations, candidate_order="critic_first"):
    captures = []

    class CountingCapture(FakeCapture):
        def __init__(self, model, sites):
            super().__init__(model, sites)
            captures.append(self)

    monkeypatch.setattr(pipeline, "MultiSiteCapture", CountingCapture)
    solver, critic, validator = make_agents(prompts)
    critic._generate = lambda prompt: (prompts.append(prompt), NATURAL_JSON)[1]
    solver.choose.return_value = "Steve Hillage"
    solver.first_token_id.side_effect = lambda text: {"Steve Hillage": 11, "Miquette Giraudy": 12}[text]
    solver.last_generated_token_ids = [12, 99]
    solver.solve.return_value = "Steve Hillage"
    result = pipeline.run_question(
        question_id="2hop__1_2", question=QUESTION, paragraphs=PARAGRAPHS, gold=GOLD,
        aliases=[], model=object(), solver=solver, critic=critic, validator=validator,
        candidate_sites=SITES, seed=42, active_conditions=["natural"],
        a2_presentations=a2_presentations, candidate_order=candidate_order,
    )
    return result, solver, captures


def test_each_presentation_is_its_own_episode_from_the_same_a1_and_feedback(monkeypatch):
    prompts: list[str] = []
    result, solver, captures = run_presentations(
        monkeypatch, prompts, a2_presentations=["feedback", "candidates"])

    feedback, candidates = (ep["record"] for ep in result["episodes"])
    assert [feedback["episode_id"], candidates["episode_id"]] == [
        "2hop__1_2__natural__feedback", "2hop__1_2__natural__candidates"]
    # one Critic review, one A1 capture and one capture per A2
    assert sum('"verdict": "agree" or' in p or "Your own answer:" in p for p in prompts) == 1
    assert len(captures) == 3
    assert len(result["episodes"]) == 2
    solver.revise.assert_called_once()
    solver.choose.assert_called_once_with(
        QUESTION, PARAGRAPHS, "Steve Hillage", "Miquette Giraudy", "critic_first")
    assert feedback["critic_feedback"] == candidates["critic_feedback"]

    assert feedback["a2_presentation"] == "feedback"
    assert feedback["candidate_order"] is None and feedback["decision_token_distinct"] is None
    assert candidates["a2_presentation"] == "candidates"
    assert candidates["solver_attempt_2"] == "Steve Hillage"
    assert candidates["candidate_order"] == "critic_first"
    assert candidates["candidate_first_token_ids"] == {"a1": 11, "critic": 12}
    assert candidates["decision_token_distinct"] is True
    assert candidates["solver_a2_first_token_id"] == 12


def test_shared_first_token_is_flagged():
    solver = Mock(last_generated_token_ids=[5])
    solver.first_token_id.side_effect = lambda text: 5   # e.g. "1953" vs "1967"
    fields = pipeline._candidate_fields(
        solver, pipeline.A2Presentation.CANDIDATES, "a1_first", "1953", "1967")
    assert fields["candidate_first_token_ids"] == {"a1": 5, "critic": 5}
    assert fields["decision_token_distinct"] is False
    solver.first_token_id.side_effect = lambda text: None
    fields = pipeline._candidate_fields(
        solver, pipeline.A2Presentation.CANDIDATES, "a1_first", "", "1967")
    assert fields["decision_token_distinct"] is False


def test_missing_recommendation_aborts_before_any_second_attempt(monkeypatch):
    monkeypatch.setattr(pipeline, "MultiSiteCapture", FakeCapture)
    solver, critic, validator = make_agents([])
    critic._generate = lambda prompt: '{"verdict": "disagree", "advocated_answer": "", "explanation": "x"}'
    with pytest.raises(pipeline.CriticRecommendationError, match="no recommendation"):
        pipeline.run_question(
            question_id="2hop__1_2", question=QUESTION, paragraphs=PARAGRAPHS, gold=GOLD,
            aliases=[], model=object(), solver=solver, critic=critic, validator=validator,
            candidate_sites=SITES, seed=42, active_conditions=["natural"],
            a2_presentations=["feedback", "candidates"], candidate_order="a1_first",
        )
    solver.revise.assert_not_called()
    solver.choose.assert_not_called()


def test_candidates_presentation_requires_an_order(monkeypatch):
    with pytest.raises(ValueError, match="candidate_order must be one of"):
        run_presentations(monkeypatch, [], a2_presentations=["candidates"], candidate_order=None)


def test_presentations_omitted_keeps_the_record_schema(monkeypatch):
    result = run(monkeypatch, [])
    for record in records(result).values():
        assert "a2_presentation" not in record and "candidate_order" not in record
        assert record["episode_id"] == f"2hop__1_2__{record['critic_condition']}"


def test_two_presentations_save_with_aligned_activation_rows(monkeypatch, tmp_path):
    from mas_sae.data.activation_store import ActivationStore
    from mas_sae.experiments import collection
    from mas_sae.experiments.collection_artifacts import save_collection_artifacts
    from mas_sae.experiments.records import read_jsonl

    calls = iter(range(100))

    class RowCapture(FakeCapture):
        def __init__(self, model, sites):
            value = float(next(calls))   # one distinct row per capture, in call order
            self.activations = {site: torch.full((1, 4), value) for site in sites}

    monkeypatch.setattr(pipeline, "MultiSiteCapture", RowCapture)
    solver, critic, validator = make_agents([])
    critic._generate = lambda prompt: NATURAL_JSON
    solver.solve.return_value = "Steve Hillage"
    solver.choose.return_value = "Steve Hillage"
    solver.first_token_id.side_effect = lambda text: len(text)
    solver.last_generated_token_ids = [3]
    solver.last_generation = solver.last_prompt = validator.last_generation = None
    examples = [{"id": q, "question": QUESTION, "paragraphs": PARAGRAPHS, "answer": GOLD}
                for q in ("2hop__1_2", "2hop__3_4")]
    orders = {"2hop__1_2": "a1_first", "2hop__3_4": "critic_first"}
    result = collection.collect_examples(
        examples=examples, source_split="train", model=object(), solver=solver, critic=critic,
        validator=validator, candidate_sites=SITES, base_seed=42, active_conditions=["natural"],
        a2_presentations=["feedback", "candidates"], candidate_orders=orders,
    )
    resolved = {"collection": {"active_conditions": ["natural"], "a2_presentations": ["feedback", "candidates"]}}
    save_collection_artifacts(
        activation_root=tmp_path / "activations", result_root=tmp_path / "results",
        run_name="r", source_split="train", candidate_sites=SITES,
        attempt1_by_site=result["attempt1_by_site"], attempt2_by_site=result["attempt2_by_site"],
        records=result["records"], resolved_config=resolved, exclusions=result["exclusions"],
    )

    rows = read_jsonl(tmp_path / "results" / "r" / "train" / "interactions.jsonl")
    assert [(r["question_id"], r["a2_presentation"], r["candidate_order"]) for r in rows] == [
        ("2hop__1_2", "feedback", None), ("2hop__1_2", "candidates", "a1_first"),
        ("2hop__3_4", "feedback", None), ("2hop__3_4", "candidates", "critic_first")]
    store = ActivationStore(tmp_path / "activations" / "r" / "layer_08")
    a1, a2 = store.load_activations("train_attempt1"), store.load_activations("train_attempt2")
    # captures run A1, A2 feedback, A2 candidates per question: rows 0,1,2 then 3,4,5
    assert a1[:, 0].tolist() == [0.0, 3.0]
    assert [a2[r["attempt2_activation_index"], 0].item() for r in rows] == [1.0, 2.0, 4.0, 5.0]
    assert [a1[r["attempt1_activation_index"], 0].item() for r in rows] == [0.0, 0.0, 3.0, 3.0]

    with pytest.raises(ValueError, match="a2_presentations"):
        save_collection_artifacts(
            activation_root=tmp_path / "activations", result_root=tmp_path / "results",
            run_name="other", source_split="train", candidate_sites=SITES,
            attempt1_by_site=result["attempt1_by_site"], attempt2_by_site=result["attempt2_by_site"],
            records=result["records"], resolved_config={"collection": {"active_conditions": ["natural"]}},
            exclusions=result["exclusions"],
        )
