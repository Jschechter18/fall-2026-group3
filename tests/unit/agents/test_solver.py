import pytest
from unittest.mock import Mock

import torch

from mas_sae.agents.solver import Solver


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


def make_solver(max_new_tokens=32):
    model = Mock()
    model.device = "cpu"
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    processor = Mock()
    processor.apply_chat_template.return_value = FakeInputs(
        {"input_ids": torch.tensor([[1, 2, 3]])}
    )
    processor.decode.return_value = "  Miquette Giraudy  "

    solver = Solver(
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
    )

    return solver, model, processor


def test_build_prompt_matches_baseline_exactly():
    solver, _, _ = make_solver()

    paragraphs = [
        {
            "idx": 5,
            "title": "Miquette Giraudy",
            "paragraph_text": "Miquette Giraudy is the partner of Steve Hillage.",
        },
        {
            "idx": 10,
            "title": "Green",
            "paragraph_text": "Green is an album by Steve Hillage.",
        },
    ]

    prompt = solver.build_prompt(
        "Who is the spouse of the Green performer?",
        paragraphs,
    )

    expected = (
        "Answer the question using only the paragraphs provided.\n"
        "Respond with the answer only, no explanation.\n\n"
        "[5] Miquette Giraudy: Miquette Giraudy is the partner of Steve Hillage.\n\n"
        "[10] Green: Green is an album by Steve Hillage.\n\n"
        "Question: Who is the spouse of the Green performer?\n"
        "Answer:"
    )

    assert prompt == expected


def test_format_paragraphs_preserves_indices_and_spacing():
    paragraphs = [
        {
            "idx": 2,
            "title": "First",
            "paragraph_text": "First paragraph.",
        },
        {
            "idx": 7,
            "title": "Second",
            "paragraph_text": "Second paragraph.",
        },
    ]

    assert Solver.format_paragraphs(paragraphs) == (
        "[2] First: First paragraph.\n\n"
        "[7] Second: Second paragraph."
    )
    assert Solver.format_paragraphs([]) == ""


def test_solve_uses_expected_chat_template():
    solver, _, processor = make_solver()

    solver.solve(
        "Question?",
        [{"idx": 0, "title": "Title", "paragraph_text": "Text."}],
    )

    messages = processor.apply_chat_template.call_args.args[0]

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][0]["text"].endswith(
        "Question: Question?\nAnswer:"
    )

    assert processor.apply_chat_template.call_args.kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }


def test_solve_uses_baseline_generation_settings():
    solver, model, _ = make_solver()

    solver.solve(
        "Question?",
        [{"idx": 0, "title": "Title", "paragraph_text": "Text."}],
    )

    kwargs = model.generate.call_args.kwargs

    assert kwargs["max_new_tokens"] == 32
    assert kwargs["do_sample"] is False


def test_solve_returns_stripped_decoded_output():
    solver, _, _ = make_solver()

    answer = solver.solve(
        "Question?",
        [{"idx": 0, "title": "Title", "paragraph_text": "Text."}],
    )

    assert answer == "Miquette Giraudy"


def test_custom_max_new_tokens_is_respected():
    solver, model, _ = make_solver(max_new_tokens=64)

    solver.solve(
        "Question?",
        [{"idx": 0, "title": "Title", "paragraph_text": "Text."}],
    )

    assert model.generate.call_args.kwargs["max_new_tokens"] == 64


def test_build_revise_prompt_contains_original_answer_and_feedback():
    solver, _, _ = make_solver()

    prompt = solver.build_revise_prompt(
        question="What city is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        previous_answer="Lyon",
        feedback="The answer should be Paris.",
    )

    assert "Previous answer: Lyon" in prompt
    assert "Reviewer feedback: The answer should be Paris." in prompt
    assert prompt.endswith("Final answer:")


def test_revise_uses_revision_prompt():
    solver, _, processor = make_solver()

    solver.revise(
        question="What city is the capital of France?",
        paragraphs=[
            {
                "idx": 0,
                "title": "France",
                "paragraph_text": "Paris is the capital of France.",
            }
        ],
        previous_answer="Lyon",
        feedback="The answer should be Paris.",
    )

    messages = processor.apply_chat_template.call_args.args[0]
    text = messages[0]["content"][0]["text"]

    assert "Previous answer: Lyon" in text
    assert "Reviewer feedback: The answer should be Paris." in text
    assert text.endswith("Final answer:")


CANDIDATE_PARAGRAPHS = [{"idx": 0, "title": "Green", "paragraph_text": "Green is by Steve Hillage."}]


def test_candidate_prompt_lists_both_answers_with_their_source_in_either_order():
    solver, _, _ = make_solver()

    a1_first = solver.build_candidate_prompt("Who made Green?", CANDIDATE_PARAGRAPHS,
                                             "Steve Hillage", "Miquette Giraudy", "a1_first")
    assert a1_first == (
        "Answer the question using only the paragraphs provided.\n"
        "Consider the two candidate answers below. You may use either candidate, "
        "provide another answer, or state that the evidence is insufficient.\n"
        "Respond with the final answer text only, no explanation.\n\n"
        "[0] Green: Green is by Steve Hillage.\n\nQuestion: Who made Green?\n"
        "Candidate A (your previous answer): Steve Hillage\n"
        "Candidate B (the reviewer's recommendation): Miquette Giraudy\n"
        "Final answer:"
    )
    critic_first = solver.build_candidate_prompt("Who made Green?", CANDIDATE_PARAGRAPHS,
                                                 "Steve Hillage", "Miquette Giraudy", "critic_first")
    assert critic_first.endswith(
        "Candidate A (the reviewer's recommendation): Miquette Giraudy\n"
        "Candidate B (your previous answer): Steve Hillage\n"
        "Final answer:"
    )
    with pytest.raises(ValueError, match="order must be one of"):
        solver.build_candidate_prompt("Q", CANDIDATE_PARAGRAPHS, "a", "b", "random")


def test_candidate_prompt_is_the_adoption_pilot_labeled_prompt():
    """The production prompt is the labeled condition of PR #138, word for word."""
    pilot = pytest.importorskip("mas_sae.experiments.anonymized_adoption")
    assert Solver.CANDIDATE_PROMPT == pilot.CANDIDATE_TEMPLATE
    assert Solver.CANDIDATE_LABELS == pilot.LABELS
    assert Solver.CANDIDATE_ORDERS == pilot.ORDERS
    solver, _, _ = make_solver()
    for order in Solver.CANDIDATE_ORDERS:
        assert solver.build_candidate_prompt("Q?", CANDIDATE_PARAGRAPHS, "x", "y", order) == (
            pilot.CandidateSolver.build_candidate_prompt(solver, "Q?", CANDIDATE_PARAGRAPHS, "x", "y", "labeled", order))


def test_choose_generates_from_the_candidate_prompt():
    solver, _, processor = make_solver()
    assert solver.choose("Q?", CANDIDATE_PARAGRAPHS, "x", "y", "critic_first") == "Miquette Giraudy"
    messages = processor.apply_chat_template.call_args.args[0]
    assert messages[0]["content"][0]["text"] == solver.build_candidate_prompt(
        "Q?", CANDIDATE_PARAGRAPHS, "x", "y", "critic_first")


def test_first_token_id_uses_the_tokenizer_on_the_stripped_text():
    solver, _, processor = make_solver()
    processor.tokenizer = Mock(return_value={"input_ids": [7, 8]})
    assert solver.first_token_id("  1953 ") == 7
    processor.tokenizer.assert_called_once_with("1953", add_special_tokens=False)
    processor.tokenizer.return_value = {"input_ids": []}
    assert solver.first_token_id("") is None
