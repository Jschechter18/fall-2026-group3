from unittest.mock import Mock

import torch

from mas_sae.agents import Solver


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
