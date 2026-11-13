"""The harness's reply parsing (scripts/sae/llm_edit_sae.py) and steer_test's answer classification."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas_sae.llm_edit.steer_test import classify_answer

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/sae/llm_edit_sae.py"
spec = importlib.util.spec_from_file_location("llm_edit_sae", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.mark.parametrize("text,expected", [
    ("I'll measure first.\n```python\nprint(separation(4409))\n```", ("python", "print(separation(4409))\n")),
    ("Let's check the effect.\n\nsteer_test()", ("steer", "")),
    ("Let's check.\n`steer_test()`", ("steer", "")),
    ("DONE\nI rotated three rows.", ("done", "DONE\nI rotated three rows.")),
    ("**DONE**: summary here", ("done", "**DONE**: summary here")),
])
def test_one_action_is_recognized(text, expected):
    assert harness.parse_action(text) == expected


@pytest.mark.parametrize("text,problem", [
    ("Just thinking aloud.", "no action"),
    ("```python\na = 1\n```\n```python\nb = 2\n```", "2 python blocks"),
    ("```python\na = 1\n```\nsteer_test()", "more than one action"),
])
def test_anything_but_exactly_one_action_is_an_error(text, problem):
    kind, message = harness.parse_action(text)
    assert kind == "error" and problem in message


def test_words_inside_code_are_not_actions():
    text = "```python\nprint('DONE')\n# steer_test()\n```"
    assert harness.parse_action(text)[0] == "python"


def test_answers_are_classified_by_exact_match_only():
    episode = SimpleNamespace(solver_attempt_1="The United Nations", critic_advocated_answer="Big Four")
    assert classify_answer("the United Nations.", episode) == "retention"
    assert classify_answer("Big Four", episode) == "adoption"
    assert classify_answer("NATO", episode) is None


def test_long_output_keeps_its_start_and_end():
    text = "a" * 5000 + "END"
    cut = harness.truncate(text)
    assert cut.startswith("aaa") and cut.endswith("END") and "characters cut" in cut and len(cut) < 4100
