"""Tests for the Claude binary behavioral validator.

All tests use mocked ClaudeClient responses — no real Bedrock calls.
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mas_sae.evaluation.claude_validator import (
    BINARY_LABEL_COLUMNS,
    LABELING_SYSTEM,
    PASSTHROUGH_FIELDS,
    PROMPT_VERSION,
    VALID_LABELS,
    build_labeling_prompt,
    export_binary_labels,
    export_exclusions,
    parse_label_response,
    validate_batch,
    validate_episode,
)
from mas_sae.models.claude import ClaudeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _episode(
    *,
    question_id="musique:2hop__test_1",
    episode_id="musique:2hop__test_1__natural__feedback",
    question="Who founded the city where the Eiffel Tower is located?",
    a1="Napoleon",
    c1="The city is Paris, founded by the Parisii tribe.",
    a2="The Parisii tribe",
    source_split="train",
    partition="train",
    dataset_source="musique",
    extra=None,
):
    """Build a minimal episode dict for testing."""
    ep = {
        "question_id": question_id,
        "episode_id": episode_id,
        "question": question,
        "solver_attempt_1": a1,
        "critic_advocated_answer": c1,
        "solver_attempt_2": a2,
        "source_split": source_split,
        "partition": partition,
        "dataset_source": dataset_source,
    }
    if extra:
        ep.update(extra)
    return ep


def _mock_client(label="adoption", reason="A2 matches C1.", stop_reason="end_turn"):
    """Return a mock ClaudeClient whose generate() returns a canned result."""
    text = json.dumps({"label": label, "reason": reason})
    result = ClaudeResult(
        text=text,
        model="claude-test-model",
        input_tokens=100,
        output_tokens=50,
        task="labeling",
        stop_reason=stop_reason,
    )
    client = MagicMock(spec=["generate", "close"])
    client.generate.return_value = result
    return client


# ---------------------------------------------------------------------------
# build_labeling_prompt
# ---------------------------------------------------------------------------

class TestBuildLabelingPrompt:

    def test_contains_all_fields(self):
        prompt = build_labeling_prompt("Q?", "A1", "C1", "A2")
        assert '"question": "Q?"' in prompt
        assert '"A1": "A1"' in prompt
        assert '"C1": "C1"' in prompt
        assert '"A2": "A2"' in prompt

    def test_no_gold_answer(self):
        prompt = build_labeling_prompt("Q?", "A1", "C1", "A2")
        assert "gold" not in prompt.lower()
        assert "ground_truth" not in prompt
        assert "answer_aliases" not in prompt

    def test_instructions_present(self):
        prompt = build_labeling_prompt("Q?", "A1", "C1", "A2")
        assert "adoption" in prompt
        assert "retention" in prompt
        assert "agreement" in prompt
        assert "out" in prompt
        assert "factual correctness" in prompt


# ---------------------------------------------------------------------------
# parse_label_response
# ---------------------------------------------------------------------------

class TestParseLabelResponse:

    def test_plain_json(self):
        text = '{"label": "adoption", "reason": "A2 matches C1."}'
        result = parse_label_response(text)
        assert result == {"label": "adoption", "reason": "A2 matches C1."}

    def test_fenced_json(self):
        text = 'Some preamble\n```json\n{"label": "retention", "reason": "Kept A1."}\n```'
        result = parse_label_response(text)
        assert result["label"] == "retention"

    def test_last_fence_wins(self):
        text = (
            '```json\n{"label": "adoption", "reason": "first"}\n```\n'
            'correction:\n'
            '```json\n{"label": "retention", "reason": "second"}\n```'
        )
        assert parse_label_response(text)["label"] == "retention"

    def test_all_valid_labels(self):
        for label in VALID_LABELS:
            text = json.dumps({"label": label, "reason": "test"})
            assert parse_label_response(text)["label"] == label

    def test_invalid_label_raises(self):
        text = '{"label": "maybe", "reason": "unsure"}'
        with pytest.raises(ValueError, match="Unexpected label"):
            parse_label_response(text)

    def test_missing_reason_raises(self):
        text = '{"label": "adoption"}'
        with pytest.raises(ValueError, match="missing"):
            parse_label_response(text)

    def test_empty_reason_raises(self):
        text = '{"label": "adoption", "reason": "   "}'
        with pytest.raises(ValueError, match="non-empty reason"):
            parse_label_response(text)

    def test_extra_field_raises(self):
        text = '{"label": "adoption", "reason": "ok", "confidence": 0.9}'
        with pytest.raises(ValueError, match="unexpected"):
            parse_label_response(text)

    def test_not_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_label_response("This is not JSON at all.")

    def test_json_array_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            parse_label_response('[{"label": "adoption", "reason": "ok"}]')


# ---------------------------------------------------------------------------
# validate_episode
# ---------------------------------------------------------------------------

class TestValidateEpisode:

    def test_adoption(self):
        client = _mock_client(label="adoption", reason="A2 follows C1.")
        result = validate_episode(client, _episode())

        assert result["primary_target"] == 1
        assert result["eligible_primary"] is True
        assert result["claude_label"] == "adoption"
        assert result["claude_reason"] == "A2 follows C1."
        assert result["exclusion_reason"] is None
        assert result["claude_model"] == "claude-test-model"
        assert result["prompt_version"] == PROMPT_VERSION
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_retention(self):
        client = _mock_client(label="retention", reason="A2 keeps A1.")
        result = validate_episode(client, _episode())

        assert result["primary_target"] == 0
        assert result["eligible_primary"] is True
        assert result["claude_label"] == "retention"

    def test_agreement(self):
        client = _mock_client(label="agreement", reason="A1 and C1 agree.")
        result = validate_episode(client, _episode())

        assert result["primary_target"] is None
        assert result["eligible_primary"] is False
        assert result["claude_label"] == "agreement"
        assert result["exclusion_reason"] == "no_conflict"

    def test_out(self):
        client = _mock_client(label="out", reason="A2 matches neither.")
        result = validate_episode(client, _episode())

        assert result["primary_target"] is None
        assert result["eligible_primary"] is False
        assert result["claude_label"] == "out"
        assert result["exclusion_reason"] == "claude_unclassifiable"

    def test_missing_required_field(self):
        client = _mock_client()
        ep = _episode()
        del ep["solver_attempt_2"]
        result = validate_episode(client, ep)

        assert result["eligible_primary"] is False
        assert "missing_field:solver_attempt_2" in result["exclusion_reason"]
        # No API call should have been made.
        client.generate.assert_not_called()

    def test_empty_required_field(self):
        client = _mock_client()
        ep = _episode(a1="   ")
        result = validate_episode(client, ep)

        assert result["eligible_primary"] is False
        assert "missing_field:solver_attempt_1" in result["exclusion_reason"]
        client.generate.assert_not_called()

    def test_incomplete_response(self):
        client = _mock_client(stop_reason="max_tokens")
        result = validate_episode(client, _episode())

        assert result["eligible_primary"] is False
        assert "incomplete_response" in result["exclusion_reason"]

    def test_parse_error(self):
        bad_result = ClaudeResult(
            text="Not JSON at all",
            model="claude-test-model",
            input_tokens=100, output_tokens=50,
            task="labeling", stop_reason="end_turn",
        )
        client = MagicMock(spec=["generate", "close"])
        client.generate.return_value = bad_result
        result = validate_episode(client, _episode())

        assert result["eligible_primary"] is False
        assert result["exclusion_reason"] == "parse_error"

    def test_passthrough_fields_preserved(self):
        client = _mock_client()
        ep = _episode(extra={
            "attempt2_activation_index": 42,
            "sae_attempt2_index": 99,
            "critic_condition": "natural",
        })
        result = validate_episode(client, ep)

        assert result["episode_id"] == ep["episode_id"]
        assert result["question_id"] == ep["question_id"]
        assert result["source_split"] == "train"
        assert result["partition"] == "train"
        assert result["dataset_source"] == "musique"
        assert result["attempt2_activation_index"] == 42
        assert result["sae_attempt2_index"] == 99
        assert result["critic_condition"] == "natural"

    def test_missing_optional_fields_not_fabricated(self):
        client = _mock_client()
        ep = {
            "question_id": "test:q1",
            "question": "Q?",
            "solver_attempt_1": "A1",
            "critic_advocated_answer": "C1",
            "solver_attempt_2": "A2",
        }
        result = validate_episode(client, ep)

        assert "episode_id" not in result
        assert "partition" not in result
        assert "attempt2_activation_index" not in result

    def test_synthetic_id_preserved(self):
        client = _mock_client()
        ep = _episode(
            question_id="synthetic:2hop__abc123",
            episode_id="synthetic:2hop__abc123__natural__feedback",
            dataset_source="synthetic",
        )
        result = validate_episode(client, ep)

        assert result["question_id"] == "synthetic:2hop__abc123"
        assert result["episode_id"] == "synthetic:2hop__abc123__natural__feedback"
        assert result["dataset_source"] == "synthetic"

    def test_system_prompt_used(self):
        client = _mock_client()
        validate_episode(client, _episode())

        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["system"] == LABELING_SYSTEM
        assert call_kwargs.kwargs["task"] == "labeling"
        assert call_kwargs.kwargs["temperature"] == 0.0

    def test_gold_answer_not_in_prompt(self):
        client = _mock_client()
        ep = _episode(extra={
            "ground_truth": "SecretGold42",
            "answer_aliases": ["secretgold42"],
        })
        validate_episode(client, ep)

        prompt_sent = client.generate.call_args.args[0]
        # The gold answer value must not leak into the prompt.
        assert "SecretGold42" not in prompt_sent
        # The field keys themselves must not appear either.
        assert "ground_truth" not in prompt_sent
        assert "answer_aliases" not in prompt_sent


# ---------------------------------------------------------------------------
# validate_batch
# ---------------------------------------------------------------------------

class TestValidateBatch:

    def test_basic_batch(self, tmp_path):
        client = _mock_client(label="adoption", reason="ok")
        episodes = [_episode(question_id=f"q:{i}") for i in range(3)]
        out = tmp_path / "results.jsonl"

        summary = validate_batch(client, episodes, output_path=out)

        assert summary["n_episodes"] == 3
        assert summary["label_counts"]["adoption"] == 3
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150
        assert len(summary["results"]) == 3

        # Verify incremental JSONL was written.
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            row = json.loads(line)
            assert row["primary_target"] == 1

    def test_mixed_labels(self, tmp_path):
        episodes = [
            _episode(question_id="q:adopt"),
            _episode(question_id="q:retain"),
            _episode(question_id="q:agree"),
            _episode(question_id="q:out"),
        ]
        responses = [
            ("adoption", "A2 follows C1."),
            ("retention", "A2 keeps A1."),
            ("agreement", "A1 and C1 agree."),
            ("out", "A2 matches neither."),
        ]

        call_count = 0
        def fake_generate(prompt, **kwargs):
            nonlocal call_count
            label, reason = responses[call_count]
            call_count += 1
            return ClaudeResult(
                text=json.dumps({"label": label, "reason": reason}),
                model="test", input_tokens=10, output_tokens=5,
                task="labeling", stop_reason="end_turn",
            )

        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = fake_generate

        summary = validate_batch(client, episodes, output_path=tmp_path / "out.jsonl")

        assert summary["label_counts"] == {
            "adoption": 1, "retention": 1, "agreement": 1, "out": 1, "error": 0,
        }

    def test_api_error_caught(self, tmp_path):
        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = RuntimeError("Connection timeout")

        summary = validate_batch(
            client, [_episode()], output_path=tmp_path / "out.jsonl",
        )

        assert summary["label_counts"]["error"] == 1
        row = summary["results"][0]
        assert row["eligible_primary"] is False
        assert row["exclusion_reason"] == "api_error"

    def test_output_directory_created(self, tmp_path):
        client = _mock_client()
        out = tmp_path / "nested" / "dir" / "results.jsonl"
        validate_batch(client, [_episode()], output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# export_binary_labels
# ---------------------------------------------------------------------------

class TestExportBinaryLabels:

    def test_only_eligible_rows(self, tmp_path):
        results = [
            {"episode_id": "e1", "question_id": "q1", "primary_target": 1,
             "eligible_primary": True, "claude_label": "adoption",
             "claude_reason": "ok", "prompt_version": PROMPT_VERSION},
            {"episode_id": "e2", "question_id": "q2", "primary_target": None,
             "eligible_primary": False, "claude_label": "agreement",
             "claude_reason": "A1 and C1 agree"},
            {"episode_id": "e3", "question_id": "q3", "primary_target": 0,
             "eligible_primary": True, "claude_label": "retention",
             "claude_reason": "kept", "prompt_version": PROMPT_VERSION},
        ]
        out = tmp_path / "labels.csv"
        export_binary_labels(results, out)

        with out.open(newline="") as fh:
            reader = list(csv.DictReader(fh))
        assert len(reader) == 2
        assert reader[0]["primary_target"] == "1"
        assert reader[0]["eligible_primary"] == "True"
        assert reader[1]["primary_target"] == "0"

    def test_csv_columns(self, tmp_path):
        results = [
            {"episode_id": "e1", "question_id": "q1", "primary_target": 1,
             "eligible_primary": True, "claude_label": "adoption",
             "claude_reason": "ok", "prompt_version": PROMPT_VERSION},
        ]
        out = tmp_path / "labels.csv"
        export_binary_labels(results, out)

        with out.open(newline="") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames) == BINARY_LABEL_COLUMNS

    def test_empty_when_no_eligible(self, tmp_path):
        results = [
            {"episode_id": "e1", "question_id": "q1", "primary_target": None,
             "eligible_primary": False},
        ]
        out = tmp_path / "labels.csv"
        export_binary_labels(results, out)

        with out.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# export_exclusions
# ---------------------------------------------------------------------------

class TestExportExclusions:

    def test_only_excluded_rows(self, tmp_path):
        results = [
            {"question_id": "q1", "eligible_primary": True},
            {"question_id": "q2", "eligible_primary": False,
             "exclusion_reason": "no_conflict"},
            {"question_id": "q3", "eligible_primary": False,
             "exclusion_reason": "missing_field:solver_attempt_2"},
        ]
        out = tmp_path / "exclusions.jsonl"
        export_exclusions(results, out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        row0 = json.loads(lines[0])
        assert row0["question_id"] == "q2"

    def test_empty_when_all_eligible(self, tmp_path):
        results = [
            {"question_id": "q1", "eligible_primary": True},
        ]
        out = tmp_path / "exclusions.jsonl"
        export_exclusions(results, out)

        content = out.read_text().strip()
        assert content == ""
