"""Tests for the Claude binary behavioral validator.

All tests use mocked ClaudeClient responses — no real Bedrock calls.
"""

import csv
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mas_sae.evaluation.claude_validator import (
    BINARY_LABEL_COLUMNS,
    LABELING_SYSTEM,
    PASSTHROUGH_FIELDS,
    PROMPT_VERSION,
    RULE_NONCOMMITTAL,
    VALID_LABELS,
    alignment_exclusions,
    build_labeling_prompt,
    is_technical_error,
    export_binary_labels,
    export_exclusions,
    parse_label_response,
    validate_batch,
    validate_episode,
)
from mas_sae.experiments.records import write_jsonl
from mas_sae.models.claude import ClaudeResult


def _load_cli():
    """scripts/ is not on the test path, so load the CLI by file like the other script tests."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_claude_validation.py"
    spec = importlib.util.spec_from_file_location("run_claude_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli()


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

    def test_agreement_requires_all_three(self):
        """Agreement rule must require A1, C1, AND A2 to agree (PR #153)."""
        prompt = build_labeling_prompt("Q?", "A1", "C1", "A2")
        assert "A1, C1, and A2 all agree" in prompt
        assert "regardless of A2" not in prompt


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

    def test_excluded_is_not_a_valid_label(self):
        """'excluded' is not a behavioral label (PR #153 four-outcome framework)."""
        text = '{"label": "excluded", "reason": "A1 and C1 agree."}'
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

    def test_noncommittal_critic_is_rule_out(self):
        client = _mock_client()
        ep = _episode(c1=None, extra={"critic_noncommittal": True})
        result = validate_episode(client, ep)

        client.generate.assert_not_called()
        assert result["behavior_label"] == "out"
        assert result["label_source"] == RULE_NONCOMMITTAL
        assert result["claude_label"] is None
        assert result["claude_model"] is None
        assert result["exclusion_reason"] == "missing_field:critic_advocated_answer"
        assert result["eligible_primary"] is False
        assert result["primary_target"] is None
        assert result["critic_noncommittal"] is True

    @pytest.mark.parametrize("ep_kwargs", [
        {"c1": None},                                                 # flag absent
        {"c1": None, "extra": {"critic_noncommittal": False}},        # flag false
        {"c1": None, "a2": "", "extra": {"critic_noncommittal": True}},  # also missing A2
    ])
    def test_other_missing_c1_stays_error(self, ep_kwargs):
        client = _mock_client()
        result = validate_episode(client, _episode(**ep_kwargs))

        client.generate.assert_not_called()
        assert result["behavior_label"] is None
        assert result["label_source"] is None
        assert result["exclusion_reason"] == "missing_field:critic_advocated_answer"

    def test_noncommittal_flag_with_c1_goes_to_claude(self):
        client = _mock_client(label="adoption")
        result = validate_episode(client, _episode(extra={"critic_noncommittal": True}))

        client.generate.assert_called_once()
        assert result["behavior_label"] == "adoption"
        assert result["label_source"] == "claude"

    def test_claude_labels_record_source(self):
        result = validate_episode(_mock_client(label="agreement"), _episode())
        assert result["behavior_label"] == result["claude_label"] == "agreement"
        assert result["label_source"] == "claude"

    def test_incomplete_response_not_retried(self):
        client = _mock_client(stop_reason="max_tokens")
        result = validate_episode(client, _episode())
        assert client.generate.call_count == 1
        assert result["exclusion_reason"] == "incomplete_response:max_tokens"
        assert is_technical_error(result)

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
        assert result["behavior_label"] is None
        # Bounded retry: three identical requests, every raw response kept, tokens summed.
        assert client.generate.call_count == 3
        assert result["claude_raw_responses"] == ["Not JSON at all"] * 3
        assert result["claude_attempts"] == 3
        assert (result["input_tokens"], result["output_tokens"]) == (300, 150)
        assert is_technical_error(result)

    def test_parse_retry_succeeds(self):
        bad = ClaudeResult(text='{"label": "adoption", "label_reason": "x"}', model="m",
                           input_tokens=10, output_tokens=5, task="labeling", stop_reason="end_turn")
        good = _mock_client(label="retention", reason="keeps A1").generate.return_value
        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = [bad, good]

        result = validate_episode(client, _episode())

        assert client.generate.call_count == 2
        prompts = [c.args[0] for c in client.generate.call_args_list]
        assert prompts[0] == prompts[1]  # identical request; prompt v2 unchanged
        assert result["behavior_label"] == "retention"
        assert result["claude_raw_responses"] == [bad.text, good.text]
        assert result["claude_attempts"] == 2
        assert (result["input_tokens"], result["output_tokens"]) == (110, 55)

    def test_max_attempts_one_disables_retry(self):
        client = _mock_client()
        client.generate.return_value = ClaudeResult(
            text="nope", model="m", input_tokens=1, output_tokens=1,
            task="labeling", stop_reason="end_turn")
        result = validate_episode(client, _episode(), max_attempts=1)
        assert client.generate.call_count == 1
        assert result["exclusion_reason"] == "parse_error"

    def test_success_keeps_raw_response(self):
        client = _mock_client(label="adoption", reason="ok")
        result = validate_episode(client, _episode())
        assert result["claude_raw_responses"] == [client.generate.return_value.text]
        assert result["claude_attempts"] == 1

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

    def test_rule_out_counted_as_out_not_error(self, tmp_path):
        from botocore.exceptions import ClientError

        responses = iter([
            _mock_client(label="adoption").generate.return_value,
            ClientError({"Error": {"Code": "ThrottlingException", "Message": "x"}}, "InvokeModel"),
            _mock_client(label="out").generate.return_value,
        ])
        def fake_generate(prompt, **kwargs):
            item = next(responses)
            if isinstance(item, Exception):
                raise item
            return item
        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = fake_generate
        episodes = [
            _episode(question_id="q:adopt"),
            _episode(question_id="q:throttled"),
            _episode(question_id="q:claude_out"),
            _episode(question_id="q:noncommittal", c1=None, extra={"critic_noncommittal": True}),
            _episode(question_id="q:corrupt", c1=None),
        ]

        summary = validate_batch(client, episodes, output_path=tmp_path / "out.jsonl")

        assert summary["label_counts"] == {
            "adoption": 1, "retention": 0, "agreement": 0, "out": 2, "error": 2,
        }
        assert summary["label_sources"] == {"claude": 2, RULE_NONCOMMITTAL: 1}
        assert summary["error_reasons"] == {
            "api_error": 1, "missing_field:critic_advocated_answer": 1,
        }
        assert client.generate.call_count == 3

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

    def test_refuses_to_overwrite_existing_results(self, tmp_path):
        out = tmp_path / "raw_results.jsonl"
        out.write_text('{"earlier": "result"}\n')
        client = _mock_client()

        with pytest.raises(FileExistsError):
            validate_batch(client, [_episode()], output_path=out)

        assert out.read_text() == '{"earlier": "result"}\n'
        client.generate.assert_not_called()

    def test_stops_on_expired_token(self, tmp_path):
        from botocore.exceptions import ClientError

        expired = ClientError(
            {"Error": {"Code": "ExpiredTokenException", "Message": "token expired"}},
            "InvokeModel",
        )
        ok = _mock_client().generate.return_value
        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = [ok, expired, ok]
        out = tmp_path / "out.jsonl"
        episodes = [_episode(question_id=f"q:{i}") for i in range(3)]

        with pytest.raises(ClientError):
            validate_batch(client, episodes, output_path=out)

        assert client.generate.call_count == 2
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["question_id"] for r in rows] == ["q:0"]

    def test_stops_on_missing_credentials(self, tmp_path):
        from botocore.exceptions import NoCredentialsError

        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = NoCredentialsError()
        with pytest.raises(NoCredentialsError):
            validate_batch(client, [_episode()], output_path=tmp_path / "out.jsonl")

    def test_throttling_recorded_not_fatal(self, tmp_path):
        from botocore.exceptions import ClientError

        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "InvokeModel",
        )
        summary = validate_batch(
            client, [_episode(question_id="q:0"), _episode(question_id="q:1")],
            output_path=tmp_path / "out.jsonl",
        )
        assert summary["label_counts"]["error"] == 2
        assert {r["exclusion_reason"] for r in summary["results"]} == {"api_error"}

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


# ---------------------------------------------------------------------------
# Prompt version
# ---------------------------------------------------------------------------

class TestPromptVersion:

    def test_version_is_v2(self):
        """Prompt version bumped to v2 after Agreement rule correction."""
        assert PROMPT_VERSION == "claude_binary_v2"

    def test_valid_labels_are_four_outcomes(self):
        """Exactly the four PR #153 behavioral outcomes, no 'excluded'."""
        assert VALID_LABELS == ("adoption", "retention", "agreement", "out")


# ---------------------------------------------------------------------------
# filter_by_condition (CLI helper)
# ---------------------------------------------------------------------------

class TestFilterByCondition:

    def test_keeps_only_matching_condition(self):
        filter_by_condition = CLI.filter_by_condition

        episodes = [
            {"question_id": "q1", "critic_condition": "natural"},
            {"question_id": "q2", "critic_condition": "controlled_correct"},
            {"question_id": "q3", "critic_condition": "natural"},
            {"question_id": "q4", "critic_condition": "controlled_incorrect"},
        ]
        matching, counts = filter_by_condition(episodes, "natural")

        assert len(matching) == 2
        assert all(ep["critic_condition"] == "natural" for ep in matching)
        assert counts == {
            "natural": 2,
            "controlled_correct": 1,
            "controlled_incorrect": 1,
        }

    def test_missing_condition_counted(self):
        filter_by_condition = CLI.filter_by_condition

        episodes = [
            {"question_id": "q1", "critic_condition": "natural"},
            {"question_id": "q2"},  # no critic_condition
        ]
        matching, counts = filter_by_condition(episodes, "natural")

        assert len(matching) == 1
        assert counts == {"natural": 1, "MISSING": 1}

    def test_empty_when_no_match(self):
        filter_by_condition = CLI.filter_by_condition

        episodes = [
            {"question_id": "q1", "critic_condition": "controlled_correct"},
        ]
        matching, counts = filter_by_condition(episodes, "natural")

        assert len(matching) == 0
        assert counts == {"controlled_correct": 1}

    def test_all_natural(self):
        filter_by_condition = CLI.filter_by_condition

        episodes = [
            {"question_id": f"q{i}", "critic_condition": "natural"}
            for i in range(5)
        ]
        matching, counts = filter_by_condition(episodes, "natural")

        assert len(matching) == 5
        assert counts == {"natural": 5}


# ---------------------------------------------------------------------------
# technical vs. behavioral outcomes
# ---------------------------------------------------------------------------

class TestTechnicalErrors:

    def test_classification(self):
        rule_out = validate_episode(_mock_client(), _episode(c1=None, extra={"critic_noncommittal": True}))
        claude_out = validate_episode(_mock_client(label="out"), _episode())
        data_error = validate_episode(_mock_client(), _episode(a2=""))
        assert not is_technical_error(rule_out)
        assert not is_technical_error(claude_out)
        assert not is_technical_error(data_error)
        assert is_technical_error({"behavior_label": None, "exclusion_reason": "api_error"})

    def test_batch_reports_retries_and_technical_errors(self, tmp_path):
        bad = ClaudeResult(text="x", model="m", input_tokens=1, output_tokens=1,
                           task="labeling", stop_reason="end_turn")
        good = _mock_client(label="adoption").generate.return_value
        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = [bad, good, bad, bad, bad]

        summary = validate_batch(client, [_episode(question_id="q:1"), _episode(question_id="q:2")],
                                 output_path=tmp_path / "out.jsonl")

        assert summary["label_counts"]["adoption"] == 1
        assert summary["error_reasons"] == {"parse_error": 1}
        assert summary["n_technical_errors"] == 1
        assert summary["n_retried"] == 2
        assert summary["max_attempts"] == 3


# ---------------------------------------------------------------------------
# alignment_exclusions
# ---------------------------------------------------------------------------

def _labelled_results(labels):
    """Run validate_episode once per (question_id, label) with a mocked client."""
    return [
        validate_episode(
            _mock_client(label=label),
            _episode(question_id=qid, episode_id=f"{qid}__natural"),
        )
        for qid, label in labels
    ]


class TestAlignmentExclusions:

    def test_maps_every_ineligible_result(self):
        results = _labelled_results([
            ("q1", "adoption"), ("q2", "retention"), ("q3", "agreement"), ("q4", "out"),
        ])
        results.append(validate_episode(_mock_client(), _episode(question_id="q5", a2="")))

        assert alignment_exclusions(results) == {
            "q3": "claude:agreement",
            "q4": "claude:out",
            "q5": "claude:missing_field:solver_attempt_2",
        }

    def test_rule_out_reason_names_its_source(self):
        row = validate_episode(
            _mock_client(),
            _episode(question_id="q9", c1=None, extra={"critic_noncommittal": True}),
        )
        assert alignment_exclusions([row]) == {"q9": f"{RULE_NONCOMMITTAL}:out"}

    def test_missing_question_id_raises(self):
        row = validate_episode(_mock_client(), _episode(question_id=""))
        with pytest.raises(ValueError, match="no question_id"):
            alignment_exclusions([row])

    def test_duplicate_question_id_raises(self):
        results = _labelled_results([("q1", "out"), ("q1", "agreement")])
        with pytest.raises(ValueError, match="Duplicate"):
            alignment_exclusions(results)


class TestProbeLoaderCompatibility:
    """binary_labels.csv + alignment_exclusions through Raye's unmodified loader."""

    @pytest.fixture
    def probe_inputs(self, tmp_path):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("torch")
        from mas_sae.probe import behavior_data as bd

        labels = [("q0", "adoption"), ("q1", "retention"), ("q2", "agreement"),
                  ("q3", "out"), ("q4", "adoption"), ("q5", "retention")]
        interactions = [
            dict(_episode(question_id=qid, episode_id=f"{qid}__natural",
                          partition="train"),
                 critic_condition="natural", attempt2_activation_index=i)
            for i, (qid, _) in enumerate(labels)
        ]
        results_dir = tmp_path / "run"
        write_jsonl(results_dir / "train" / "interactions.jsonl", interactions)
        results = _labelled_results(labels)
        export_binary_labels(results, tmp_path / "binary_labels.csv")

        episodes = bd.build_episode_table(results_dir, ["train"], ["solver_attempt_2"])
        manifest = pd.DataFrame({
            "question_id": [qid for qid, _ in labels],
            "partition": "train", "source_split": "train",
            "full_index": range(len(labels)),
        })
        return bd, episodes, pd.read_csv(tmp_path / "binary_labels.csv"), manifest, results

    def test_binary_labels_alone_fail_loudly(self, probe_inputs):
        bd, episodes, labels, manifest, _ = probe_inputs
        with pytest.raises(bd.AlignmentError, match="2 episode\\(s\\) have no row in labels.csv"):
            bd.align_episodes(episodes, labels, manifest, bd.Columns(), "full_index")

    def test_exclusions_account_for_every_episode(self, probe_inputs):
        bd, episodes, labels, manifest, results = probe_inputs
        cols = bd.Columns()

        aligned, diag = bd.align_episodes(
            episodes, labels, manifest, cols, "full_index",
            excluded=alignment_exclusions(results),
        )
        selected, _ = bd.select_partitions(aligned, cols, ["train"])

        assert diag["labels_join"] == "episode_id"
        assert diag["excluded_episodes"] == {"q2": "claude:agreement", "q3": "claude:out"}
        assert diag["n_joined"] + diag["n_excluded"] == diag["n_episodes"] == 6
        assert dict(zip(selected["ep_question_id"], selected["primary_target"])) == {
            "q0": 1, "q1": 0, "q4": 1, "q5": 0,
        }
        assert dict(zip(selected["ep_question_id"], selected["activation_row"])) == {
            "q0": 0, "q1": 1, "q4": 4, "q5": 5,
        }


# ---------------------------------------------------------------------------
# CLI scope, selection and overwrite guards
# ---------------------------------------------------------------------------

class TestCliScope:

    def test_condition_option_removed(self):
        parse_args = CLI.parse_args

        with pytest.raises(SystemExit):
            parse_args(["--episodes", "x", "--output", "y",
                        "--condition", "controlled_correct"])

    @pytest.mark.parametrize("partition", ["test", "intervention"])
    def test_sealed_partitions_refused(self, partition):
        parse_args = CLI.parse_args

        with pytest.raises(SystemExit):
            parse_args(["--episodes", "x", "--output", "y", "--partition", partition])

    def test_unpartitioned_records_refused(self):
        filter_by_partition = CLI.filter_by_partition

        with pytest.raises(ValueError, match="no 'partition' field"):
            filter_by_partition([{"episode_id": "e1"}], "train")

    def test_partition_filter(self):
        filter_by_partition = CLI.filter_by_partition

        eps = [{"partition": "train"}, {"partition": "test"}, {"partition": "train"}]
        assert filter_by_partition(eps, "train") == [eps[0], eps[2]]


class TestSelectEpisodes:

    def test_limit_takes_first_complete_in_file_order(self):
        select_episodes = CLI.select_episodes

        eps = [_episode(question_id="q0", a2=""), _episode(question_id="q1"),
               _episode(question_id="q2", c1=None), _episode(question_id="q3"),
               _episode(question_id="q4")]
        selected, skipped = select_episodes(eps, 2)

        assert [ep["question_id"] for ep in selected] == ["q1", "q3"]
        assert skipped == 2

    def test_no_limit_keeps_incomplete_for_exclusion_records(self):
        select_episodes = CLI.select_episodes

        eps = [_episode(question_id="q0", a2=""), _episode(question_id="q1")]
        assert select_episodes(eps, None) == (eps, 0)

    def test_repeated_question_id_refused(self):
        select_episodes = CLI.select_episodes

        with pytest.raises(ValueError, match="repeat"):
            select_episodes([_episode(question_id="q1"), _episode(question_id="q1")], 5)

    def test_selection_record_keeps_identity_and_indices(self):
        selection_record = CLI.selection_record

        ep = _episode(extra={"source_activation_index": 7, "attempt2_activation_index": 3})
        assert selection_record([ep]) == [{
            "episode_id": ep["episode_id"], "question_id": ep["question_id"],
            "partition": "train", "source_split": "train",
            "source_activation_index": 7, "attempt2_activation_index": 3,
        }]


class TestCliOutputGuard:

    def _write_episodes(self, tmp_path):
        path = tmp_path / "interactions.jsonl"
        write_jsonl(path, [dict(_episode(), critic_condition="natural")])
        return path

    def test_existing_output_refused_before_any_work(self, tmp_path, monkeypatch):
        cli = CLI

        out = tmp_path / "pilot"
        out.mkdir()
        monkeypatch.setattr(cli, "ClaudeClient", MagicMock(side_effect=AssertionError))
        with pytest.raises(FileExistsError):
            cli.main(["--episodes", str(self._write_episodes(tmp_path)),
                      "--output", str(out), "--limit", "1"])

    def test_rejected_credentials_stop_before_output_created(self, tmp_path, monkeypatch):
        from botocore.exceptions import ClientError
        cli = CLI

        session = MagicMock()
        session.client.return_value.get_caller_identity.side_effect = ClientError(
            {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
            "GetCallerIdentity",
        )
        monkeypatch.setattr(cli, "load_dotenv", lambda: None)
        monkeypatch.setattr(cli.boto3, "Session", MagicMock(return_value=session))
        monkeypatch.setattr(cli, "ClaudeClient", MagicMock(side_effect=AssertionError))
        out = tmp_path / "pilot"

        with pytest.raises(SystemExit):
            cli.main(["--episodes", str(self._write_episodes(tmp_path)),
                      "--output", str(out), "--limit", "1"])

        assert not out.exists()

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        cli = CLI

        out = tmp_path / "pilot"
        cli.main(["--episodes", str(self._write_episodes(tmp_path)),
                  "--output", str(out), "--limit", "1", "--dry-run"])

        assert not out.exists()
        assert "musique:2hop__test_1__natural__feedback" in capsys.readouterr().out


class TestRetryTechnicalErrorsFrom:

    def _run_dir(self, tmp_path, sha, rows):
        run = tmp_path / "run"
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({"episodes_sha256": sha}))
        write_jsonl(run / "raw_results.jsonl", rows)
        return run

    def test_selects_only_technical_errors(self, tmp_path):
        technical_error_ids = CLI.technical_error_ids

        rows = [
            {"episode_id": "e1", "behavior_label": None, "exclusion_reason": "parse_error"},
            {"episode_id": "e2", "behavior_label": None, "exclusion_reason": "api_error"},
            {"episode_id": "e3", "behavior_label": "out", "exclusion_reason": "claude_unclassifiable"},
            {"episode_id": "e4", "behavior_label": None, "exclusion_reason": "missing_field:solver_attempt_2"},
            {"episode_id": "e5", "behavior_label": "adoption", "exclusion_reason": None},
        ]
        assert technical_error_ids(self._run_dir(tmp_path, "abc", rows), "abc") == {"e1", "e2"}

    def test_refuses_a_different_source_file(self, tmp_path):
        technical_error_ids = CLI.technical_error_ids

        with pytest.raises(ValueError, match="different episodes file"):
            technical_error_ids(self._run_dir(tmp_path, "abc", []), "xyz")


class TestAuditFixes:

    def test_max_attempts_below_one_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="max_attempts"):
            validate_episode(_mock_client(), _episode(), max_attempts=0)
        with pytest.raises(ValueError, match="max_attempts"):
            validate_batch(_mock_client(), [_episode()], output_path=tmp_path / "o.jsonl", max_attempts=0)
        assert not (tmp_path / "o.jsonl").exists()

    @pytest.mark.parametrize("flag", ["--limit", "--max-attempts"])
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_cli_rejects_nonpositive_counts(self, flag, value):
        with pytest.raises(SystemExit):
            CLI.parse_args(["--episodes", "x", "--output", "y", flag, value])

    def test_api_error_after_malformed_response_keeps_audit(self, tmp_path):
        bad = ClaudeResult(text="not json", model="m", input_tokens=100, output_tokens=20,
                           task="labeling", stop_reason="end_turn")
        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = [bad, RuntimeError("connection reset")]

        summary = validate_batch(client, [_episode()], output_path=tmp_path / "o.jsonl")
        row = summary["results"][0]

        assert row["exclusion_reason"] == "api_error"
        assert row["claude_raw_responses"] == ["not json"]
        assert row["claude_attempts"] == 2
        assert (row["input_tokens"], row["output_tokens"]) == (100, 20)
        assert summary["total_input_tokens"] == 100
        assert summary["n_retried"] == 1

    def test_credential_error_after_malformed_response_still_stops(self, tmp_path):
        from botocore.exceptions import ClientError

        bad = ClaudeResult(text="not json", model="m", input_tokens=1, output_tokens=1,
                           task="labeling", stop_reason="end_turn")
        client = MagicMock(spec=["generate", "close"])
        client.generate.side_effect = [bad, ClientError(
            {"Error": {"Code": "ExpiredTokenException", "Message": "x"}}, "InvokeModel")]
        with pytest.raises(ClientError):
            validate_batch(client, [_episode()], output_path=tmp_path / "o.jsonl")
