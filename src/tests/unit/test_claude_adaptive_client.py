"""Tests for the Claude 5.x Bedrock client used by the validator (no network)."""

import io
import json
from unittest.mock import MagicMock

import pytest

from mas_sae.evaluation.claude_adaptive_client import (
    AdaptiveThinkingClaudeClient,
    uses_adaptive_thinking,
)


def _client(response, effort="medium"):
    session = MagicMock()
    runtime = session.client.return_value
    runtime.invoke_model.return_value = {"body": io.BytesIO(json.dumps(response).encode())}
    return AdaptiveThinkingClaudeClient(session, model_id="us.anthropic.claude-opus-5-5", effort=effort), runtime


RESPONSE = {"model": "claude-opus-5-5", "stop_reason": "end_turn",
            "usage": {"input_tokens": 500, "output_tokens": 120},
            "content": [{"type": "thinking", "thinking": "", "signature": "x"},
                        {"type": "text", "text": '{"label": "adoption", "reason": "A2 = C1"}'}]}


@pytest.mark.parametrize("model_id, adaptive", [
    ("us.anthropic.claude-opus-5-5", True), ("us.anthropic.claude-sonnet-5-5", True),
    ("global.anthropic.claude-opus-5", True), ("us.anthropic.claude-fable-5-1", True),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", False), ("anthropic.claude-haiku-4-5", False),
])
def test_model_family_detection(model_id, adaptive):
    assert uses_adaptive_thinking(model_id) is adaptive


def test_request_body_has_no_sampling_params_and_explicit_effort():
    client, runtime = _client(RESPONSE, effort="high")
    client.generate("prompt", system="sys", task="labeling", max_tokens=6000)
    body = json.loads(runtime.invoke_model.call_args.kwargs["body"])
    assert "temperature" not in body and "top_p" not in body and "top_k" not in body
    assert "thinking" not in body  # adaptive by default on Claude 5.x
    assert body["output_config"] == {"effort": "high"}
    assert body["anthropic_version"] == "bedrock-2023-05-31" and body["max_tokens"] == 6000
    assert runtime.invoke_model.call_args.kwargs["modelId"] == "us.anthropic.claude-opus-5-5"


def test_only_text_blocks_are_returned():
    client, _ = _client(RESPONSE)
    result = client.generate("prompt", task="labeling")
    assert result.text == '{"label": "adoption", "reason": "A2 = C1"}'
    assert (result.model, result.input_tokens, result.output_tokens, result.stop_reason) == \
        ("claude-opus-5-5", 500, 120, "end_turn")


def test_temperature_and_bad_effort_refused():
    client, runtime = _client(RESPONSE)
    with pytest.raises(ValueError, match="sampling parameters"):
        client.generate("prompt", task="labeling", temperature=0.0)
    runtime.invoke_model.assert_not_called()
    with pytest.raises(ValueError, match="effort"):
        _client(RESPONSE, effort="extreme")


def test_validator_omits_temperature_for_claude5(tmp_path):
    from mas_sae.evaluation.claude_validator import validate_episode

    client, runtime = _client(RESPONSE)
    episode = {"question_id": "q1", "episode_id": "q1__natural", "question": "Q?",
               "solver_attempt_1": "Paris", "critic_advocated_answer": "Lyon", "solver_attempt_2": "Lyon"}
    row = validate_episode(client, episode, temperature=None, max_tokens=8000)
    assert row["behavior_label"] == "adoption" and row["claude_model"] == "claude-opus-5-5"
    assert "temperature" not in json.loads(runtime.invoke_model.call_args.kwargs["body"])


def test_transport_config_reaches_the_bedrock_client():
    from botocore.config import Config
    session = MagicMock()
    config = Config(max_pool_connections=8)
    AdaptiveThinkingClaudeClient(session, model_id="us.anthropic.claude-opus-5-5", config=config)
    assert session.client.call_args.kwargs["config"] is config
