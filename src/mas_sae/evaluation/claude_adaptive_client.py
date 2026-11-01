"""Bedrock client for Claude 5.x models (Opus 5.5, Sonnet 5.5, ...) used by the validator.

The shared ``mas_sae.models.claude.ClaudeClient`` always sends ``temperature``; Claude 5.x
models reject sampling parameters with a 400 and always think (adaptive thinking, depth set
by ``output_config.effort``).  This subclass keeps the shared client untouched: it reuses its
Bedrock runtime client and result type and changes only the request body - no sampling
parameters, ``thinking`` omitted (adaptive is the default), explicit ``effort``.  Responses
carry thinking blocks before the answer; only ``text`` blocks are returned, as in the parent.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from botocore.config import Config

from mas_sae.models.claude import ClaudeClient, ClaudeResult

EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Model IDs whose request surface rejects sampling parameters and always runs adaptive thinking.
_ADAPTIVE_MODEL = re.compile(r"claude-(?:opus|sonnet|haiku)-5(?:-|$)|claude-(?:fable|mythos)-")


def uses_adaptive_thinking(model_id: str) -> bool:
    """True for Claude 5.x model / inference-profile IDs (e.g. ``us.anthropic.claude-opus-5-5``)."""
    return bool(_ADAPTIVE_MODEL.search(model_id))


class AdaptiveThinkingClaudeClient(ClaudeClient):
    """ClaudeClient for Claude 5.x on Bedrock: no temperature, adaptive thinking, explicit effort."""

    def __init__(self, session, *, model_id: str, region: str = "us-east-1", effort: str = "medium",
                 config: Config | None = None):
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
        super().__init__(session, model_id=model_id, region=region, config=config)
        self.effort = effort

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 8000,
                 temperature: float | None = None,
                 task: Literal["data-gen", "labeling", "other"]) -> ClaudeResult:
        """Same contract as ``ClaudeClient.generate``; ``max_tokens`` also covers thinking tokens."""
        if temperature is not None:
            raise ValueError("Claude 5.x models reject sampling parameters; leave temperature unset")
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if task not in ("data-gen", "labeling", "other"):
            raise ValueError("task must be 'data-gen', 'labeling', or 'other'.")

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }
        response = self._client.invoke_model(
            modelId=self.model_id, contentType="application/json", accept="application/json",
            body=json.dumps(body),
        )
        stream = response["body"]
        try:
            data = json.loads(stream.read())
        finally:
            stream.close()
        return ClaudeResult(
            text="".join(block["text"] for block in data["content"] if block["type"] == "text"),
            model=data["model"],
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            task=task,
            stop_reason=data.get("stop_reason"),
        )
