"""Multi-turn Claude on Bedrock for the LLM-edit harness.

AdaptiveThinkingClaudeClient.generate is single-turn and returns only text. The harness needs a
conversation: it sends the whole message list each turn and appends each response's content blocks
UNCHANGED (Claude 5.x thinking blocks carry signatures and must be resent exactly; Fable 5.2 rejects
any edit to earlier history). This subclass reuses the parent's Bedrock runtime client and request
body (no sampling parameters, explicit effort) and returns the raw response.

thinking.display = "summarized": the model still thinks the same way and is billed the same; the
response then carries a readable summary of its reasoning instead of an empty string, which the
transcript keeps for the write-up.
"""

from __future__ import annotations

import json

from mas_sae.evaluation.claude_adaptive_client import AdaptiveThinkingClaudeClient


class ConversationClient(AdaptiveThinkingClaudeClient):
    def send(self, messages: list[dict], *, system: str, max_tokens: int) -> dict:
        """One turn: the full conversation in, the raw response (content blocks, usage, stop_reason) out."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.effort},
        }
        response = self._client.invoke_model(modelId=self.model_id, contentType="application/json",
                                             accept="application/json", body=json.dumps(body))
        stream = response["body"]
        try:
            return json.loads(stream.read())
        finally:
            stream.close()
