from __future__ import annotations

from typing import Any, Protocol

import torch


class ProcessorProtocol(Protocol):
    """Processor interface required by generation agents."""

    def apply_chat_template(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        ...


class Agent:
    """Shared generation behavior for Solver, Critic, and Validator."""

    def __init__(
        self,
        model: Any,
        processor: ProcessorProtocol,
        *,
        max_new_tokens: int = 32,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def _generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_tokens = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        return self.processor.decode(
            output[0][prompt_tokens:],
            skip_special_tokens=True,
        ).strip()
