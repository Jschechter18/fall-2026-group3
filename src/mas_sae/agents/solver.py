from typing import Any, Protocol

import torch


class ProcessorProtocol(Protocol):
    """Processor interface required by the Solver."""

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        ...


class Solver:
    """Generate answers to questions from provided context paragraphs."""

    SOLVE_PROMPT_V1 = (
        "Answer the question using only the paragraphs provided.\n"
        "Respond with the answer only, no explanation.\n\n"
        "{paragraphs}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

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

    @staticmethod
    def format_paragraphs(
        paragraphs: list[dict[str, Any]],
    ) -> str:
        return "\n\n".join(
            f"[{p['idx']}] {p['title']}: {p['paragraph_text']}"
            for p in paragraphs
        )

    def build_prompt(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
    ) -> str:
        context = self.format_paragraphs(paragraphs)

        return self.SOLVE_PROMPT_V1.format(
            paragraphs=context,
            question=question,
        )

    def solve(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
    ) -> str:
        prompt = self.build_prompt(question, paragraphs)
        return self._generate(prompt)

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
