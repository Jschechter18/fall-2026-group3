from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from mas_sae.agents.base import Agent, ProcessorProtocol


@dataclass
class ValidatorVerdict:
    is_correct: bool | None
    raw_output: str


class Validator(Agent):
    """LLM-based secondary answer-equivalence judge."""

    VALIDATE_PROMPT = (
        "You are an exact answer-equivalence checker.\n"
        "Compare the candidate answer ONLY with the gold answer "
        "and accepted aliases.\n"
        "Do not solve the original question.\n"
        "Do not treat a related person, company, place, event, "
        "or concept as equivalent.\n"
        "Reply YES only if the candidate refers to the same "
        "entity or value as the gold answer or an accepted alias.\n"
        "Otherwise reply NO.\n"
        "Reply with one word only: YES or NO.\n\n"
        "Gold answer: {gold_answer}\n"
        "Accepted aliases: {aliases}\n"
        "Candidate answer: {candidate}\n"
        "Reply:"
    )

    def __init__(
        self,
        model: Any,
        processor: ProcessorProtocol,
        *,
        max_new_tokens: int = 4,
    ) -> None:
        super().__init__(
            model,
            processor,
            max_new_tokens=max_new_tokens,
        )

    def build_prompt(
        self,
        question: str,
        gold_answer: str,
        aliases: Iterable[str],
        candidate: str,
    ) -> str:
        _ = question

        if isinstance(aliases, str):
            aliases = (aliases,)

        alias_text = ", ".join(aliases) or "none"

        return self.VALIDATE_PROMPT.format(
            gold_answer=gold_answer,
            aliases=alias_text,
            candidate=candidate,
        )

    def validate(
        self,
        question: str,
        gold_answer: str,
        aliases: Iterable[str],
        candidate: str,
    ) -> ValidatorVerdict:
        raw = self._generate(
            self.build_prompt(
                question,
                gold_answer,
                aliases,
                candidate,
            )
        )

        tokens = raw.strip().split()

        if not tokens:
            return ValidatorVerdict(
                is_correct=None,
                raw_output=raw,
            )

        first = tokens[0].strip(
            ".,:;!?\"'"
        ).upper()

        if first == "YES":
            verdict = True
        elif first == "NO":
            verdict = False
        else:
            verdict = None

        return ValidatorVerdict(
            is_correct=verdict,
            raw_output=raw,
        )
