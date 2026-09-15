from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from mas_sae.agents.base import Agent
from mas_sae.agents.solver import Solver
from mas_sae.evaluation.scoring import answer_matches


class CriticCondition(StrEnum):
    NATURAL = "natural"
    CONTROLLED_CORRECT = "controlled_correct"
    CONTROLLED_INCORRECT = "controlled_incorrect"


@dataclass(frozen=True)
class CriticFeedback:
    condition: CriticCondition
    verdict: str | None
    advocated_answer: str | None
    explanation: str
    raw_output: str
    noncommittal: bool

    def to_solver_text(self) -> str:
        if self.verdict == "agree":
            verdict_text = "The reviewer agrees with your previous answer."
        elif self.verdict == "disagree":
            verdict_text = "The reviewer disagrees with your previous answer."
        else:
            verdict_text = (
                "The reviewer provided feedback without a clear "
                "agree or disagree verdict."
            )

        parts = [verdict_text]

        if self.advocated_answer:
            parts.append(
                f"The reviewer advocates this answer: {self.advocated_answer}"
            )

        if self.explanation:
            parts.append(
                f"Reviewer explanation: {self.explanation}"
            )

        return " ".join(parts)


NATURAL_PROMPT_V1 = (
    "Review another model's answer using only the paragraphs provided.\n"
    "Decide whether the proposed answer is correct.\n"
    "Always provide the answer you believe is correct in advocated_answer. "
    "If you agree, use the proposed answer. "
    "If you disagree, provide the answer you believe is correct.\n"
    "Return JSON only in exactly this form:\n"
    '{{"verdict": "agree" or "disagree", '
    '"advocated_answer": "<answer>", '
    '"explanation": "<brief explanation>"}}\n\n'
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "Proposed answer: {solver_answer}\n"
    "JSON:"
)


CONTROLLED_PROMPT_V1 = (
    "Review another model's answer using only the paragraphs provided.\n"
    "For this review, advocate the supplied target answer.\n"
    "Explain briefly why that answer should be preferred.\n\n"
    "{paragraphs}\n\n"
    "Question: {question}\n"
    "Proposed answer: {solver_answer}\n"
    "Answer to advocate: {advocated_answer}\n"
    "Review:"
)


_JSON_DECODER = json.JSONDecoder()


def _extract_first_json_object(
    raw_output: str,
) -> dict | None:
    for index, char in enumerate(raw_output):
        if char != "{":
            continue

        try:
            parsed, _ = _JSON_DECODER.raw_decode(
                raw_output[index:]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    return None


class Critic(Agent):
    def __init__(
        self,
        model,
        processor,
        max_new_tokens: int = 128,
    ) -> None:
        super().__init__(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def parse_natural(raw_output: str) -> CriticFeedback:
        parsed = _extract_first_json_object(raw_output)

        if parsed is None:
            return CriticFeedback(
                condition=CriticCondition.NATURAL,
                verdict=None,
                advocated_answer=None,
                explanation="",
                raw_output=raw_output,
                noncommittal=True,
            )

        verdict = parsed.get("verdict")

        if isinstance(verdict, str):
            verdict = verdict.strip().casefold()

        if verdict not in {"agree", "disagree"}:
            verdict = None

        advocated_answer = parsed.get("advocated_answer")

        if advocated_answer is not None:
            advocated_answer = str(advocated_answer).strip()

            if not advocated_answer:
                advocated_answer = None

        explanation = str(
            parsed.get("explanation", "")
        ).strip()

        noncommittal = (
            verdict is None
            or advocated_answer is None
        )

        return CriticFeedback(
            condition=CriticCondition.NATURAL,
            verdict=verdict,
            advocated_answer=advocated_answer,
            explanation=explanation,
            raw_output=raw_output,
            noncommittal=noncommittal,
        )

    def critique(
        self,
        *,
        question: str,
        paragraphs: list[dict],
        solver_answer: str,
        condition: CriticCondition,
        advocated_answer: str | None = None,
    ) -> CriticFeedback:
        formatted_paragraphs = Solver.format_paragraphs(
            paragraphs
        )

        if condition is CriticCondition.NATURAL:
            prompt = NATURAL_PROMPT_V1.format(
                paragraphs=formatted_paragraphs,
                question=question,
                solver_answer=solver_answer,
            )

            raw_output = self._generate(prompt)

            return self.parse_natural(raw_output)

        if advocated_answer is None:
            raise ValueError(
                "advocated_answer is required for controlled conditions"
            )

        prompt = CONTROLLED_PROMPT_V1.format(
            paragraphs=formatted_paragraphs,
            question=question,
            solver_answer=solver_answer,
            advocated_answer=advocated_answer,
        )

        raw_output = self._generate(prompt)

        verdict = (
            "agree"
            if answer_matches(
                solver_answer,
                advocated_answer,
            )
            else "disagree"
        )

        return CriticFeedback(
            condition=condition,
            verdict=verdict,
            advocated_answer=advocated_answer,
            explanation=raw_output,
            raw_output=raw_output,
            noncommittal=False,
        )
