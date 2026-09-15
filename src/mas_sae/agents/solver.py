from __future__ import annotations

from typing import Any

from mas_sae.agents.base import Agent


class Solver(Agent):
    """Generate and revise answers using provided context paragraphs."""

    SOLVE_PROMPT_V1 = (
        "Answer the question using only the paragraphs provided.\n"
        "Respond with the answer only, no explanation.\n\n"
        "{paragraphs}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    REVISE_PROMPT_V1 = (
        "You previously answered a question using only the paragraphs provided.\n"
        "A reviewer has now given feedback about your answer.\n"
        "Consider the feedback carefully. You may keep your original answer "
        "or change it.\n"
        "Respond with the final answer only, no explanation.\n\n"
        "{paragraphs}\n\n"
        "Question: {question}\n"
        "Previous answer: {previous_answer}\n"
        "Reviewer feedback: {feedback}\n"
        "Final answer:"
    )

    @staticmethod
    def format_paragraphs(
        paragraphs: list[dict[str, Any]],
    ) -> str:
        return "\n\n".join(
            f"[{paragraph['idx']}] "
            f"{paragraph['title']}: "
            f"{paragraph['paragraph_text']}"
            for paragraph in paragraphs
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

    def build_revise_prompt(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
        previous_answer: str,
        feedback: str,
    ) -> str:
        context = self.format_paragraphs(paragraphs)

        return self.REVISE_PROMPT_V1.format(
            paragraphs=context,
            question=question,
            previous_answer=previous_answer,
            feedback=feedback,
        )

    def solve(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
    ) -> str:
        return self._generate(
            self.build_prompt(question, paragraphs)
        )

    def revise(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
        previous_answer: str,
        feedback: str,
    ) -> str:
        return self._generate(
            self.build_revise_prompt(
                question,
                paragraphs,
                previous_answer,
                feedback,
            )
        )
