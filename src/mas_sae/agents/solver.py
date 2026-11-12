"""The stage-1 Solver: Attempt 1 from the context paragraphs, then Attempt 2 after the Critic.

Attempt 2 is prompted either with the Critic's full feedback (``revise``) or with Attempt 1 and
the Critic's recommendation as two labeled candidates (``choose``).
"""

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

    # Second attempt with the previous answer and the reviewer's recommendation
    # as two labeled candidates. Text and labels are the labeled condition of the
    # adoption feedback pilot (PR #138, CANDIDATE_TEMPLATE and LABELS in
    # mas_sae.experiments.anonymized_adoption); a test keeps the two identical.
    CANDIDATE_PROMPT = (
        "Answer the question using only the paragraphs provided.\n"
        "Consider the two candidate answers below. You may use either candidate, "
        "provide another answer, or state that the evidence is insufficient.\n"
        "Respond with the final answer text only, no explanation.\n\n"
        "{paragraphs}\n\nQuestion: {question}\n"
        "Candidate A{label_a}: {answer_a}\n"
        "Candidate B{label_b}: {answer_b}\n"
        "Final answer:"
    )
    CANDIDATE_LABELS = {
        "a1": " (your previous answer)",
        "critic": " (the reviewer's recommendation)",
    }
    # Which candidate is listed as A.
    CANDIDATE_ORDERS = ("a1_first", "critic_first")

    @staticmethod
    def format_paragraphs(
        paragraphs: list[dict[str, Any]],
    ) -> str:
        """Render paragraphs as ``[idx] title: paragraph_text`` blocks separated by blank lines."""
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
        """The Attempt 1 prompt (``SOLVE_PROMPT_V1``)."""
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
        """The Attempt 2 prompt for the feedback presentation (``REVISE_PROMPT_V1``)."""
        context = self.format_paragraphs(paragraphs)

        return self.REVISE_PROMPT_V1.format(
            paragraphs=context,
            question=question,
            previous_answer=previous_answer,
            feedback=feedback,
        )

    def build_candidate_prompt(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
        previous_answer: str,
        recommendation: str,
        order: str,
    ) -> str:
        """The Attempt 2 prompt for the candidates presentation (``CANDIDATE_PROMPT``).

        ``order`` (one of ``CANDIDATE_ORDERS``) decides whether Attempt 1 or the Critic's
        recommendation is Candidate A; raises ValueError on any other value.
        """
        if order not in self.CANDIDATE_ORDERS:
            raise ValueError(
                f"order must be one of {list(self.CANDIDATE_ORDERS)}, got {order!r}"
            )

        origins = ("a1", "critic") if order == "a1_first" else ("critic", "a1")
        answers = {"a1": previous_answer, "critic": recommendation}

        return self.CANDIDATE_PROMPT.format(
            paragraphs=self.format_paragraphs(paragraphs),
            question=question,
            answer_a=answers[origins[0]],
            answer_b=answers[origins[1]],
            label_a=self.CANDIDATE_LABELS[origins[0]],
            label_b=self.CANDIDATE_LABELS[origins[1]],
        )

    def first_token_id(self, text: str) -> int | None:
        """First token id of ``text`` alone, as the Solver's tokenizer splits it."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        token_ids = tokenizer(str(text).strip(), add_special_tokens=False)["input_ids"]
        return int(token_ids[0]) if token_ids else None

    def solve(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
    ) -> str:
        """Generate Attempt 1."""
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
        """Generate Attempt 2 from the Critic's feedback text."""
        return self._generate(
            self.build_revise_prompt(
                question,
                paragraphs,
                previous_answer,
                feedback,
            )
        )

    def choose(
        self,
        question: str,
        paragraphs: list[dict[str, Any]],
        previous_answer: str,
        recommendation: str,
        order: str,
    ) -> str:
        """Generate Attempt 2 as a choice between Attempt 1 and the Critic's recommendation."""
        return self._generate(
            self.build_candidate_prompt(
                question,
                paragraphs,
                previous_answer,
                recommendation,
                order,
            )
        )
