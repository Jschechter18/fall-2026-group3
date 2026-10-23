from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from mas_sae.agents.base import Agent, ProcessorProtocol


@dataclass
class ValidatorVerdict:
    is_correct: bool | None
    raw_output: str


@dataclass
class AnswerRelation:
    """Whether two answers to one question name the same thing.

    ``relation`` is "same", "different" or "uncertain", or None when the reply
    was none of these.
    """
    relation: str | None
    raw_output: str


class Validator(Agent):
    """LLM-based secondary answer-equivalence judge."""

    SAME_ANSWER_PROMPT = (
        "You compare two answers to the same question.\n"
        "Decide whether both answers name the same entity, value or event "
        "as the answer to this question.\n"
        "Different wording, abbreviations, spelled-out numbers or extra detail "
        "about the same answer count as the same.\n"
        "A related but different person, place, organisation, date or quantity "
        "is different.\n"
        "Do not decide which answer is correct.\n"
        "Reply with one word only: SAME, DIFFERENT or UNCERTAIN.\n\n"
        "Question: {question}\n"
        "Answer 1: {answer_a}\n"
        "Answer 2: {answer_b}\n"
        "Reply:"
    )
    SAME_ANSWER_REPLIES = {"SAME": "same", "DIFFERENT": "different", "UNCERTAIN": "uncertain"}

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

    def build_same_answer_prompt(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> str:
        return self.SAME_ANSWER_PROMPT.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
        )

    def judge_same_answer(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> AnswerRelation:
        """Ask whether two answers to ``question`` name the same thing."""
        raw = self._generate(
            self.build_same_answer_prompt(question, answer_a, answer_b)
        )
        tokens = raw.strip().split()
        first = tokens[0].strip(".,:;!?\"'").upper() if tokens else ""

        return AnswerRelation(
            relation=self.SAME_ANSWER_REPLIES.get(first),
            raw_output=raw,
        )
