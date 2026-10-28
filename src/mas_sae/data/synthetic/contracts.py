"""Contracts for answerable synthetic MuSiQue records (provenance stays separate)."""
from dataclasses import dataclass, field
from typing import TypedDict


class Paragraph(TypedDict):
    idx: int
    title: str
    paragraph_text: str
    is_supporting: bool


class DecompositionStep(TypedDict):
    id: int
    question: str
    answer: str
    paragraph_support_idx: int


class SyntheticCandidate(TypedDict):
    id: str
    question: str
    answer: str
    answer_aliases: list[str]
    answerable: bool
    paragraphs: list[Paragraph]
    question_decomposition: list[DecompositionStep]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    field: str
    message: str


@dataclass
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class DuplicateMatch:
    record_id: str
    source: str
    normalized_question: str
