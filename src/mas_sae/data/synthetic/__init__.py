"""Structural validation and exact deduplication for synthetic MuSiQue."""
from .contracts import DuplicateMatch, SyntheticCandidate, ValidationIssue, ValidationResult
from .deduplication import DuplicateIndex, normalize_question
from .validation import validate_candidate

__all__ = ["DuplicateMatch", "SyntheticCandidate", "ValidationIssue", "ValidationResult",
           "DuplicateIndex", "normalize_question", "validate_candidate"]
