"""Synthetic generation services; preserves the original generation imports."""


from ..contracts import DecompositionStep, SyntheticCandidate, TextGenerator
from ..hop_types import HOP_SHAPES, HOP_TYPES
from ..responses import parse_response
from .assembly import ParagraphPool, assemble_candidate
from .examples import build_example
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, build_repair_prompt, describe_shape
from .service import generate_one

__all__ = [
    "DecompositionStep", "SyntheticCandidate", "TextGenerator", "parse_response",
    "ParagraphPool", "assemble_candidate", "PROMPT_VERSION", "SYSTEM_PROMPT",
    "build_prompt", "build_repair_prompt", "describe_shape", "generate_one",
    "HOP_SHAPES", "HOP_TYPES", "build_example",
]
