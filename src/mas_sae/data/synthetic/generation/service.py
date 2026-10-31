"""Generate one candidate with bounded repair retries and attempt provenance."""
import random
from dataclasses import asdict
from typing import Any

from ..contracts import SyntheticCandidate, TextGenerator
from ..responses import parse_response
from ..validation import validate_candidate
from .assembly import ParagraphPool, assemble_candidate
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, build_repair_prompt


def generate_one(
    client: TextGenerator,
    hop_type: str,
    pool: ParagraphPool,
    rng: random.Random,
    *,
    max_attempts: int = 2,
    max_tokens: int = 6000,
    temperature: float | None= None,
    provenance: dict[str, Any] | None = None,
) -> tuple[SyntheticCandidate | None, dict]:
    """Generate one synthetic candidate using the API. Make up to ``max_attempts`` API calls; return the candidate (or None) and provenance.

    Parameters
    ----------
    client : TextGenerator
        API client for text generation.
    hop_type : str
        Type of hop for the question.
    pool : ParagraphPool
        Pool of paragraphs to use.
    rng : random.Random
        Random number generator.
    max_attempts : int, optional
        Max API calls, by default 2
    max_tokens : int, optional
        Max tokens for generation, by default 6000
    temperature : float, None
        Sampling temperature. None omits the parameter from the API request. by default None
    provenance : dict[str, Any] | None, optional
        Caller-owned dictionary retaining completed attempts if generation raises.

    Returns
    -------
    tuple[SyntheticCandidate | None, dict]
        The candidate and its provenance.
    """

    base_prompt = build_prompt(hop_type)
    prompt = base_prompt

    if provenance is None:
        provenance = {}

    provenance.update({
        "record_id": None,
        "hop_type": hop_type,
        "model_id": client.model_id,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_calls": 0,
        "attempts": [],
    })

    for _ in range(max_attempts):
        provenance["api_calls"] += 1
        result = client.generate(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens,
                                 temperature=temperature, task="data-gen")
        attempt = {"model": result.model, "input_tokens": result.input_tokens,
                   "output_tokens": result.output_tokens, "stop_reason": result.stop_reason,
                   "raw_text": result.text, "error": None, "precheck_errors": [],
                   "sdk_retry_attempts": getattr(result, "sdk_retry_attempts", 0)}
        provenance["attempts"].append(attempt)
        if result.stop_reason != "end_turn":
            attempt["error"] = f"stop_reason={result.stop_reason}"
            prompt = build_repair_prompt(base_prompt, attempt)
            continue
        try:
            candidate = assemble_candidate(parse_response(result.text), hop_type, pool, rng)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            prompt = build_repair_prompt(base_prompt, attempt)
            continue
        errors = validate_candidate(candidate).errors
        attempt["precheck_errors"] = [asdict(error) for error in errors]
        provenance["record_id"] = candidate["id"]
        if not errors:
            return candidate, provenance
        prompt = build_repair_prompt(base_prompt, attempt)
    return None, provenance
