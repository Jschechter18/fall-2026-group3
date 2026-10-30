"""Generate one synthetic MuSiQue-style candidate for review.

Run from the repository root:
    python examples/claude/claude_for_data_gen.py

Requires the local MuSiQue train/validation JSONL files and AWS credentials
in .env. Makes one paid API call; prints and saves JSON without modifying the dataset.
Generated facts and semantic novelty require review before appending.
"""

import json
import os
import random
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import boto3
from dotenv import load_dotenv

from mas_sae.data.synthetic import normalize_question
from mas_sae.models.claude import ClaudeClient


def parse_candidate(text: str) -> dict:
    """Parse a JSON object, allowing a surrounding Markdown code fence."""
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].lower() in ("```json", "```") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    candidate = json.loads(text)
    if not isinstance(candidate, dict):
        raise ValueError("Expected one JSON object.")
    if not isinstance(candidate.get("question"), str) or not candidate["question"].strip():
        raise ValueError("Expected a non-empty question string.")
    return candidate


def main() -> None:
    """Request one varied question and reject exact local question duplicates."""
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")

    # These files are JSONL despite their .json extension.
    existing_questions = set()
    for split in ("train", "validation"):
        with (root / "data" / "MuSiQue" / "clean" / f"{split}.json").open() as source:
            for line in source:
                question = json.loads(line)["question"]
                existing_questions.add(normalize_question(question))

    # Vary the topic and reasoning depth instead of relying only on temperature.
    topic = random.choice([
        "botany and exploration", "architecture and literature",
        "astronomy and scientific instruments", "music and geography",
        "inventions and transportation", "archaeology and languages",
    ])
    hops = random.choice([2, 3, 4])
    prompt = f"""Generate exactly ONE new, answerable multi-hop QA example.
    Use the theme '{topic}' and a chain of {hops} supporting paragraphs.
    Choose unusual real-world entities and relationships; avoid common trivia.
    Create an original question, not a paraphrase of a known MuSiQue question.
    The question must require combining all {hops} supporting paragraphs, with
    each paragraph supplying a necessary step. No single paragraph may answer it.
    Use accurate, stable facts; do not invent entities, dates, or relationships.
    Include two plausible distractor paragraphs that do not introduce ambiguity.
    Return one JSON object only, without Markdown fences, with these fields:
    - question: a question with one unambiguous short answer
    - answer: the short answer
    - answer_aliases: a list of equivalent answers, or []
    - answerable: true
    - paragraphs: a shuffled list of objects, each with idx (sequential integer
    starting at 0), title, paragraph_text, and is_supporting (boolean)
    """

    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
    )
    claude = ClaudeClient(
        session=session,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    started = perf_counter()
    try:
        result = claude.generate(
            prompt,
            system="You create diverse, evidence-grounded multi-hop QA examples.",
            task="data-gen",
            temperature=1.0, # High temperature for more diverse outputs, this is perfect for our data generation task.
            max_tokens=4096,
        )
    finally:
        elapsed = perf_counter() - started
        claude.close()

    artifacts = root / "examples" / "claude" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    print(f"Generation took {elapsed:.2f} seconds")
    if result.stop_reason != "end_turn":
        raise RuntimeError(f"Generation did not finish normally: {result.stop_reason}")
    try:
        candidate = parse_candidate(result.text)
    except ValueError as exc:
        raise ValueError(
            "Claude did not return a usable JSON question. "
            f"Response starts with: {result.text[:200]!r}"
        ) from exc
    normalized = normalize_question(candidate["question"])
    if normalized in existing_questions:
        raise ValueError("Generated question duplicates a local MuSiQue question.")

    candidate["id"] = f"{hops}hop__synthetic_{uuid4().hex}"
    output = artifacts / "data_gen_example.json"
    serialized = json.dumps(candidate, indent=2, ensure_ascii=False)
    output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
