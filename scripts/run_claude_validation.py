"""Run Claude behavioral validation on Solver-Critic episodes (issue #155).

Reads episodes from an interactions.jsonl file, sends each A1/C1/A2 triplet
to Claude through Amazon Bedrock, and writes probe-ready binary labels
(Adoption = 1, Retention = 0) alongside an exclusion audit file.

Requires AWS Bedrock credentials in ``.env`` (see README).  Use ``--dry-run``
to preview without making API calls.

Usage:
    PYTHONPATH=src python scripts/run_claude_validation.py \\
        --episodes results/collection/natural_4b_full/train/interactions.jsonl \\
        --output results/claude_validation/pilot_10 \\
        --limit 10 \\
        --dry-run

    # Real pilot (after authorization):
    PYTHONPATH=src python scripts/run_claude_validation.py \\
        --episodes results/collection/natural_4b_full/train/interactions.jsonl \\
        --output results/claude_validation/pilot_10 \\
        --limit 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

from mas_sae.evaluation.claude_validator import (
    PROMPT_VERSION,
    build_labeling_prompt,
    export_binary_labels,
    export_exclusions,
    validate_batch,
)
from mas_sae.experiments.records import read_jsonl
from mas_sae.models.claude import ClaudeClient

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Claude behavioral validation for Solver-Critic episodes.",
    )
    parser.add_argument(
        "--episodes", type=Path, required=True,
        help="Path to interactions.jsonl or a directory containing one.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for labels, exclusions and summary.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N episodes (for pilot runs).",
    )
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help=f"Bedrock model ID (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build prompts and print counts without making API calls.",
    )
    return parser.parse_args(argv)


def load_episodes(path: Path, limit: int | None = None) -> list[dict]:
    """Load episodes from a JSONL file or from a directory containing one."""
    if path.is_dir():
        path = path / "interactions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Episodes file not found: {path}")
    episodes = read_jsonl(path)
    if limit is not None:
        episodes = episodes[:limit]
    return episodes


def main(argv=None) -> None:
    args = parse_args(argv)

    episodes = load_episodes(args.episodes, args.limit)
    print(f"Loaded {len(episodes)} episode(s) from {args.episodes}")

    if args.dry_run:
        print(f"\n--- DRY RUN (prompt_version={PROMPT_VERSION}) ---")
        print(f"Model: {args.model_id}")
        print(f"Episodes to process: {len(episodes)}")
        for i, ep in enumerate(episodes[:3]):
            prompt = build_labeling_prompt(
                question=ep.get("question", ""),
                a1=ep.get("solver_attempt_1", ""),
                c1=ep.get("critic_advocated_answer", ""),
                a2=ep.get("solver_attempt_2", ""),
            )
            print(f"\n--- Episode {i} ({ep.get('question_id', '?')}) ---")
            print(f"Prompt length: {len(prompt)} chars")
        if len(episodes) > 3:
            print(f"\n... and {len(episodes) - 3} more episode(s)")
        print("\nNo API calls made. Remove --dry-run to process.")
        return

    # Real run: create Bedrock client from .env credentials.
    load_dotenv()
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
    )
    client = ClaudeClient(session=session, model_id=args.model_id)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary = validate_batch(
            client, episodes,
            output_path=output_dir / "raw_results.jsonl",
        )
    finally:
        client.close()

    results = summary.pop("results")
    export_binary_labels(results, output_dir / "binary_labels.csv")
    export_exclusions(results, output_dir / "exclusions.jsonl")

    summary["model_id"] = args.model_id
    summary["prompt_version"] = PROMPT_VERSION
    summary["episodes_path"] = str(args.episodes)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\nDone. {summary['n_episodes']} episodes processed.")
    print(f"Labels: {summary['label_counts']}")
    print(f"Tokens: {summary['total_input_tokens']} in / {summary['total_output_tokens']} out")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
