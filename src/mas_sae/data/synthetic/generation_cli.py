"""Generate synthetic MuSiQue candidates with Claude (paid API calls)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from uuid import uuid4

from .generation import TOPICS, ParagraphPool, generate_one
from .validation import HOP_TYPES


# TODO: CODEX AUDIT
# P1 — Standalone validation accepts incorrect hop topology.
# cli.py:55 only calls validate_candidate(). The subtype-specific check_shape() runs only during generation. Consequently, a chained decomposition can be written to accepted.jsonl while claiming to be 3hop2, 4hop2, or 4hop3. Move hop-shape validation into validate_candidate() or explicitly call it from the validation CLI based on the ID prefix.

# P2 — Invalid model support values can silently become valid indices.
# In generation.py:215-217, dictionary lookup treats 0.0 as 0, and True/False as 1/0. The invalid type is then replaced with a valid integer paragraph index, so downstream validation cannot detect it. I confirmed that support: 0.0 passes validation. Require type(step["support"]) is int and bounds-check it before remapping.

# P2 — Unexpected programming errors are reported as ordinary generation failures.
# generation_cli.py:69-73 catches every Exception, despite the comment saying network/auth errors. A regression such as an AttributeError could make the batch finish with exit code 0 and only failed records. Catch the relevant Bedrock/network exceptions, or re-raise unexpected errors after recording context.


# TODO: SHOULD PROBABLY USE OPUS 5 RATHER THAN SONNET 4.5 - ALL REASONING SHOULD BE DONE ON THE MODEL BEST FOR COMPLEX REASONING
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def make_client(model_id: str):
    """Build a Bedrock ClaudeClient from .env credentials (imported lazily for tests)."""
    import boto3
    from dotenv import load_dotenv

    from mas_sae.models.claude import ClaudeClient

    load_dotenv()
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
    )
    return ClaudeClient(session=session, model_id=model_id)


def main(argv: list[str] | None = None, *, client=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="Number of records to request (one API call each, plus retries)")
    parser.add_argument("--hop-types", nargs="+", choices=sorted(HOP_TYPES), default=list(HOP_TYPES),
                        help="Hop types to cycle through (default: all)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, help="Seed for topics, filler paragraphs and shuffling")
    parser.add_argument("--original", type=Path, default=Path("data/MuSiQue/clean/train.json"),
                        help="Original MuSiQue JSONL used for filler distractor paragraphs")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory (default: data/MuSiQue/synthetic/runs/<run-id>/generation)")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be positive")
    if not args.original.is_file():
        parser.error(f"Original file does not exist: {args.original} (run scripts/get_musique_dataset.py)")
    if args.output_dir is None:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex}"
        args.output_dir = Path("data/MuSiQue/synthetic/runs") / run_id / "generation"
    if args.output_dir.exists():
        parser.error("Output directory must be new to avoid overwriting prior results")

    rng = random.Random(args.seed)
    pool = ParagraphPool.from_jsonl(args.original)
    args.output_dir.mkdir(parents=True)
    owns_client = client is None
    client = client or make_client(args.model_id)

    counts: Counter = Counter(generated=0, failed=0, api_calls=0, input_tokens=0, output_tokens=0)
    by_hop: Counter = Counter()
    try:
        # TODO: we need to set up a a directory for synthetic dataset that maps to master dataset merger from PR: issue160-partition-collection-runs
        with (args.output_dir / "candidates.jsonl").open("w", encoding="utf-8") as candidates, \
                (args.output_dir / "provenance.jsonl").open("w", encoding="utf-8") as provenance_file:
            for i in range(args.n):
                hop_type = args.hop_types[i % len(args.hop_types)]
                topic = rng.choice(TOPICS)
                try:
                    candidate, provenance = generate_one(client, hop_type, topic, pool, rng)
                except Exception as exc:  # network/auth errors: record and keep the batch going
                    candidate, provenance = None, {"hop_type": hop_type, "topic": topic, "attempts": [],
                                                   "error": f"{type(exc).__name__}: {exc}"}
                for attempt in provenance["attempts"]:
                    counts["api_calls"] += 1
                    counts["input_tokens"] += attempt["input_tokens"]
                    counts["output_tokens"] += attempt["output_tokens"]
                provenance_file.write(json.dumps(provenance, ensure_ascii=False) + "\n")
                provenance_file.flush()
                status = "ok" if candidate else "FAILED"
                print(f"[{i + 1}/{args.n}] {hop_type:6} {topic}: {status}", flush=True)
                if candidate is None:
                    counts["failed"] += 1
                    continue
                candidates.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                candidates.flush()
                counts["generated"] += 1
                by_hop[hop_type] += 1
    finally:
        if owns_client:
            client.close()

    summary = {**counts, "generated_by_hop_type": dict(by_hop), "model_id": args.model_id,
               "seed": args.seed, "output_dir": str(args.output_dir.resolve())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
