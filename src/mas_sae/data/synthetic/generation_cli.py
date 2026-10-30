"""Generate synthetic MuSiQue candidates with Claude (paid API calls)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from uuid import uuid4

from .generation import ParagraphPool, generate_one
from .hop_types import HOP_TYPES


DEFAULT_MODEL_ID = "us.anthropic.claude-opus-5"


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
    parser.add_argument("--seed", type=int, help="Seed for filler paragraph sampling and shuffling")
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

    counts: Counter = Counter(
        generated=0,
        failed=0,
        api_calls=0,
        completed_responses=0,
        input_tokens=0,
        output_tokens=0,
    )
    by_hop: Counter = Counter()
    fatal_error = None

    # TODO: connect reviewed synthetic data to the master dataset workflow.
    try:
        if owns_client:
            client = make_client(args.model_id)
        with (
            (args.output_dir / "candidates.jsonl").open("w", encoding="utf-8") as candidates,
            (args.output_dir / "provenance.jsonl").open("w", encoding="utf-8") as provenance_file,
        ):
            for i in range(args.n):
                hop_type = args.hop_types[i % len(args.hop_types)]
                provenance = {
                    "hop_type": hop_type,
                    "attempts": [],
                }

                try:
                    candidate, provenance = generate_one(
                        client,
                        hop_type,
                        pool,
                        rng,
                        provenance=provenance,
                    )
                except Exception as exc:
                    # Record context, then preserve the traceback and failure status.
                    fatal_error = f"{type(exc).__name__}: {exc}"
                    provenance["error"] = fatal_error
                    counts["failed"] += 1
                    raise
                finally:
                    counts["api_calls"] += provenance.get("api_calls", 0)
                    for attempt in provenance["attempts"]:
                        counts["completed_responses"] += 1
                        counts["input_tokens"] += attempt["input_tokens"]
                        counts["output_tokens"] += attempt["output_tokens"]

                    provenance_file.write(
                        json.dumps(provenance, ensure_ascii=False) + "\n"
                    )
                    provenance_file.flush()

                status = "ok" if candidate is not None else "FAILED"
                print(
                    f"[{i + 1}/{args.n}] {hop_type:6}: {status}",
                    flush=True,
                )

                if candidate is None:
                    counts["failed"] += 1
                    continue

                candidates.write(
                    json.dumps(candidate, ensure_ascii=False) + "\n"
                )
                candidates.flush()
                counts["generated"] += 1
                by_hop[hop_type] += 1

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            summary = {
                **counts,
                "generated_by_hop_type": dict(by_hop),
                "model_id": client.model_id if client is not None else args.model_id,
                "seed": args.seed,
                "output_dir": str(args.output_dir.resolve()),
                "status": "failed" if fatal_error else "completed",
                "fatal_error": fatal_error,
            }
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2))
        finally:
            if owns_client and client is not None:
                client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
