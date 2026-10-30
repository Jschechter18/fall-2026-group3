"""Generate synthetic MuSiQue candidates with bounded concurrent API requests."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from .batch import run_batch
from .hop_types import HOP_TYPES


DEFAULT_MODEL_ID = "us.anthropic.claude-opus-5"


def make_client(model_id: str, *, connect_timeout: float = 10,
                read_timeout: float = 300, sdk_max_attempts: int = 3):
    """Create an independent session/client with bounded transport retries."""
    import boto3
    from botocore.config import Config
    from dotenv import load_dotenv
    from mas_sae.models.claude import ClaudeClient

    load_dotenv()
    session = boto3.Session(
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        aws_session_token=os.getenv("aws_session_token"),
    )
    config = Config(connect_timeout=connect_timeout, read_timeout=read_timeout,
                    retries={"mode": "standard", "total_max_attempts": sdk_max_attempts})
    return ClaudeClient(session=session, model_id=model_id, config=config)


def main(argv: list[str] | None = None, *, client=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--n", type=int, help="Fixed number of requests (default: 5); each may repair once")
    mode.add_argument("--target-per-hop", type=int, help="Validated unique records to accept per selected subtype")
    parser.add_argument("--max-requests-per-hop", type=int, help="Required request cap per subtype in quota mode")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum active generation requests")
    parser.add_argument("--hop-types", nargs="+", choices=sorted(HOP_TYPES), default=list(HOP_TYPES))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, help="Seed for independent per-request sampling/shuffling")
    parser.add_argument("--original", type=Path, default=Path("data/MuSiQue/clean/train.json"),
                        help="MuSiQue JSONL used for filler paragraphs")
    parser.add_argument("--dedup-original", type=Path, nargs="+",
                        help="Original JSONL splits to index; quota mode defaults to --original")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=300)
    parser.add_argument("--sdk-max-attempts", type=int, default=3,
                        help="SDK transport attempts per invocation, including the first")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.sdk_max_attempts < 1:
        parser.error("concurrency and sdk-max-attempts must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.connect_timeout, args.read_timeout)):
        parser.error("timeouts must be finite and positive")
    if len(args.hop_types) != len(set(args.hop_types)):
        parser.error("hop-types must be unique")
    if args.target_per_hop is not None:
        if args.target_per_hop < 1 or args.max_requests_per_hop is None or args.max_requests_per_hop < args.target_per_hop:
            parser.error("quota mode requires positive target and max-requests-per-hop >= target")
    else:
        if args.max_requests_per_hop is not None:
            parser.error("max-requests-per-hop requires target-per-hop")
        args.n = 5 if args.n is None else args.n
        if args.n < 1:
            parser.error("n must be positive")
    if client is not None and args.concurrency != 1:
        parser.error("an injected client supports concurrency=1 only")
    dedup = args.dedup_original or ([args.original] if args.target_per_hop is not None else [])
    for path in (args.original, *dedup):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if args.output_dir is None:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex}"
        args.output_dir = Path("data/MuSiQue/synthetic/runs") / run_id / "generation"
    if args.output_dir.exists():
        parser.error("Output directory must be new to avoid overwriting prior results")
    transport = {"connect_timeout": args.connect_timeout, "read_timeout": args.read_timeout,
                 "sdk_max_attempts": args.sdk_max_attempts, "retry_mode": "standard"}

    def factory():
        return make_client(args.model_id, connect_timeout=args.connect_timeout,
                           read_timeout=args.read_timeout, sdk_max_attempts=args.sdk_max_attempts)

    def progress(report):
        state = "ok" if report["accepted"] else "FAILED"
        print(f"{report['request_id']} {state}", flush=True)

    summary = run_batch(
        args.output_dir, original=args.original, model_id=args.model_id,
        n=args.n, target_per_type=args.target_per_hop,
        max_requests_per_type=args.max_requests_per_hop, hop_types=args.hop_types,
        concurrency=args.concurrency, seed=args.seed, dedup_originals=dedup,
        client=client, client_factory=None if client is not None else factory,
        transport_settings=transport if client is None else None, progress=progress,
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["status"] == "quota_not_met" else 0


if __name__ == "__main__":
    raise SystemExit(main())
