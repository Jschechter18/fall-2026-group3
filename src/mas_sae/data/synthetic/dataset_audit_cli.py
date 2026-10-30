"""Audit an existing generated dataset without regenerating or editing questions."""
import argparse
import json
from pathlib import Path

from .audit import audit_dataset
from .clients import DEFAULT_MODEL_ID, make_client


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Generated accepted.jsonl; audit is report-only")
    parser.add_argument("--output-dir", type=Path, help="New directory; default: input's parent/dataset_audit")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-prompt-chars", type=int, default=100000)
    parser.add_argument("--max-neighbor-batches", type=int, default=200)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="Write local statistics and comparison plan without API calls")
    args = parser.parse_args(argv)
    summary = audit_dataset(
        args.input, args.output_dir or args.input.parent / "dataset_audit", model_id=args.model_id,
        client_factory=lambda: make_client(args.model_id), concurrency=args.concurrency,
        batch_size=args.batch_size, max_prompt_chars=args.max_prompt_chars,
        max_neighbor_batches=args.max_neighbor_batches, max_attempts=args.max_attempts,
        max_tokens=args.max_tokens, dry_run=args.dry_run,
        transport_settings={"connect_timeout": 10, "read_timeout": 300,
                            "sdk_max_attempts": 3, "retry_mode": "standard"},
        progress=lambda report: print(f"{report['batch_id']}: {report['status']}", flush=True),
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] in {"completed", "planned"} else 1
