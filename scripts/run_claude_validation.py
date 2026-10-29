"""Run Claude behavioral validation on Natural-condition episodes (issue #155).

Reads episodes from a partition-exported interactions.jsonl (records carry
``partition``), keeps only ``critic_condition == "natural"`` episodes of one
development partition, sends each A1/C1/A2 triplet to Claude through Amazon
Bedrock, and writes probe-ready binary labels (Adoption = 1, Retention = 0)
alongside an exclusion audit file.

The Natural condition is fixed; there is no option to process Controlled
episodes.  Sealed partitions (test, intervention) cannot be selected.  The
script reports how many records were loaded and kept at each step, so no
records are silently discarded.

With ``--limit N`` the first N complete episodes in file order are taken, so
a dry run and the real run select the same episodes.  The output directory
must not exist; earlier results are never overwritten.

Requires AWS Bedrock credentials in ``.env`` (see README).  Use ``--dry-run``
to preview without making API calls.

Usage:
    PYTHONPATH=src python scripts/run_claude_validation.py \\
        --episodes results/collection/natural_4b_full9_partitioned/train/interactions.jsonl \\
        --output results/claude_validation/pilot_10 \\
        --limit 10 \\
        --dry-run

    # Real pilot (after authorization):
    PYTHONPATH=src python scripts/run_claude_validation.py \\
        --episodes results/collection/natural_4b_full9_partitioned/train/interactions.jsonl \\
        --output results/claude_validation/pilot_10 \\
        --limit 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

from mas_sae.evaluation.claude_validator import (
    DEFAULT_MAX_ATTEMPTS,
    PROMPT_VERSION,
    alignment_exclusions,
    is_technical_error,
    build_labeling_prompt,
    export_binary_labels,
    export_exclusions,
    missing_required_field,
    validate_batch,
)
from mas_sae.experiments.artifacts import (
    ensure_output_available,
    get_git_commit,
    get_git_diff_sha256,
    sha256_file,
)
from mas_sae.experiments.records import read_jsonl
from mas_sae.models.claude import ClaudeClient

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Issue #155 labels Natural episodes only.
CONDITION = "natural"

# mas_sae.data.production.DEVELOPMENT_PARTITIONS (not imported: it pulls in torch).
DEVELOPMENT_PARTITIONS = ("train", "validation")

# Fields recorded per selected episode so the run can be traced to its source rows.
SELECTION_FIELDS = (
    "episode_id", "question_id", "partition", "source_split",
    "source_run", "source_activation_index",
    "attempt1_activation_index", "attempt2_activation_index",
    "sae_attempt1_index", "sae_attempt2_index",
)


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
        help="Process only the first N complete episodes (for pilot runs).",
    )
    parser.add_argument(
        "--partition", choices=DEVELOPMENT_PARTITIONS, default="train",
        help="Frozen partition to label (default: train). Sealed partitions are refused.",
    )
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID,
        help=f"Bedrock model ID (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
        help=f"Claude calls per episode when a response cannot be parsed (default: {DEFAULT_MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--retry-technical-errors-from", type=Path, default=None, metavar="RUN_DIR",
        help="Process only episodes that ended in a technical error (API, parse, truncated) "
             "in RUN_DIR, an earlier run over the same --episodes file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build prompts and print counts without making API calls.",
    )
    return parser.parse_args(argv)


def load_episodes(path: Path) -> list[dict]:
    """Load episodes from a JSONL file or from a directory containing one."""
    if path.is_dir():
        path = path / "interactions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Episodes file not found: {path}")
    return read_jsonl(path)


def filter_by_condition(
    episodes: list[dict],
    condition: str,
) -> tuple[list[dict], dict[str, int]]:
    """Keep only episodes matching *condition* and report what was filtered.

    Returns ``(matching, condition_counts)`` where *condition_counts* maps
    every observed ``critic_condition`` value (including ``"MISSING"``) to
    the number of episodes that had it.
    """
    counts: dict[str, int] = {}
    matching: list[dict] = []
    for ep in episodes:
        value = ep.get("critic_condition")
        key = value if isinstance(value, str) else "MISSING"
        counts[key] = counts.get(key, 0) + 1
        if key == condition:
            matching.append(ep)
    return matching, counts


def filter_by_partition(episodes: list[dict], partition: str) -> list[dict]:
    """Keep episodes of *partition*; refuse records with no ``partition`` field.

    Raw collection folders are MuSiQue source splits (``train/`` also holds
    test and intervention questions), so only partition-exported records,
    which carry ``partition``, can be filtered safely.
    """
    unpartitioned = [ep.get("episode_id") for ep in episodes if "partition" not in ep]
    if unpartitioned:
        raise ValueError(
            f"{len(unpartitioned)} episode(s) have no 'partition' field "
            f"(e.g. {unpartitioned[0]!r}). Use the partition export "
            "(e.g. natural_4b_full9_partitioned/<partition>/interactions.jsonl).")
    return [ep for ep in episodes if ep["partition"] == partition]


def select_episodes(episodes: list[dict], limit: int | None) -> tuple[list[dict], int]:
    """The first *limit* complete episodes in file order (all episodes when None).

    Returns ``(selected, n_incomplete_skipped)``.  Without a limit nothing is
    skipped: incomplete episodes are recorded as exclusions by the validator.
    Raises if a question_id repeats, since labels and probe exclusions are
    keyed per question.
    """
    if limit is None:
        selected, skipped = episodes, 0
    else:
        complete = [ep for ep in episodes if missing_required_field(ep) is None]
        selected, skipped = complete[:limit], len(episodes) - len(complete)
    qids = [ep.get("question_id") for ep in selected]
    if len(set(qids)) != len(qids):
        raise ValueError("Selected episodes repeat a question_id.")
    return selected, skipped


def selection_record(episodes: list[dict]) -> list[dict]:
    """Identity and activation indices of each selected episode, as found in the source."""
    return [{f: ep[f] for f in SELECTION_FIELDS if f in ep} for ep in episodes]


def technical_error_ids(run_dir: Path, episodes_sha256: str) -> set[str]:
    """episode_ids that ended in a technical error in *run_dir*, which must have read the same source."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("episodes_sha256") != episodes_sha256:
        raise ValueError(f"{run_dir} was run on a different episodes file "
                         f"(sha256 {summary.get('episodes_sha256')}, now {episodes_sha256}).")
    return {r["episode_id"] for r in read_jsonl(run_dir / "raw_results.jsonl") if is_technical_error(r)}


def main(argv=None) -> None:
    args = parse_args(argv)

    if not args.dry_run:
        ensure_output_available(args.output)

    episodes_path = args.episodes / "interactions.jsonl" if args.episodes.is_dir() else args.episodes
    all_episodes = load_episodes(episodes_path)
    episodes_sha256 = sha256_file(episodes_path)
    print(f"Loaded {len(all_episodes)} episode(s) from {episodes_path}")
    print(f"Source sha256: {episodes_sha256}")

    natural, condition_counts = filter_by_condition(all_episodes, CONDITION)
    print(f"Condition filter (critic_condition={CONDITION!r}):")
    for cond, count in sorted(condition_counts.items()):
        marker = " <-- selected" if cond == CONDITION else ""
        print(f"  {cond}: {count}{marker}")

    try:
        in_partition = filter_by_partition(natural, args.partition)
        print(f"Partition filter (partition={args.partition!r}): "
              f"{len(in_partition)} of {len(natural)} natural episode(s)")
        if args.retry_technical_errors_from is not None:
            retry_ids = technical_error_ids(args.retry_technical_errors_from, episodes_sha256)
            in_partition = [ep for ep in in_partition if ep["episode_id"] in retry_ids]
            print(f"Retry filter: {len(in_partition)} episode(s) with technical errors in "
                  f"{args.retry_technical_errors_from}")
        episodes, n_skipped = select_episodes(in_partition, args.limit)
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not episodes:
        print(f"\nERROR: No natural episodes in partition {args.partition!r}.", file=sys.stderr)
        sys.exit(1)
    if args.limit is not None:
        print(f"Limit applied: first {len(episodes)} complete episode(s) "
              f"({n_skipped} incomplete episode(s) in the partition were not eligible)")
    selection = selection_record(episodes)

    if args.dry_run:
        print(f"\n--- DRY RUN (prompt_version={PROMPT_VERSION}) ---")
        print(f"Model: {args.model_id}")
        print(f"Condition: critic_condition={CONDITION!r}, partition={args.partition!r}")
        print(f"Episodes to process: {len(episodes)}")
        for item in selection:
            print(f"  {json.dumps(item)}")
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
    # Free STS check, so stale credentials fail before an output folder is created.
    try:
        session.client("sts", region_name="us-east-1").get_caller_identity()
    except (ClientError, NoCredentialsError) as exc:
        print(f"\nERROR: AWS credentials rejected ({exc}). Refresh them in .env and rerun.",
              file=sys.stderr)
        sys.exit(1)
    client = ClaudeClient(session=session, model_id=args.model_id)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True)
    (output_dir / "selected_episodes.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8",
    )

    started = time.perf_counter()
    try:
        summary = validate_batch(
            client, episodes,
            output_path=output_dir / "raw_results.jsonl",
            max_attempts=args.max_attempts,
        )
    except (ClientError, NoCredentialsError) as exc:
        print(f"\nERROR: stopped on a credential/permission error ({exc}). Rows completed "
              f"before it are in {output_dir / 'raw_results.jsonl'}; refresh credentials "
              "and rerun into a new --output.", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()
    wall_clock_seconds = round(time.perf_counter() - started, 2)

    results = summary.pop("results")
    export_binary_labels(results, output_dir / "binary_labels.csv")
    export_exclusions(results, output_dir / "exclusions.jsonl")
    (output_dir / "excluded_episodes.json").write_text(
        json.dumps(alignment_exclusions(results), indent=2) + "\n", encoding="utf-8",
    )

    summary["model_id"] = args.model_id
    summary["prompt_version"] = PROMPT_VERSION
    summary["episodes_path"] = str(episodes_path)
    summary["episodes_sha256"] = episodes_sha256
    summary["condition_filter"] = CONDITION
    summary["condition_counts"] = condition_counts
    summary["partition"] = args.partition
    summary["limit"] = args.limit
    summary["n_incomplete_skipped"] = n_skipped
    summary["wall_clock_seconds"] = wall_clock_seconds
    summary["retry_technical_errors_from"] = (str(args.retry_technical_errors_from)
                                              if args.retry_technical_errors_from else None)
    summary["git_commit"] = get_git_commit()
    try:
        summary["uncommitted_changes_sha256"] = get_git_diff_sha256()
    except RuntimeError:
        summary["uncommitted_changes_sha256"] = "unknown"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\nDone. {summary['n_episodes']} episodes processed.")
    print(f"Labels: {summary['label_counts']}")
    print(f"Label sources: {summary['label_sources']}")
    print(f"Errors by reason: {summary['error_reasons']} "
          f"(technical: {summary['n_technical_errors']}, retried episodes: {summary['n_retried']})")
    print(f"Tokens: {summary['total_input_tokens']} in / {summary['total_output_tokens']} out")
    print(f"Wall clock: {wall_clock_seconds}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
