"""Combine Claude validation runs over one cohort into a milestone (e.g. 500, 1,000).

Runs are given in order; a later run may only replace rows that ended in a technical
error (retries).  Writes the combined probe-ready outputs and a summary:

    summary.json, binary_labels.csv, exclusions.jsonl, excluded_episodes.json

Usage:
    PYTHONPATH=src python scripts/summarize_claude_validation.py \\
        --runs results/claude_validation/natural_hop100_seed20261029_run1 \\
               results/claude_validation/natural_hop100_seed20261029_retry1 \\
               results/claude_validation/natural_topup400_to500_seed20261029_run1 \\
        --output results/claude_validation/milestones/natural_500
"""

import argparse
import json
from pathlib import Path

from mas_sae.evaluation.claude_milestones import combine_runs, milestone_summary
from mas_sae.evaluation.claude_validator import (
    alignment_exclusions,
    export_binary_labels,
    export_exclusions,
)
from mas_sae.experiments.artifacts import ensure_output_available


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    rows, sources, totals = combine_runs(args.runs)
    summary = {**milestone_summary(rows, sources), "billed": totals}

    args.output.mkdir(parents=True)
    export_binary_labels(rows, args.output / "binary_labels.csv")
    export_exclusions(rows, args.output / "exclusions.jsonl")
    (args.output / "excluded_episodes.json").write_text(
        json.dumps(alignment_exclusions(rows), indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "billed"}, indent=2))
    print(f"Billed tokens: {totals['input_tokens']} in / {totals['output_tokens']} out; "
          f"wall clock {totals['wall_clock_seconds']:.1f}s over {len(totals['runs'])} run(s)")


if __name__ == "__main__":
    main()
