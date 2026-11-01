"""Summarize Claude's labels over a PR #153 label population (issue #166).

Combines the Claude runs that together cover a population selected by
``select_pr153_population.py`` (e.g. the pilot plus the full pass) through
``claude_milestones.combine_runs``, which refuses relabeled episodes, changed source files
and mixed prompt versions. Reports Claude's four-class distribution with 95% Wilson
intervals, the same broken down by PR #153's reason, technical errors, and a clearly
hypothetical partition balance if Claude's Adoption/Retention were added to PR #153's.
PR #153's labels are never modified.

Usage:
    PYTHONPATH=src python scripts/summarize_population_labels.py \\
        --selection results/claude_validation/samples/candidates_train_out_population/selection.json \\
        --runs results/claude_validation/candidates_pilot250_opus5_run1 \\
               results/claude_validation/candidates_train_out_population_opus5_run1 \\
        --output results/claude_validation/candidates_train_out_population_summary
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mas_sae.evaluation.candidate_selection import CLASSES
from mas_sae.evaluation.claude_agreement import label_distribution
from mas_sae.evaluation.claude_milestones import combine_runs
from mas_sae.experiments.artifacts import ensure_output_available

APPROVED_COUNTS = Path("results/experiments/candidate_final_labels_20261027/counts.json")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--counts", type=Path, default=APPROVED_COUNTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    selection = json.loads(args.selection.read_text())
    population = {e["episode_id"]: e for e in selection["episodes"]}
    rows, _, totals = combine_runs(args.runs)
    labeled = {r["episode_id"]: r for r in rows if r["episode_id"] in population}
    missing = sorted(set(population) - set(labeled))
    if missing:
        raise SystemExit(f"{len(missing)} population episode(s) have no Claude result, e.g. {missing[:3]}")

    claude = {eid: r.get("behavior_label") for eid, r in labeled.items()}
    by_reason = defaultdict(list)
    for eid, e in population.items():
        by_reason[(e.get("pr153_reason") or "none").split(" [")[0]].append(claude[eid])

    pop = selection["population"]
    approved = json.loads(args.counts.read_text())["labels_by_partition"][pop["partition"]]
    moved = Counter(claude.values())
    adoption, retention = approved["adoption"], approved["retention"]
    summary = {
        "population": pop,
        "claude_distribution": label_distribution(claude.values()),
        "by_pr153_reason": {reason: label_distribution(labels) for reason, labels in
                            sorted(by_reason.items(), key=lambda kv: -len(kv[1]))},
        "pr153_partition_counts": approved,
        "hypothetical_if_added": {
            "note": "Not applied: PR #153 labels are unchanged; unresolved disagreements stay out.",
            "adoption": adoption + moved["adoption"], "retention": retention + moved["retention"],
            "adoption_share_before": adoption / (adoption + retention),
            "adoption_share_after": (adoption + moved["adoption"]) / (adoption + retention + moved["adoption"] + moved["retention"]),
        },
        "runs": totals["runs"],
        "billed": {k: totals[k] for k in ("input_tokens", "output_tokens", "wall_clock_seconds")},
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    d = summary["claude_distribution"]
    print(json.dumps({label: f"{d[label]['count']} ({d[label]['share']:.1%})" for label in (*CLASSES, "error")}, indent=2))
    h = summary["hypothetical_if_added"]
    print(f"adoption share {h['adoption_share_before']:.1%} -> {h['adoption_share_after']:.1%} (hypothetical, not applied)")


if __name__ == "__main__":
    main()
