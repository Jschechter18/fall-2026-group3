"""Break the approved PR #153 labels down by the candidate order the Solver saw (issue #166).

The candidates run assigned each question an order before A2 (``a1_first``: A1 is Candidate A;
``critic_first``: the Critic's recommendation is). This reports, per frozen partition and for all
of them together, PR #153's four-class counts for each order with 95% Wilson intervals, and the
same split by the saved gold-answer correctness of A1 and the Critic's recommendation (e.g. how
often a useful correction - A1 wrong, Critic right - is adopted). Correctness is the collection's
string match against the gold answer, so it is approximate. The per-partition totals must
reproduce ``counts.json`` exactly. Labels and partitions are read, never written.

Usage:
    PYTHONPATH=src python scripts/summarize_order_labels.py \\
        --run results/collection/natural_4b_full9_candidates \\
        --partitions results/behavior/behavior_v012_official_20261011/partitions.csv \\
        --labels <dir>/final_labels.jsonl \\
        --output results/claude_validation/candidates_order_breakdown
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mas_sae.evaluation.candidate_selection import (
    CLASSES,
    PARTITIONS,
    load_verified_labels,
    read_run_sources,
)
from mas_sae.evaluation.claude_agreement import label_distribution
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file

APPROVED_COUNTS = Path("results/experiments/candidate_final_labels_20261027/counts.json")
ORDERS = ("a1_first", "critic_first")
SCOPES = (*PARTITIONS, "all")
CASES = ("useful_correction", "harmful_correction", "both_right", "both_wrong", "unscored")


def correctness_case(source: dict) -> str:
    """Saved correctness of A1 and of the Critic's recommendation, as one case name."""
    a1, critic = source.get("solver_attempt_1_correct"), source.get("critic_feedback_correct")
    if a1 is None or critic is None:
        return "unscored"
    return {(False, True): "useful_correction", (True, False): "harmful_correction",
            (True, True): "both_right", (False, False): "both_wrong"}[(bool(a1), bool(critic))]


def join_orders(labels: list[dict], sources: dict[str, dict]) -> tuple[list[dict], dict[str, Counter]]:
    """One row per collected, partitioned label with the order and correctness case it was collected under.

    Returns ``(rows, not_collected)``; ``not_collected[partition]`` counts the labels of questions with
    no interaction. A collected label without its interaction, or an unknown order, is refused.
    """
    rows, not_collected = [], defaultdict(Counter)
    for label in labels:
        if label["partition"] is None:
            continue
        source = sources.get(label["question_id"])
        if source is None:
            if label["collected"]:
                raise SystemExit(f"{label['question_id']} is labeled collected but has no interaction")
            not_collected[label["partition"]][label["label"]] += 1
            continue
        if source.get("a2_presentation") != "candidates" or source.get("candidate_order") not in ORDERS:
            raise SystemExit(f"{label['question_id']}: not a candidates episode with a known order "
                             f"({source.get('a2_presentation')!r}, {source.get('candidate_order')!r})")
        rows.append({"question_id": label["question_id"], "partition": label["partition"], "label": label["label"],
                     "order": source["candidate_order"], "case": correctness_case(source)})
    return rows, not_collected


def reconcile(rows: list[dict], not_collected: dict[str, Counter], approved: dict) -> None:
    """Joined plus not-collected labels must equal the approved per-partition counts."""
    for partition in PARTITIONS:
        got = Counter(r["label"] for r in rows if r["partition"] == partition) + not_collected.get(partition, Counter())
        want = approved["labels_by_partition"][partition]
        if any(got[c] != want[c] for c in CLASSES):
            raise SystemExit(f"{partition}: joined counts {dict(got)} differ from counts.json {want}")


def breakdown(rows: list[dict]) -> dict:
    """``{scope: {order or "both": label_distribution}}`` and the same per correctness case."""
    by_order, by_case = {}, {}
    for scope in SCOPES:
        scoped = [r for r in rows if scope == "all" or r["partition"] == scope]
        by_order[scope] = {o: label_distribution(r["label"] for r in scoped if o == "both" or r["order"] == o)
                           for o in (*ORDERS, "both")}
        by_case[scope] = {case: {o: label_distribution(r["label"] for r in scoped
                                                       if r["case"] == case and (o == "both" or r["order"] == o))
                                 for o in (*ORDERS, "both")} for case in CASES}
    return {"by_order": by_order, "by_correctness": by_case}


def adoption_line(d: dict) -> str:
    """``A=… R=… adoption share x% [lo, hi]`` from a label_distribution."""
    b = d["retention_share_of_binary"]
    if not b["binary"]:
        return f"A={d['adoption']['count']} R={d['retention']['count']}"
    lo, hi = b["ci95"]
    return (f"A={d['adoption']['count']:5d} R={d['retention']['count']:5d}  adoption share "
            f"{1 - b['share']:.1%} [{1 - hi:.1%}, {1 - lo:.1%}]")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--counts", type=Path, default=APPROVED_COUNTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    labels, _, approved = load_verified_labels(args.labels, args.counts, args.partitions)
    rows, not_collected = join_orders(labels, read_run_sources(args.run))
    reconcile(rows, not_collected, approved)
    summary = {
        "policy_version": approved["policy_version"],
        "labels_sha256": approved["labels_sha256"],
        **breakdown(rows),
        "not_collected": {p: dict(c) for p, c in sorted(not_collected.items())},
        "input_hashes": {str(p): sha256_file(p) for p in (args.labels, args.counts, args.partitions,
                                                          args.run / "train/interactions.jsonl",
                                                          args.run / "validation/interactions.jsonl")},
        "note": "PR #153 labels, unchanged. Order was assigned before A2. Correctness is the collection's "
                "string match against the gold answer (approximate); no labeler saw the gold answer.",
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for scope in ("train", "all"):
        for order in (*ORDERS, "both"):
            print(f"{scope:5s} {order:12s} {adoption_line(summary['by_order'][scope][order])}")
    for case in ("useful_correction", "harmful_correction"):
        for order in ORDERS:
            print(f"train {case:18s} {order:12s} {adoption_line(summary['by_correctness']['train'][case][order])}")


if __name__ == "__main__":
    main()
