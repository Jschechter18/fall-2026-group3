"""Paired comparison of a baseline and a treated collection of the same questions (#174, #156).

Use it for any one-change ablation collected twice with the same questions, seeds and candidate
orders, e.g. random vs related fillers (#174) or related fillers vs relation competitors (V1, #156).
The outcome rules and the three analyses (primary, intention-to-treat, treated-only) are fixed in
``mas_sae.evaluation.paired_ablation``; technical failures are reported apart and never counted as Out.

Each arm is a collection folder (``interactions.jsonl``, ``exclusions.jsonl``, ``sampled_questions.json``)
and the Claude label run made from it (``raw_results.jsonl``, ``summary.json``). It refuses arms whose
assigned questions, positions or candidate orders differ, labels made from other episodes, and arms
labeled with another prompt version or model. Inputs are hashed before reading and rechecked before writing.

Usage:
    PYTHONPATH=src python scripts/experiments/compare_paired_ablation.py \\
        --baseline <baseline collection>/train <baseline labels> \\
        --treated <treated collection>/train <treated labels> \\
        [--treated-ids <apply output>/treated_question_ids.json] --output <dir>
"""

import argparse
import json
from pathlib import Path

from mas_sae.evaluation.paired_ablation import analyses, arm_outcomes, breakdown, by_question, correctness_cell, hop_of
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file, write_json_atomic
from mas_sae.experiments.records import read_jsonl, write_csv

COLLECTION_FILES = ("interactions.jsonl", "exclusions.jsonl", "sampled_questions.json")
LABEL_FILES = ("raw_results.jsonl", "summary.json")


def arm_paths(collection: Path, labels: Path) -> list[Path]:
    return [collection / name for name in COLLECTION_FILES] + [labels / name for name in LABEL_FILES]


def read_arm(name: str, collection: Path, labels: Path) -> dict:
    """One arm's assignment, episodes and outcomes, checked against its own label run."""
    summary = json.loads((labels / "summary.json").read_text(encoding="utf-8"))
    if summary.get("episodes_sha256") != sha256_file(collection / "interactions.jsonl"):
        raise ValueError(f"{name}: the labels in {labels} were not made from {collection / 'interactions.jsonl'}")
    sampled = sorted(json.loads((collection / "sampled_questions.json").read_text(encoding="utf-8")),
                     key=lambda row: row["position"])
    assignment = [(row["position"], row["question_id"], row["candidate_order"]) for row in sampled]
    assigned = [qid for _, qid, _ in assignment]
    interactions = read_jsonl(collection / "interactions.jsonl")
    outcomes = arm_outcomes(assigned, interactions, read_jsonl(collection / "exclusions.jsonl"),
                            read_jsonl(labels / "raw_results.jsonl"))
    return {"assignment": assignment, "assigned": assigned, "outcomes": outcomes,
            "episodes": by_question(interactions, f"{name} interactions"),
            "labeling": {key: summary.get(key) for key in ("prompt_version", "model_id", "effort", "max_tokens")}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, nargs=2, metavar=("COLLECTION", "LABELS"), required=True)
    parser.add_argument("--treated", type=Path, nargs=2, metavar=("COLLECTION", "LABELS"), required=True)
    parser.add_argument("--treated-ids", type=Path, help="questions whose change was applied (treated-only analysis)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    paths = [*arm_paths(*args.baseline), *arm_paths(*args.treated), *([args.treated_ids] if args.treated_ids else [])]
    hashes = {str(path): sha256_file(path) for path in paths}
    arms = {name: read_arm(name, *getattr(args, name)) for name in ("baseline", "treated")}
    baseline, treated = arms["baseline"], arms["treated"]
    if baseline["assignment"] != treated["assignment"]:
        raise ValueError("the arms were assigned different questions, positions or candidate orders")
    if baseline["labeling"]["prompt_version"] != treated["labeling"]["prompt_version"] \
            or baseline["labeling"]["model_id"] != treated["labeling"]["model_id"]:
        raise ValueError(f"the arms were labeled differently: {baseline['labeling']} vs {treated['labeling']}")
    treated_ids = json.loads(args.treated_ids.read_text(encoding="utf-8")) if args.treated_ids else None
    assigned = baseline["assigned"]
    result = analyses(baseline["outcomes"], treated["outcomes"], assigned, treated_ids)
    descriptives = {name: {"correctness_cells": breakdown(arm["outcomes"], arm["episodes"],
                                                          lambda q, e: correctness_cell(e)),
                           "hop_types": breakdown(arm["outcomes"], arm["episodes"], lambda q, e: hop_of(q))}
                    for name, arm in arms.items()}
    changed = [path for path in paths if sha256_file(path) != hashes[str(path)]]
    if changed:
        raise RuntimeError(f"inputs changed while reading: {[str(path) for path in changed]}")

    chosen = set(treated_ids or [])
    rows = [{"question_id": qid, "hop_type": hop_of(qid),
             "baseline": baseline["outcomes"][qid], "treated": treated["outcomes"][qid],
             "baseline_cell": correctness_cell(baseline["episodes"].get(qid)),
             "treated_cell": correctness_cell(treated["episodes"].get(qid)),
             "change_applied": (qid in chosen) if treated_ids is not None else None} for qid in assigned]
    args.output.mkdir(parents=True)
    write_csv(args.output / "per_question.csv", rows)
    write_json_atomic(args.output / "summary.json", {
        "inputs_sha256": hashes, "assigned": len(assigned), "labeling": baseline["labeling"], **result,
        "descriptives": descriptives})
    for name in ("primary", "intention_to_treat", "treated_only"):
        if name in result:
            r = result[name]
            interval = "n/a" if r["change_ci95_pp"] is None else "[{:.1f}, {:.1f}]".format(*r["change_ci95_pp"])
            print(f"{name:>18}  n={r['n']:>4}  A+R {r['baseline']['binary']} -> {r['treated']['binary']}  "
                  f"gained {r['discordant']['gained']} lost {r['discordant']['lost']}  "
                  f"change {r['change_pp'] if r['change_pp'] is None else round(r['change_pp'], 1)} pp {interval}  "
                  f"exact McNemar p={r['exact_mcnemar_p']:.2g}")
    print(f"technical failures: {result['technical_failures']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
