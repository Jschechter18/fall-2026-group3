"""Export Josh's adoption-protocol conditions (#138) as Claude-validator inputs (issue #166).

Reuses ``natural_protocol_comparison.methods_from_records`` to rebuild the eight report rows
(six generated conditions plus the two randomized-order selections) from the saved, hash-
verified ``adoption_results/episodes.jsonl`` and the checksummed ``plan.json``. Each condition
becomes ``<output>/<slug>/sampled_episodes.jsonl`` with only what the validator reads; the order
the Solver actually saw is taken from the condition name for fixed-order rows and from the plan's
assignment for randomized rows. The old lexical outcome per episode (``adopted_target`` /
``retained_a1``) goes to ``manifest.json`` only, never to Claude's input.

Runs in the full project environment (the reused module imports the agents).

Usage:
    PYTHONPATH=src python scripts/export_adoption_conditions.py \\
        --results-root results/adoption_experiment \\
        --output results/claude_validation/samples/adoption_conditions_80q
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from mas_sae.experiments.adoption_experiment import load_sample
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.natural_protocol_comparison import METHODS, methods_from_records
from mas_sae.experiments.records import read_jsonl, write_jsonl

FEEDBACK_METHODS = frozenset(METHODS[:2])  # revision baseline and evidence verification: no candidates shown


def slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def shown_order(method: str, question_id: str, assignments: dict[str, str]) -> str | None:
    """candidate_order the Solver saw: from the condition name, or the assignment for randomized rows."""
    if method in FEEDBACK_METHODS:
        return None
    if "randomized" in method:
        return assignments[question_id]
    if method.endswith("A1 first"):
        return "a1_first"
    if method.endswith("critic first"):
        return "critic_first"
    raise ValueError(f"cannot tell the candidate order of condition {method!r}")


def validator_episode(row: dict, method: str, order: str | None) -> dict:
    """The fields the Claude validator reads; nothing that reveals the old lexical outcome."""
    episode = {"question_id": row["question_id"], "episode_id": f"{row['question_id']}__{slug(method)}",
               "question": row["question"], "solver_attempt_1": row["solver_attempt_1"],
               "critic_advocated_answer": row["critic_advocated_answer"],
               "solver_attempt_2": row["solver_attempt_2"], "critic_condition": "natural",
               "a2_presentation": "feedback" if order is None else "candidates", "condition": method}
    if order is not None:
        episode["candidate_order"] = order
    return episode


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=Path("results/adoption_experiment"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    root = args.results_root
    episodes_path = root / "adoption_results/episodes.jsonl"
    artifacts = {f["path"]: f["sha256"] for f in json.loads((root / "s3_artifacts.json").read_text())["files"]}
    actual = hashlib.sha256(episodes_path.read_bytes()).hexdigest()
    if artifacts.get(str(episodes_path)) != actual:
        raise SystemExit(f"{episodes_path} sha256 {actual[:12]}… is not the recorded S3 artifact")
    plan, plan_sha = load_sample(root / "adoption_results/plan.json", checksum_error="plan.json checksum mismatch")
    methods = methods_from_records(read_jsonl(episodes_path), plan["question_ids"], plan["order_assignments"])

    args.output.mkdir(parents=True)
    manifest = {"source": {"episodes": str(episodes_path), "episodes_sha256": actual,
                           "plan": str(root / "adoption_results/plan.json"), "plan_sha256": plan_sha},
                "conditions": []}
    for method, rows in methods:
        episodes = [validator_episode(r, method, shown_order(method, r["question_id"], plan["order_assignments"]))
                    for r in rows]
        folder = args.output / slug(method)
        folder.mkdir()
        write_jsonl(folder / "sampled_episodes.jsonl", episodes)
        manifest["conditions"].append({
            "condition": method, "slug": slug(method), "n": len(episodes),
            "a2_presentation": episodes[0]["a2_presentation"],
            "sampled_episodes_sha256": sha256_file(folder / "sampled_episodes.jsonl"),
            "lexical": [{"episode_id": e["episode_id"], "adopted_target": r["adopted_target"],
                         "retained_a1": r["retained_a1"]} for e, r in zip(episodes, rows)]})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for c in manifest["conditions"]:
        print(f"{c['slug']:42s} n={c['n']} {c['a2_presentation']}")


if __name__ == "__main__":
    main()
