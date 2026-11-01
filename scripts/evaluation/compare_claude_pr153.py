"""Compare a Claude validation run on the candidates pilot with the approved PR #153 labels.

Refuses a run whose input file is not byte-identical to the sampled episodes (sha256 recorded
in selection.json). Reports the 200-episode evaluation set and the 50-episode challenge set separately:
confusion matrix, per-class precision/recall/F1, Cohen's kappa, adoption-vs-retention side
agreement, and (evaluation only) accuracy re-weighted to the train population, because the
sample is class-stratified. Team-reviewed challenge cases are also compared with the team's
own labels. Every disagreement goes to ``disagreements.csv`` for human adjudication; PR #153
is a reference, not ground truth.

Usage:
    PYTHONPATH=src python scripts/evaluation/compare_claude_pr153.py \\
        --run results/claude_validation/candidates_pilot250_run1 \\
        --selection results/claude_validation/samples/candidates_pilot250_seed20261029/selection.json \\
        --output results/claude_validation/candidates_pilot250_comparison \\
        [--price-input-per-mtok X --price-output-per-mtok Y]
"""

import argparse
import csv
import json
from pathlib import Path

from mas_sae.evaluation.claude_agreement import (
    binary_side_agreement,
    cohen_kappa,
    confusion,
    per_class,
    weighted_accuracy,
)
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.records import read_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="Claude validation run directory")
    parser.add_argument("--selection", type=Path, required=True, help="pilot selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--price-input-per-mtok", type=float, default=None,
                        help="USD per million input tokens (current Bedrock price); omit to skip cost")
    parser.add_argument("--price-output-per-mtok", type=float, default=None)
    return parser.parse_args(argv)


def check_run_matches_selection(summary: dict, selection: dict) -> None:
    """The run must have labeled exactly the sampled file: same sha256 then and now."""
    expected = selection.get("sampled_episodes_sha256")
    if not expected:
        raise SystemExit("selection.json has no sampled_episodes_sha256; regenerate it with sample_candidate_pilot.py")
    if summary.get("episodes_sha256") != expected:
        raise SystemExit(f"run input sha256 {str(summary.get('episodes_sha256'))[:12]}… differs from the "
                         f"sampled episodes {expected[:12]}…")
    if sha256_file(Path(summary["episodes_path"])) != expected:
        raise SystemExit(f"{summary['episodes_path']} has changed since it was sampled")


def report(pairs, population=None):
    matrix = confusion(pairs)
    out = {"n": len(pairs), "confusion": matrix, "per_class": per_class(matrix),
           "cohen_kappa": cohen_kappa(matrix), "binary_side_agreement": binary_side_agreement(pairs),
           "technical_errors": sum(row["error"] for row in matrix.values())}
    if population is not None:
        out["population_weighted_accuracy"] = weighted_accuracy(matrix, population)
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_output_available(args.output)
    selection = json.loads(args.selection.read_text())
    rows = {r["episode_id"]: r for r in read_jsonl(args.run / "raw_results.jsonl")}
    summary = json.loads((args.run / "summary.json").read_text())
    episodes = selection["episodes"]
    check_run_matches_selection(summary, selection)
    if {e["episode_id"] for e in episodes} != set(rows):
        raise SystemExit("run results and selection cover different episodes")

    pred = {eid: rows[eid].get("behavior_label") for eid in rows}
    evaluation = [(e["pr153_label"], pred[e["episode_id"]]) for e in episodes if e["set"] == "evaluation"]
    challenge = [(e["pr153_label"], pred[e["episode_id"]]) for e in episodes if e["set"].startswith("challenge")]
    team = [(e["team_label"], pred[e["episode_id"]]) for e in episodes
            if e["set"] == "challenge:team_reviewed" and e.get("team_label")]

    tokens_in, tokens_out = summary["total_input_tokens"], summary["total_output_tokens"]
    cost = (tokens_in / 1e6 * args.price_input_per_mtok + tokens_out / 1e6 * args.price_output_per_mtok
            if args.price_input_per_mtok is not None and args.price_output_per_mtok is not None else None)
    metrics = {
        "note": "Reference = approved PR #153 labels (not ground truth). The evaluation set is "
                "class-stratified (50 per label); only population_weighted_accuracy estimates the train mix.",
        "evaluation_vs_pr153": report(evaluation, selection["train_population_label_counts"]),
        "challenge_vs_pr153": report(challenge),
        "team_reviewed_vs_team_labels": report(team),
        "run": {"prompt_version": summary.get("prompt_version"), "model_id": summary.get("model_id"),
                "episodes": summary["n_episodes"], "input_tokens": tokens_in, "output_tokens": tokens_out,
                "wall_clock_seconds": summary.get("wall_clock_seconds"), "estimated_cost_usd": cost,
                "prices_per_mtok": [args.price_input_per_mtok, args.price_output_per_mtok]},
    }

    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    source = {r["episode_id"]: r for r in read_jsonl(Path(summary["episodes_path"]))}
    with open(args.output / "disagreements.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["set", "question_id", "episode_id", "pr153_label", "team_label", "claude_label",
                  "pr153_flags", "candidate_order", "A1", "C1", "A2", "claude_reason", "exclusion_reason",
                  "adjudication"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for e in episodes:
            row, ep = rows[e["episode_id"]], source[e["episode_id"]]
            if row.get("behavior_label") != e["pr153_label"]:
                writer.writerow({"set": e["set"], "question_id": e["question_id"], "episode_id": e["episode_id"],
                                 "pr153_label": e["pr153_label"], "team_label": e.get("team_label") or "",
                                 "claude_label": row.get("behavior_label") or "", "pr153_flags": ";".join(e["pr153_flags"]),
                                 "candidate_order": e.get("candidate_order"), "A1": ep["solver_attempt_1"],
                                 "C1": ep["critic_advocated_answer"], "A2": ep["solver_attempt_2"],
                                 "claude_reason": row.get("claude_reason") or "",
                                 "exclusion_reason": row.get("exclusion_reason") or "", "adjudication": ""})
    ev = metrics["evaluation_vs_pr153"]
    print(json.dumps({"evaluation_kappa": ev["cohen_kappa"], "evaluation_weighted_accuracy": ev.get("population_weighted_accuracy"),
                      "challenge_kappa": metrics["challenge_vs_pr153"]["cohen_kappa"], **metrics["run"]}, indent=2))


if __name__ == "__main__":
    main()
