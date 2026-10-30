"""Select every frozen-partition episode PR #153 gives one label, for a full Claude pass (issue #166).

For example, all collected *train* episodes PR #153 labels ``out``: Claude labels each one so we
can measure how many it would place in Adoption/Retention. Episodes already labeled by earlier
Claude runs (``--already-labeled RUN_DIR``, same prompt version) are skipped, so none is paid for
twice; their results are combined with the new run at analysis time.

Writes ``sampled_episodes.jsonl`` (unmodified source records, no labels - the only file Claude
reads) and ``selection.json`` (population definition, PR #153 label/reason/flags per episode,
input hashes and the sampled file's sha256).

Usage:
    PYTHONPATH=src python scripts/select_pr153_population.py \\
        --run results/collection/natural_4b_full9_candidates \\
        --partitions results/behavior/behavior_v012_official_20261011/partitions.csv \\
        --labels <dir>/final_labels.jsonl --label out --partition train \\
        --already-labeled results/claude_validation/candidates_pilot250_opus5_run1 \\
        --output results/claude_validation/samples/candidates_train_out_population
"""

import argparse
import json
from pathlib import Path

from mas_sae.evaluation.candidate_selection import (
    CLASSES,
    load_verified_labels,
    read_run_sources,
)
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.records import read_jsonl, write_jsonl

APPROVED_COUNTS = Path("results/experiments/candidate_final_labels_20261027/counts.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--counts", type=Path, default=APPROVED_COUNTS)
    parser.add_argument("--label", choices=CLASSES, required=True, help="PR #153 label defining the population")
    parser.add_argument("--partition", choices=("train", "validation"), default="train")
    parser.add_argument("--already-labeled", type=Path, nargs="*", default=[],
                        help="earlier Claude run dirs; their behaviorally labeled episodes are skipped")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def already_labeled(run_dirs: list[Path]) -> tuple[set[str], list[dict]]:
    """Episode ids with a behavioral label in earlier runs (technical errors are re-labeled)."""
    ids, runs, versions = set(), [], set()
    for run in run_dirs:
        summary = json.loads((run / "summary.json").read_text())
        versions.add(summary.get("prompt_version"))
        rows = read_jsonl(run / "raw_results.jsonl")
        ids |= {r["episode_id"] for r in rows if r.get("behavior_label")}
        runs.append({"run_dir": str(run), "prompt_version": summary.get("prompt_version"),
                     "models_returned": summary.get("models_returned"),
                     "episodes_sha256": summary.get("episodes_sha256")})
    if len(versions) > 1:
        raise SystemExit(f"--already-labeled runs mix prompt versions {sorted(map(str, versions))}")
    return ids, runs


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_output_available(args.output)
    labels, frozen, approved = load_verified_labels(args.labels, args.counts, args.partitions)
    population = sorted((r for r in labels if r["partition"] == args.partition and r["collected"]
                         and r["label"] == args.label), key=lambda r: r["question_id"])
    if any(frozen.get(r["question_id"]) != args.partition for r in population):
        raise ValueError("population contains a question outside the frozen partition")
    source = read_run_sources(args.run)
    missing = [r["question_id"] for r in population if r["question_id"] not in source]
    if missing:
        raise SystemExit(f"{len(missing)} population question(s) are not in {args.run}: {missing[:3]}")

    done, runs = already_labeled(args.already_labeled)
    to_label = [r for r in population if source[r["question_id"]]["episode_id"] not in done]
    episodes = [source[r["question_id"]] for r in to_label]

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "sampled_episodes.jsonl", episodes)
    packet = {
        "population": {"pr153_label": args.label, "partition": args.partition, "size": len(population)},
        "policy_version": approved["policy_version"],
        "labels_sha256": approved["labels_sha256"],
        "already_labeled_runs": runs,
        "already_labeled_in_population": len(population) - len(to_label),
        "to_label": len(to_label),
        "input_hashes": {str(p): sha256_file(p) for p in (args.labels, args.counts, args.partitions,
                                                          args.run / "train/interactions.jsonl",
                                                          args.run / "validation/interactions.jsonl")},
        "sampled_episodes_sha256": sha256_file(args.output / "sampled_episodes.jsonl"),
        "episodes": [{"question_id": r["question_id"], "episode_id": source[r["question_id"]]["episode_id"],
                      "pr153_label": r["label"], "pr153_reason": r.get("reason"), "pr153_flags": r["flags"],
                      "already_labeled": source[r["question_id"]]["episode_id"] in done} for r in population],
    }
    (args.output / "selection.json").write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: packet[k] for k in ("population", "already_labeled_in_population", "to_label")}, indent=2))


if __name__ == "__main__":
    main()
