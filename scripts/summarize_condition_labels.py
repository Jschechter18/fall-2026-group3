"""Compare Adoption/Retention balance across Josh's adoption-protocol conditions under Claude labels (#166).

Reads the ``manifest.json`` written by ``export_adoption_conditions.py`` and one Claude run per
condition (``<runs-root>/<slug>``). Each run must have labeled exactly that condition's exported
file (same sha256). Reports, per condition, Claude's four-class counts and the Retention share of
Adoption+Retention with a 95% Wilson interval, next to the old lexical counts.

Usage:
    PYTHONPATH=src python scripts/summarize_condition_labels.py \\
        --manifest results/claude_validation/samples/adoption_conditions_80q/manifest.json \\
        --runs-root results/claude_validation/adoption_conditions_80q_opus5 \\
        --output results/claude_validation/adoption_conditions_80q_opus5_summary
"""

import argparse
import json
from pathlib import Path

from mas_sae.evaluation.claude_agreement import label_distribution
from mas_sae.experiments.artifacts import ensure_output_available
from mas_sae.experiments.records import read_jsonl


def lexical_counts(rows: list[dict]) -> dict[str, int]:
    """Josh's disjoint lexical stacks: overlap (both), adopted only, retained only, other."""
    out = {"adopted": 0, "retained_a1": 0, "overlap": 0, "other": 0}
    for r in rows:
        key = ("overlap" if r["adopted_target"] and r["retained_a1"] else "adopted" if r["adopted_target"]
               else "retained_a1" if r["retained_a1"] else "other")
        out[key] += 1
    return out


def condition_summary(condition: dict, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("episodes_sha256") != condition["sampled_episodes_sha256"]:
        raise SystemExit(f"{run_dir} did not label the exported {condition['slug']} file")
    rows = {r["episode_id"]: r for r in read_jsonl(run_dir / "raw_results.jsonl")}
    if set(rows) != {e["episode_id"] for e in condition["lexical"]}:
        raise SystemExit(f"{run_dir} covers different episodes than {condition['slug']}")
    return {"condition": condition["condition"], "a2_presentation": condition["a2_presentation"],
            "claude": label_distribution(r.get("behavior_label") for r in rows.values()),
            "lexical": lexical_counts(condition["lexical"]),
            "model": summary.get("models_returned"), "prompt_version": summary.get("prompt_version"),
            "input_tokens": summary["total_input_tokens"], "output_tokens": summary["total_output_tokens"],
            "wall_clock_seconds": summary.get("wall_clock_seconds")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)
    manifest = json.loads(args.manifest.read_text())
    results = [condition_summary(c, args.runs_root / c["slug"]) for c in manifest["conditions"]]
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps({"source": manifest["source"], "conditions": results},
                                                         indent=2) + "\n", encoding="utf-8")
    print(f"{'condition':40s} {'A':>4s} {'R':>4s} {'Ag':>4s} {'Out':>4s} {'err':>4s}  R/(A+R)          lexical A/R")
    for r in results:
        c, b = r["claude"], r["claude"]["retention_share_of_binary"]
        ci = f"[{b['ci95'][0]:.0%}-{b['ci95'][1]:.0%}]" if b["ci95"] else ""
        share = f"{b['share']:.0%}" if b["share"] is not None else "-"
        print(f"{r['condition']:40s} {c['adoption']['count']:>4d} {c['retention']['count']:>4d} "
              f"{c['agreement']['count']:>4d} {c['out']['count']:>4d} {c['error']['count']:>4d}  "
              f"{share:>4s} {ci:12s}  {r['lexical']['adopted']}/{r['lexical']['retained_a1']}")


if __name__ == "__main__":
    main()
