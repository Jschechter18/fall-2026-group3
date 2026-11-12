"""Held-out steer_test for the LLM-edited SAE experiment (cookbooks/LLM_EDITED_SAE.md).

The LLM saw steer_test results on episodes 0..199 of the seeded train order and chose its edit from
them, so its 24 flips there are in-sample. This runs the same test (the stage-6 "both" arm, r from
the study config) on later episodes of the same order, which nobody saw, for all three edited SAEs
plus the unedited one, with one Gemma load and one shared baseline and unedited row per episode.

It is a go/no-go check before stage 6, not a result: train episodes, one arm, no labeler.
Flips use the same exact-match rule as steer_test. Each pair of SAEs is also compared episode by
episode with an exact McNemar test on the episodes where exactly one of the two flipped.

Writes results/sae/llm_edit_heldout/<run_id>/: summary.json, and steer_test/ with unedited.jsonl and
call_1..3.jsonl (llm, mechanical, random). GPU only, no Bedrock. Rerun with --resume <that folder> to continue a run
that stopped; finished generations are skipped.

    PYTHONPATH=src python scripts/sae/heldout_steer_test.py
"""

import argparse
import itertools
import math
from pathlib import Path

from safetensors.numpy import load_file

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic
from mas_sae.intervention.experiment.config import load_config
from mas_sae.intervention.run_files.layout import new_run_id
from mas_sae.intervention.run_files.results_file import ResultsFile
from mas_sae.llm_edit.steer_test import EDITED, UNEDITED, Steerer, both_arm, classify_answer

STUDY_CONFIG = Path("configs/intervention/intervention.yaml")
OUTPUT_ROOT = Path("results/sae/llm_edit_heldout")
GROUPS = {   # call number = position + 1; order fixed so the files are easy to match up
    "llm": Path("results/sae/llm_edited/20261111T213251406629Z-c53c08f04ab9"),
    "mechanical": Path("results/sae/mechanical_edit/20261111T211535990035Z-6a61cf6c6b31"),
    "random": Path("results/sae/random_edit/20261111T234022791132Z-c53c08f04ab9"),
}
FIRST_EPISODE = 200   # the LLM run used 0..199
N_EPISODES = 400


def run(first_episode: int, n_episodes: int, resume: Path | None) -> Path:
    study = load_config(STUDY_CONFIG)
    out_dir = resume or OUTPUT_ROOT / new_run_id()
    out_dir.mkdir(parents=True, exist_ok=resume is not None)
    rows_dir = out_dir / "steer_test"    # per-row files; gitignored like the other runs' steer_test/
    steerer = Steerer(study, n_episodes, rows_dir, first_episode=first_episode)

    summaries = {}
    for call, (name, run_dir) in enumerate(GROUPS.items(), start=1):
        w_dec = load_file(str(run_dir / "sae" / "params.safetensors"))["w_dec"]
        summaries[name], _ = steerer.test(w_dec, call)
        print(f"{name}: {summaries[name]['edited']}", flush=True)

    edit_size = study.experiment.edit_size
    flipped = {"unedited": flipped_episodes(steerer, rows_dir / "unedited.jsonl", both_arm(UNEDITED, edit_size).key)}
    for call, name in enumerate(GROUPS, start=1):
        flipped[name] = flipped_episodes(steerer, rows_dir / f"call_{call}.jsonl", both_arm(EDITED, edit_size).key)
    paired = {f"{a} vs {b}": mcnemar(flipped[a], flipped[b]) for a, b in itertools.combinations(flipped, 2)}

    record = {
        "run_id": out_dir.name, "command": get_run_command(), "git_commit": get_git_commit(),
        "episodes": {"first": first_episode, "count": n_episodes, "order": "seeded order of eligible train episodes"},
        "groups": {name: {"run": str(path), "params_sha256": sha256_file(path / "sae" / "params.safetensors")}
                   for name, path in GROUPS.items()},
        "steer_test": summaries, "paired_flips": paired,
    }
    write_json_atomic(out_dir / "summary.json", record)
    print_report(record, flipped)
    print(f"\nWritten to {out_dir / 'summary.json'}")
    return out_dir


def flipped_episodes(steerer: Steerer, path: Path, condition_key: str) -> set[str]:
    """Episodes whose second answer exactly matches the opposite behavior (steer_test's flip rule)."""
    rows = {r["episode_id"]: r for r in ResultsFile(path).rows if r["condition"] == condition_key}
    return {e.episode_id for e in steerer.episodes
            if (behavior := classify_answer(rows[e.episode_id]["answer"], e)) is not None and behavior != e.stored_label}


def mcnemar(a: set[str], b: set[str]) -> dict:
    """Exact two-sided McNemar test on the discordant episodes (flipped under one SAE only)."""
    only_a, only_b = len(a - b), len(b - a)
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_first": only_a, "only_second": only_b, "both": len(a & b), "p_value": p}


def print_report(record: dict, flipped: dict[str, set]) -> None:
    episodes = record["episodes"]["count"]
    first = next(iter(record["steer_test"].values()))
    print(f"\nHeld-out steer_test: episodes {record['episodes']['first']}..{record['episodes']['first'] + episodes - 1} "
          f"({episodes}; {first['adopted']} adopted; baseline reproduces stored answer on {first['baseline_reproduces_stored_a2']})")
    rows = [("unedited", first["unedited"])] + [(name, s["edited"]) for name, s in record["steer_test"].items()]
    print(f"  {'':11s} {'flips':>6s} {'A->R':>5s} {'R->A':>5s} {'neither':>8s} {'margin':>7s}")
    for name, s in rows:
        print(f"  {name:11s} {s['flips']:>6d} {s['adoption_to_retention']:>5d} {s['retention_to_adoption']:>5d} "
              f"{s['unclear']:>8d} {s['mean_margin_change']:>+7.2f}")
    print("  paired (flipped under first only / second only, exact McNemar p):")
    for pair, t in record["paired_flips"].items():
        print(f"    {pair:26s} {t['only_first']:>3d} / {t['only_second']:<3d}  p = {t['p_value']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-episode", type=int, default=FIRST_EPISODE)
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--resume", type=Path, help="an earlier run folder to continue")
    args = parser.parse_args()
    if args.first_episode < FIRST_EPISODE:
        raise SystemExit(f"Episodes before {FIRST_EPISODE} are the LLM run's own steer_test episodes, not held out.")
    run(args.first_episode, args.episodes, args.resume)


if __name__ == "__main__":
    main()
