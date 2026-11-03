"""Check that the reusing replay path (#214) writes bitwise the rows a finished run wrote.

Regenerates every edited condition of the given finished episodes with the current code, on the run's
device type, and compares each row with the run's, field by field, except ``seconds``. Nothing is
written into the run; the new rows go to ``--output``, and a summary is printed. The inputs are hashed
before they are read and must be unchanged at the end, so check a copy of a shard that is still running.

    python scripts/check_replay_reuse.py \
        --run-dir results/causal_topk_mixed_layer25/<run> \
        --rows results/causal_topk_mixed_layer25/<run>/distributed/shards/000002/per_row.jsonl \
        --question-ids <id> [<id> ...] --output /tmp/check_214.jsonl
"""

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

from mas_sae.causal import replay
from mas_sae.experiments.artifacts import sha256_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="the run's directory with replay_config.json")
    p.add_argument("--rows", type=Path, nargs="+", required=True, help="per_row.jsonl files holding the episodes")
    p.add_argument("--question-ids", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; give a new path.")

    inputs = [args.run_dir / "replay_config.json", *args.rows]
    hashes = {str(path): sha256_file(path) for path in inputs}
    meta = json.loads((args.run_dir / "replay_config.json").read_text())
    cfg = replay.ReplayConfig.from_dict(meta["config"], source=str(args.run_dir))
    wanted = set(args.question_ids)
    episodes = replay.load_episodes(cfg)
    episodes = episodes[episodes.question_id.astype(str).isin(wanted)]
    if set(episodes.question_id.astype(str)) != wanted:
        raise SystemExit("Some question ids are not eligible episodes of this run.")
    found = [r for path in args.rows for r in replay.read_rows(path) if r["question_id"] in wanted]
    old = {replay.key(r): r for r in found}
    if len(old) != len(found):
        raise SystemExit("The rows hold some (episode, condition, draw) more than once.")

    comp = replay.load_components(cfg, episodes)
    comp.info["label_policy_name"] = cfg.label_policy
    conds = replay.configured_conditions(cfg, comp.features)
    if json.loads(json.dumps(conds, default=str)) != meta["conditions"]:
        raise SystemExit("These conditions differ from the run's.")
    todo = [(ep, c) for ep in episodes.to_dict("records") for c in conds if c["features"] is not None]
    expected = [(ep["question_id"], c["experiment"], c["condition"], c["draw"]) for ep, c in todo]
    missing = [k for k in expected if k not in old]
    if missing or not all((q, replay.BASELINE, "no_sae", None) in old for q in wanted):
        raise SystemExit(f"The rows do not hold these episodes in full ({len(missing)} edited rows missing).")

    first = {k: r["first_token_id"] for k, r in old.items() if k[1] == replay.BASELINE}
    t0 = time.perf_counter()
    replay.generate_all(todo, comp, args.output, first, "check")
    wall = time.perf_counter() - t0

    if {str(path): sha256_file(path) for path in inputs} != hashes:
        raise SystemExit("The run's config or rows changed during the check; check a copy.")
    regenerated = replay.read_rows(args.output)
    if sorted(map(replay.key, regenerated), key=str) != sorted(expected, key=str):
        raise SystemExit("The check did not regenerate exactly these episodes' edited conditions.")
    fields, rows = Counter(), 0
    for new in regenerated:
        ref = old[replay.key(new)]
        diff = [k for k in set(new) | set(ref) if k != "seconds" and new.get(k) != ref.get(k)]
        fields.update(diff)
        rows += bool(diff)
    run_seconds = sum(old[k]["seconds"] for k in expected)
    print(json.dumps({"inputs_sha256": hashes, "episodes": sorted(wanted), "rows": len(todo), "rows_differing": rows,
                      "fields_differing": dict(fields), "seconds_now": round(wall, 1),
                      "seconds_in_run": round(run_seconds, 1), "speedup": round(run_seconds / wall, 2)}, indent=2))
    if rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
