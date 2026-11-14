"""Build a full-pool feature_sets.json from an existing feature-selection run, without rerunning it.

The intervention gives each question its own random draws from a pool of matched random draws
(experiment.random_draw_pool_size, 100). Feature-selection runs made before that change wrote only
the first 3 draws per label into feature_sets.json, but every draw is in the same run's
feature_selection_summary.json (saes.<name>.random_control_draws / random_control_draws_retention).

This copies feature_sets.json with random_draws_adoption and random_draws_retention replaced by every
draw from the summary, after checking that both files describe the same selection: the same
representations, the same targeted sets, and the file's draws are the summary's first draws.
Everything else in each entry (checkpoint path and sha256, features and coefficients) is unchanged.
The run folder is only read; the new file goes to a new folder.

    python scripts/intervention/build_feature_set_pool.py \\
        results/feature_selection_intervention_layer25/20261105T025730344698Z \\
        results/feature_selection_intervention_layer25/20261105T025730344698Z_pool
"""

import argparse
import json
from pathlib import Path

LABELS = (("adoption", "targeted_features", "random_control_draws"),
          ("retention", "retention_features", "random_control_draws_retention"))


def run(run_dir: Path, out_dir: Path) -> Path:
    """Write ``out_dir/feature_sets.json`` with the full random-draw pools of ``run_dir``'s selection,
    after ``check_same_selection``. Returns the new file's path."""
    feature_sets = read_json(run_dir / "feature_sets.json")
    summaries = read_json(run_dir / "feature_selection_summary.json")["saes"]
    check_same_selection(feature_sets, summaries)
    pooled = with_full_pools(feature_sets, summaries)
    return write_new(out_dir / "feature_sets.json", pooled)


def check_same_selection(feature_sets: dict, summaries: dict) -> None:
    """Refuse unless the summary is the selection feature_sets.json was cut from."""
    problems = []
    if missing := sorted(set(feature_sets) - set(summaries)):
        raise SystemExit(f"Not the same selection: {missing} are not in the summary")
    for name, entry in feature_sets.items():
        summary = summaries.get(name, {})
        for label, targeted_key, draws_key in LABELS:
            if [f["feature"] for f in entry[f"{label}_features"]] != summary.get(targeted_key):
                problems.append(f"{name}: the {label} targeted sets differ")
            kept, pool = entry[f"random_draws_{label}"], summary.get(draws_key, [])
            if pool[:len(kept)] != kept:
                problems.append(f"{name}: the file's {label} draws are not the summary's first {len(kept)}")
            if len(pool) <= len(kept):
                problems.append(f"{name}: the summary has {len(pool)} {label} draws, no more than the file's")
    if problems:
        raise SystemExit("Not the same selection:\n  - " + "\n  - ".join(problems))


def with_full_pools(feature_sets: dict, summaries: dict) -> dict:
    """``feature_sets`` with each entry's random_draws_adoption and random_draws_retention replaced by
    every draw in the summary."""
    return {name: {**entry, **{f"random_draws_{label}": summaries[name][draws_key] for label, _, draws_key in LABELS}}
            for name, entry in feature_sets.items()}


def write_new(path: Path, value: dict) -> Path:
    """Write ``value`` as JSON to ``path`` and return it. Exits (SystemExit) rather than overwrite."""
    if path.exists():
        raise SystemExit(f"{path} already exists; not overwriting it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


def read_json(path: Path) -> dict:
    """Parse the JSON file at ``path``."""
    return json.loads(Path(path).read_text())


def main() -> None:
    """Command line: build the full-pool file and print each representation's pool sizes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="the feature-selection run folder (read only)")
    parser.add_argument("out_dir", type=Path, help="a new folder for the full-pool feature_sets.json")
    args = parser.parse_args()
    path = run(args.run_dir, args.out_dir)
    pools = {name: (len(e["random_draws_adoption"]), len(e["random_draws_retention"]))
             for name, e in read_json(path).items()}
    print(f"Wrote {path}: random draw pools (adoption, retention) {pools}")


if __name__ == "__main__":
    main()
