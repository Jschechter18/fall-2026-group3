"""Phase 6 of the LLM-edited SAE experiment (cookbooks/LLM_EDITED_SAE_PROMPT.md): stage 6's feature_sets.json.

Each edited SAE has the Pretrained SAE's encoder, so by construction it has the Pretrained SAE's
features: the same selected sets and the same random-control pool. This writes a feature_sets.json
whose entry for each edited SAE is the study's Pretrained SAE entry (selected features,
coefficients, all 100 random draws per label) with that SAE's own path and sha256s.

Why not use the edited SAEs' own stage-5 run directly: run on this machine, stage 5 selects the same
10 + 10 features, but 24 of the 100 random draws differ from the study's pool, and the unedited
Pretrained SAE gives the same different pool here, so the cause is the machine (tiny numerical
differences in the encodings move firing frequencies), not the edits. Using the study's pool keeps
decision 10: the new groups' random-control generations must reproduce the study run's text for text.

Checks, and refuses to write if any fails: the stage-5 run covers exactly the expected groups; each
group's selected adoption and retention features equal the study's Pretrained SAE's, in order; and
each group's path and sha256s are the SAE files on disk now.

    PYTHONPATH=src python scripts/sae/llm_edit_feature_sets.py
"""

import argparse
import json
from pathlib import Path

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic

STUDY_FEATURE_SETS = Path("results/feature_selection_intervention_layer25/20261105T025730344698Z_pool/feature_sets.json")
EDITED_STAGE5 = Path("results/feature_selection_llm_edited_layer25/20261112T002233925900Z")
GROUPS = ("LLM-edited SAE", "Mechanical edit", "Random edit")
SAE_NAME = "Pretrained SAE"
LABELS = ("adoption", "retention")


def run(study_path: Path, stage5_dir: Path) -> Path:
    study_entry = read_json(study_path)[SAE_NAME]
    edited = read_json(stage5_dir / "feature_sets.json")
    problems = check(study_entry, edited)
    if problems:
        raise SystemExit("Not writing feature_sets.json:\n  - " + "\n  - ".join(problems))

    feature_sets = {name: {**study_entry, "path": edited[name]["path"], "sha256": edited[name]["sha256"]}
                    for name in GROUPS}
    out_dir = stage5_dir.with_name(stage5_dir.name + "_study_pool")
    out_dir.mkdir(exist_ok=False)
    write_json_atomic(out_dir / "feature_sets.json", feature_sets)
    write_json_atomic(out_dir / "provenance.json", {
        "command": get_run_command(), "git_commit": get_git_commit(),
        "copied_from": {"entry": f"{study_path} [{SAE_NAME}]", "sha256": sha256_file(study_path)},
        "paths_from": {"run": str(stage5_dir), "sha256": sha256_file(stage5_dir / "feature_sets.json")},
        "random_draws_identical_to_this_machines_stage5": {
            name: sum(a == b for a, b in zip(study_entry["random_draws_adoption"], edited[name]["random_draws_adoption"]))
            for name in GROUPS},
        "why": "same encoder, so the same features; the study's pool keeps decision 10's exact reproduction check",
    })
    print(f"Wrote {out_dir / 'feature_sets.json'}: {', '.join(GROUPS)} with the study's {SAE_NAME} features and pool.")
    return out_dir


def check(study_entry: dict, edited: dict) -> list[str]:
    problems = []
    if sorted(edited) != sorted(GROUPS):
        problems.append(f"the stage-5 run has {sorted(edited)}, expected {sorted(GROUPS)}")
    for name in set(GROUPS) & set(edited):
        entry = edited[name]
        for label in LABELS:
            ours = [f["feature"] for f in entry[f"{label}_features"]]
            if ours != [f["feature"] for f in study_entry[f"{label}_features"]]:
                problems.append(f"{name}: its {label} features {ours} are not the {SAE_NAME}'s")
        for file, recorded in entry["sha256"].items():
            if not Path(file).is_file() or sha256_file(Path(file)) != recorded:
                problems.append(f"{name}: {file} is missing or no longer matches its recorded sha256")
    return problems


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-feature-sets", type=Path, default=STUDY_FEATURE_SETS)
    parser.add_argument("--stage5-run", type=Path, default=EDITED_STAGE5)
    args = parser.parse_args()
    run(args.study_feature_sets, args.stage5_run)


if __name__ == "__main__":
    main()
