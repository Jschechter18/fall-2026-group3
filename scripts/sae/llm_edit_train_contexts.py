"""Phase 1 of the LLM-edited SAE experiment (LLM_EDITED_SAE_PROMPT.md): train contexts, no model runs.

steer_test regenerates the Solver's second answer (A2) on train episodes, which needs each episode's
passages. Collection wrote a contexts.jsonl only for the intervention partition
(src/mas_sae/experiments/distributed_collection/core.py, `write_jsonl(result_root/"contexts.jsonl", ...)`).
For train, the passages are in the collection manifest, where `questions[*]` holds `question_id` and
`payload` (paragraphs, question, ...). Non-intervention questions have `partition: null` there,
because the train/validation/test split was assigned later, so questions are selected by the IDs
in the train interactions.jsonl.

This script:
  1. writes the train contexts in the intervention file's format, {"question_id", "payload"}, in
     manifest order;
  2. checks that `eligible_episodes` builds every train episode from them (no "missing or
     mismatched collected context" error);
  3. prints the 10 adoption and 10 retention features of "Pretrained SAE" from the final
     feature_sets.json.

Inputs are only read; the output folder must not exist yet.

    PYTHONPATH=src python scripts/sae/llm_edit_train_contexts.py
"""

import argparse
from collections import Counter
from pathlib import Path

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic
from mas_sae.experiments.distributed_collection.dataset import read_manifest
from mas_sae.experiments.records import read_jsonl, write_jsonl
from mas_sae.intervention.experiment.episodes import eligible_episodes
from mas_sae.intervention.experiment.representations import read_feature_sets

MANIFEST = Path("artifacts/collection/mixed_final_001/manifest.json")
TRAIN_INTERACTIONS = Path(
    "artifacts/collection/mixed_final_001/prepared/results/mixed_final_001/train/interactions.jsonl")
TRAIN_LABELS = Path("results/claude_validation/mixed_final_001/train")
FEATURE_SETS = Path(
    "results/feature_selection_intervention_layer25/20261105T025730344698Z_pool/feature_sets.json")
SAE_NAME = "Pretrained SAE"
DEFAULT_OUTPUT_DIR = Path("artifacts/llm_edit")


def run(output_dir: Path) -> None:
    refuse_existing(output_dir)
    contexts_path = output_dir / "train_contexts.jsonl"

    question_ids = train_question_ids(TRAIN_INTERACTIONS)
    contexts = contexts_from_manifest(MANIFEST, question_ids)
    write_jsonl(contexts_path, contexts)

    episodes = eligible_episodes(TRAIN_INTERACTIONS, contexts_path, TRAIN_LABELS, "train")
    selected = selected_features(FEATURE_SETS, SAE_NAME)

    write_record(output_dir, contexts_path, question_ids, episodes, selected)
    report(contexts_path, question_ids, episodes, selected)


# =====================================================================
# STEPS
# =====================================================================

def refuse_existing(output_dir: Path) -> None:
    """Never overwrite: a rerun goes to a new folder (or delete the old one on purpose)."""
    if output_dir.exists():
        raise SystemExit(f"{output_dir} already exists; choose another --output-dir or remove it.")


def train_question_ids(interactions: Path) -> set[str]:
    return {row["question_id"] for row in read_jsonl(interactions)}


def contexts_from_manifest(manifest_path: Path, question_ids: set[str]) -> list[dict]:
    """{"question_id", "payload"} for every train question, in manifest order (as collection writes them)."""
    manifest = read_manifest(manifest_path)  # raises if the manifest's recorded sha256 no longer matches
    contexts = [{"question_id": q["question_id"], "payload": q["payload"]}
                for q in manifest["questions"] if q["question_id"] in question_ids]
    missing = question_ids - {c["question_id"] for c in contexts}
    if missing:
        raise SystemExit(f"{len(missing)} train questions are not in {manifest_path}, e.g. {sorted(missing)[:3]}")
    return contexts


def selected_features(feature_sets_path: Path, sae_name: str) -> dict[str, list[int]]:
    entry = read_feature_sets(feature_sets_path)[sae_name]
    return {label: [f["feature"] for f in entry[f"{label}_features"]] for label in ("adoption", "retention")}


def write_record(output_dir: Path, contexts_path: Path, question_ids: set[str], episodes, selected: dict) -> None:
    """Where the contexts came from, so later phases can check they use the same inputs."""
    write_json_atomic(output_dir / "train_contexts.json", {
        "git_commit": get_git_commit(),
        "command": get_run_command(),
        "inputs_sha256": {str(p): sha256_file(p) for p in (MANIFEST, TRAIN_INTERACTIONS, FEATURE_SETS)},
        "train_labels": str(TRAIN_LABELS),
        "output_sha256": sha256_file(contexts_path),
        "n_train_questions": len(question_ids),
        "n_eligible_episodes": len(episodes),
        "eligible_by_label": dict(Counter(e.stored_label for e in episodes)),
        "selected_features": {SAE_NAME: selected},
    }, overwrite=False)


def report(contexts_path: Path, question_ids: set[str], episodes, selected: dict) -> None:
    labels = Counter(e.stored_label for e in episodes)
    print(f"Wrote {len(question_ids)} train contexts to {contexts_path}.")
    print(f"eligible_episodes loads: {len(episodes)} train episodes "
          f"({labels['adoption']} adoption, {labels['retention']} retention).")
    for label, features in selected.items():
        print(f"{SAE_NAME} {label} features: {features}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
