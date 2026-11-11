"""Phase 2 of the LLM-edited SAE experiment (LLM_EDITED_SAE_PROMPT.md): the sandbox's read-only inputs.

Writes one folder that the Docker sandbox mounts at /inputs (the SAE itself is mounted separately,
read-only, from its own folder). Everything here is TRAIN data, or derived from the weights alone:

    X_train.npy        layer-25 activations of the stage-5 train rows, float32 (rows × 2560)
    y_train.npy        1 = adoption, 0 = retention (stage 5's primary_target), int8
    meta_train.jsonl   per row: episode/question id, dataset source, label, question, A1, the Critic's
                       answer, A2 (from the train interactions.jsonl, joined on episode id)
    Z_train.npz        encode(X_train) with the Pretrained SAE, scipy sparse (computed once here: the
                       encoder never changes, and a dense copy would be ~1.7 GB)
    hints.json         the 20 features' step-4 descriptions and profiles (train) and logit-lens tokens
                       (weights); description_test_spearman is dropped because it used validation
    rules.json         the edit rules and the 20 selected features (mas_sae.llm_edit.edit_rules)
    inputs.json        provenance: input sha256s, git commit, command, counts

The rows are loaded exactly as stage 5 loads them (configs/feature_selection/feature_selection_
intervention_layer25.yaml), so they align with the labels the features were selected with.

    PYTHONPATH=src python scripts/sae/llm_edit_sandbox_inputs.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import torch

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic
from mas_sae.experiments.records import read_jsonl
from mas_sae.intervention.experiment.representations import read_feature_sets
from mas_sae.llm_edit.edit_rules import rules_record
from mas_sae.probe import behavior_data as bd
from mas_sae.probe import layer_selection as ls
from mas_sae.probe.feature_selection import FeatureSelectionConfig
from mas_sae.sae.pretrained import load_gemma_scope_sae

STAGE5_CONFIG = Path("configs/feature_selection/feature_selection_intervention_layer25.yaml")
TRAIN_INTERACTIONS = Path(
    "artifacts/collection/mixed_final_001/prepared/results/mixed_final_001/train/interactions.jsonl")
FEATURE_SETS = Path(
    "results/feature_selection_intervention_layer25/20261105T025730344698Z_pool/feature_sets.json")
FEATURE_SUMMARY = Path("results/feature_interpretation_layer25/20261106T221912476834Z/feature_summary.csv")
SAE_NAME = "Pretrained SAE"
HINT_COLUMNS = ["feature_set", "claude_description", "claude_confidence", "strongest_property",
                "top_promoted_tokens", "fires_in_pct_of_episodes", "mean_value_adoption", "mean_value_retention"]
META_TEXT_FIELDS = ["question", "solver_attempt_1", "critic_advocated_answer", "solver_attempt_2"]
ENCODE_CHUNK_ROWS = 2048
DEFAULT_OUTPUT_DIR = Path("artifacts/llm_edit/sandbox_inputs")


def run(output_dir: Path) -> None:
    if output_dir.exists():
        raise SystemExit(f"{output_dir} already exists; choose another --output-dir or remove it.")
    stage5 = FeatureSelectionConfig.from_yaml(STAGE5_CONFIG)
    sae_dir = Path(stage5.saes[SAE_NAME]["path"])

    rows, X_train, y_train = stage5_train_rows(stage5)
    meta_train = train_metadata(rows, TRAIN_INTERACTIONS)
    Z_train = encode_train(X_train, sae_dir)
    selected = selected_features(FEATURE_SETS, SAE_NAME)
    hints = feature_hints(FEATURE_SUMMARY, SAE_NAME, selected)

    output_dir.mkdir(parents=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    meta_train.to_json(output_dir / "meta_train.jsonl", orient="records", lines=True, force_ascii=False)
    scipy.sparse.save_npz(output_dir / "Z_train.npz", Z_train)
    (output_dir / "hints.json").write_text(json.dumps(hints, indent=1, ensure_ascii=False))
    (output_dir / "rules.json").write_text(json.dumps(rules_record(selected), indent=1))
    write_provenance(output_dir, sae_dir, X_train, y_train, Z_train, selected)
    print(f"Sandbox inputs in {output_dir}: X_train {X_train.shape}, {int(y_train.sum())} adoption / "
          f"{int((y_train == 0).sum())} retention, Z_train nnz {Z_train.nnz} "
          f"({Z_train.nnz / Z_train.shape[0]:.0f} active features per row).")


# =====================================================================
# STEPS
# =====================================================================

def stage5_train_rows(stage5: FeatureSelectionConfig):
    """The rows, activations and labels stage 5 fits on (feature_selection.py loads them the same way)."""
    selected_rows, _, columns = ls.load_inputs(stage5.layer_selection_config())
    X = bd.load_layer_matrix(selected_rows, Path(stage5.activations_dir) / f"layer_{stage5.layer:02d}",
                             stage5.activation_file)
    is_train = selected_rows[columns.partition].astype(str).to_numpy() == stage5.fit_partition
    rows = selected_rows[is_train].reset_index(drop=True)
    y = rows[columns.target].to_numpy().astype(np.int8)
    expected = np.where(rows["behavior_label"] == "adoption", 1, 0)
    if not np.array_equal(y, expected):
        raise SystemExit("primary_target does not match behavior_label (1 = adoption): stage 5's contract changed.")
    return rows, np.ascontiguousarray(X[is_train], dtype=np.float32), y


def train_metadata(rows: pd.DataFrame, interactions_path: Path) -> pd.DataFrame:
    """Per stage-5 row, in the same order: ids, dataset source, label, and the episode's texts."""
    by_episode = {r["episode_id"]: r for r in read_jsonl(interactions_path)}
    missing = [e for e in rows["ep_episode_id"] if e not in by_episode]
    if missing:
        raise SystemExit(f"{len(missing)} stage-5 rows have no train interaction, e.g. {missing[:3]}")
    texts = pd.DataFrame([{field: by_episode[e][field] for field in META_TEXT_FIELDS} for e in rows["ep_episode_id"]])
    return pd.concat([pd.DataFrame({"episode_id": rows["ep_episode_id"], "question_id": rows["question_id"],
                                    "dataset_source": rows["meta_dataset_source"],
                                    "behavior_label": rows["behavior_label"]}), texts], axis=1)


def encode_train(X_train: np.ndarray, sae_dir: Path) -> scipy.sparse.csr_matrix:
    """encode(X_train) with mas_sae's JumpReLU SAE, in chunks, stored sparse."""
    sae, _ = load_gemma_scope_sae(sae_dir)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_train), ENCODE_CHUNK_ROWS):
            codes = sae.encoder(torch.from_numpy(X_train[start:start + ENCODE_CHUNK_ROWS]).float())
            chunks.append(scipy.sparse.csr_matrix(codes.numpy()))
    return scipy.sparse.vstack(chunks).tocsr().astype(np.float32)


def selected_features(feature_sets_path: Path, sae_name: str) -> dict[str, list[int]]:
    entry = read_feature_sets(feature_sets_path)[sae_name]
    return {label: [f["feature"] for f in entry[f"{label}_features"]] for label in ("adoption", "retention")}


def feature_hints(summary_path: Path, sae_name: str, selected: dict[str, list[int]]) -> dict:
    summary = pd.read_csv(summary_path)
    summary = summary[summary["representation"] == sae_name].set_index("feature")
    features = selected["adoption"] + selected["retention"]
    if missing := sorted(set(features) - set(summary.index)):
        raise SystemExit(f"{summary_path} has no row for features {missing}")
    return {str(f): summary.loc[f, HINT_COLUMNS].to_dict() for f in features}


def write_provenance(output_dir: Path, sae_dir: Path, X_train, y_train, Z_train, selected) -> None:
    inputs = [STAGE5_CONFIG, TRAIN_INTERACTIONS, FEATURE_SETS, FEATURE_SUMMARY, sae_dir / "params.safetensors"]
    outputs = sorted(p.name for p in output_dir.iterdir())
    write_json_atomic(output_dir / "inputs.json", {
        "git_commit": get_git_commit(), "command": get_run_command(),
        "inputs_sha256": {str(p): sha256_file(p) for p in inputs},
        "outputs_sha256": {name: sha256_file(output_dir / name) for name in outputs},
        "sae_dir": str(sae_dir), "n_rows": len(X_train), "n_adoption": int(y_train.sum()),
        "z_train_nnz": int(Z_train.nnz), "selected": selected,
    }, overwrite=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
