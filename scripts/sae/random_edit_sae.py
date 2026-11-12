"""Phase 4 of the LLM-edited SAE experiment (cookbooks/LLM_EDITED_SAE.md): the random-edit control.

The same 20 rows, each rotated by the same angle as in the LLM's final edit, but toward a random
direction. It tests "any rotation of this size changes steering this much", so the LLM edit only
counts as targeted if it beats this control.

  1. Replay the LLM run's edits.py onto the original SAE (mas_sae.llm_edit.edit_rules), the same way
     the audit does, and take each selected row's cosine c with its original row.
  2. For each selected row r, in the fixed order adoption then retention (the feature-set file's
     order): draw g ~ N(0, I) from one generator seeded with SEED, and set
     new = |r|·(c·u + sqrt(1 − c²)·p), with u = r/|r| and p the unit part of g perpendicular to u
     (mas_sae.llm_edit.directions.rotated_toward). Same angle and length as the LLM's row, random direction.
  3. Every other row unchanged.

Writes results/sae/random_edit/<run_id>/: edits.py and rows/ (the same record as the other two groups),
sae/ (kind: gemma_scope, for stages 5 and 6) and run.json. With --steer-test it also runs steer_test on
the same 200 train episodes as the other two groups (GPU only, no Bedrock).

    PYTHONPATH=src python scripts/sae/random_edit_sae.py --steer-test
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic
from mas_sae.intervention.experiment.config import load_config
from mas_sae.intervention.run_files.layout import new_run_id
from mas_sae.llm_edit.directions import direction_report, mean_difference_direction, rotated_toward
from mas_sae.llm_edit.edit_rules import (COSINE_CAP, Edit, apply_edits, cosine_similarity, read_edits_py,
                                         require_valid_edits, write_edits_py)
from mas_sae.llm_edit.edited_sae import SAE_DIR, original_sae, save_edited_sae
from mas_sae.llm_edit.steer_test import Steerer

STUDY_CONFIG = Path("configs/intervention/intervention.yaml")   # read-only: seed, r, feature sets, solver
SANDBOX_INPUTS = Path("artifacts/llm_edit/sandbox_inputs")      # only for d, a host-side diagnostic
LLM_RUN = Path("results/sae/llm_edited/20261111T213251406629Z-c53c08f04ab9")   # the real LLM run
OUTPUT_ROOT = Path("results/sae/random_edit")
SEED = 42                 # the study's seed; fixed so the control is reproducible
STEER_EPISODES = 200      # the same episodes as the other groups' steer_test (first 200 of the seeded order)


def run(llm_run: Path, steer_test: bool) -> Path:
    study = load_config(STUDY_CONFIG)
    original_w_dec, sae_config, selected = original_sae(study.inputs.feature_sets)

    llm_w_dec = apply_edits(original_w_dec, read_edits_py(llm_run))
    llm_cosines = {feature: cosine_similarity(llm_w_dec[feature], original_w_dec[feature])
                   for label in ("adoption", "retention") for feature in selected[label]}

    edits = random_edits(original_w_dec, selected, llm_cosines, SEED)
    require_valid_edits(original_w_dec, selected, edits, "random edit")
    edited = apply_edits(original_w_dec, edits)

    run_dir = OUTPUT_ROOT / new_run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_edits_py(edits, run_dir)
    sae_dir = save_edited_sae(edited, sae_config, run_dir / "sae",
                              {"edit_source": f"random edit (scripts/sae/random_edit_sae.py), angles from {llm_run}"})
    d = mean_difference_direction(SANDBOX_INPUTS)
    record = run_record(run_dir, sae_dir, llm_run, edits, original_w_dec, llm_w_dec, llm_cosines, edited, selected, d)
    write_json_atomic(run_dir / "run.json", record)
    print(f"Random edit in {run_dir}: {len(edits)} rows rotated by the LLM run's angles, seed {SEED}.")

    if steer_test:
        summary, text = Steerer(study, STEER_EPISODES, run_dir / "steer_test").test(edited, call=1)
        write_json_atomic(run_dir / "run.json", {**record, "steer_test": summary})
        print(text)
    return run_dir


# =====================================================================
# STEPS
# =====================================================================

def random_edits(original_w_dec: np.ndarray, selected: dict[str, list[int]], cosines: dict[int, float],
                 seed: int) -> list[Edit]:
    """Each selected row rotated to its given cosine with the original, toward a random direction."""
    rng = np.random.default_rng(seed)
    edits = []
    for label in ("adoption", "retention"):
        for feature in selected[label]:
            random_target = rng.standard_normal(original_w_dec.shape[1])
            row = rotated_toward(original_w_dec[feature], random_target, cosines[feature])
            why = f"random: {label} row rotated toward a random direction to the LLM edit's cosine {cosines[feature]:.6f}"
            edits.append(Edit(len(edits) + 1, feature, row.astype(np.float32), why))
    return edits


def run_record(run_dir, sae_dir, llm_run, edits, original_w_dec, llm_w_dec, llm_cosines, edited, selected, d) -> dict:
    inputs = [SAE_DIR / "params.safetensors", SAE_DIR / "config.json", STUDY_CONFIG,
              llm_run / "edits.py", SANDBOX_INPUTS / "inputs.json"]
    return {
        "run_id": run_dir.name, "command": get_run_command(), "git_commit": get_git_commit(),
        "formula": "cookbooks/LLM_EDITED_SAE.md, Section 6 (same angles as the LLM's final edit, random directions)",
        "llm_run": str(llm_run), "seed": SEED, "cosine_cap": COSINE_CAP,
        "d_sha256": hashlib.sha256(np.ascontiguousarray(d, dtype=np.float64).tobytes()).hexdigest(),
        "inputs_sha256": {str(p): sha256_file(p) for p in inputs},
        "llm_row_cosines_with_original": {str(f): c for f, c in llm_cosines.items()},
        "row_cosines_with_original": {str(e.feature): cosine_similarity(e.row, original_w_dec[e.feature]) for e in edits},
        "row_cosines_with_llm_row": {str(e.feature): cosine_similarity(e.row, llm_w_dec[e.feature]) for e in edits},
        "set_directions": direction_report(edited, original_w_dec, selected, d),
        "edited_sae_sha256": {f.name: sha256_file(f) for f in sorted(sae_dir.iterdir())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-run", type=Path, default=LLM_RUN, help=f"the LLM run whose angles to copy (default {LLM_RUN})")
    parser.add_argument("--steer-test", action="store_true",
                        help="also run steer_test on 200 train episodes (GPU, about 10 minutes, no Bedrock)")
    args = parser.parse_args()
    run(args.llm_run, args.steer_test)


if __name__ == "__main__":
    main()
