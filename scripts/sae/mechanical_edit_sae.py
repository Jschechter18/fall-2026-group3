"""Phase 4 of the LLM-edited SAE experiment (cookbooks/LLM_EDITED_SAE_PROMPT.md): the mechanical control.

The same kind of edit as the LLM's, by a fixed formula and no LLM (cookbook Section 6). Built BEFORE the
real LLM run, so the control is fixed before anyone sees the LLM's edits.

  1. d = normalize(mean(train adoption activations) − mean(train retention activations)), on the stage-5
     train rows and labels (the sandbox inputs' X_train, y_train; mas_sae.llm_edit.directions).
  2. Each selected adoption row r, with u = r/|r|: p = normalize(d − (d·u)u), the part of d perpendicular
     to u, and new = |r|·(c·u + sqrt(1 − c²)·p). That is cosine c with the original row, as far toward d
     as the cap allows, with the original length. c is the cap (0.95) plus a margin of 1e-5, so float32
     rounding of the saved row cannot put it below the cap.
  3. Each selected retention row: the same toward −d.
  4. Every other row unchanged.

Writes results/sae/mechanical_edit/<run_id>/: edits.py and rows/ (the same record as the LLM run, so
the audit treats both alike), sae/ (kind: gemma_scope, for stages 5 and 6) and run.json. With
--steer-test it also runs steer_test (the stage-6 "both" arm on the same 200 train episodes the LLM
run uses; GPU only, no Bedrock), as a check that the test registers an edit this size at all.

    PYTHONPATH=src python scripts/sae/mechanical_edit_sae.py --steer-test
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic
from mas_sae.intervention.experiment.config import load_config
from mas_sae.intervention.run_files.layout import new_run_id
from mas_sae.llm_edit.directions import direction_report, mean_difference_direction, unit
from mas_sae.llm_edit.edit_rules import COSINE_CAP, Edit, apply_edits, cosine_similarity, edit_problems, write_edits_py
from mas_sae.llm_edit.edited_sae import SAE_DIR, original_sae, save_edited_sae
from mas_sae.llm_edit.steer_test import Steerer

STUDY_CONFIG = Path("configs/intervention/intervention.yaml")   # read-only: seed, r, feature sets, solver
SANDBOX_INPUTS = Path("artifacts/llm_edit/sandbox_inputs")      # the stage-5 train rows (Phase 2)
OUTPUT_ROOT = Path("results/sae/mechanical_edit")
CAP_MARGIN = 1e-5
TARGET_COSINE = COSINE_CAP + CAP_MARGIN
STEER_EPISODES = 200      # the same episodes as the LLM run's steer_test (first 200 of the seeded order)


def run(steer_test: bool) -> Path:
    study = load_config(STUDY_CONFIG)
    original_w_dec, sae_config, selected = original_sae(study.inputs.feature_sets)
    d = mean_difference_direction(SANDBOX_INPUTS)

    edits = mechanical_edits(original_w_dec, selected, d)
    require_valid(original_w_dec, selected, edits)
    edited = apply_edits(original_w_dec, edits)

    run_dir = OUTPUT_ROOT / new_run_id()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_edits_py(edits, run_dir)
    sae_dir = save_edited_sae(edited, sae_config, run_dir / "sae",
                              {"edit_source": "mechanical formula (scripts/sae/mechanical_edit_sae.py)"})
    record = run_record(run_dir, sae_dir, edits, original_w_dec, edited, selected, d)
    write_json_atomic(run_dir / "run.json", record)
    print(f"Mechanical edit in {run_dir}: {len(edits)} rows rotated to cosine {TARGET_COSINE} toward ±d.")

    if steer_test:
        summary, text = Steerer(study, STEER_EPISODES, run_dir / "steer_test").test(edited, call=1)
        write_json_atomic(run_dir / "run.json", {**record, "steer_test": summary})
        print(text)
    return run_dir


# =====================================================================
# STEPS
# =====================================================================

def mechanical_edits(original_w_dec: np.ndarray, selected: dict[str, list[int]], d: np.ndarray) -> list[Edit]:
    """Adoption rows toward d, retention rows toward −d, each to TARGET_COSINE with its original."""
    edits = []
    for label, target in (("adoption", d), ("retention", -d)):
        for feature in selected[label]:
            row = rotated_toward(original_w_dec[feature], target, TARGET_COSINE)
            why = f"mechanical: {label} row rotated toward {'d' if label == 'adoption' else '-d'} to cosine {TARGET_COSINE}"
            edits.append(Edit(len(edits) + 1, feature, row.astype(np.float32), why))
    return edits


def rotated_toward(row: np.ndarray, target: np.ndarray, cosine: float) -> np.ndarray:
    """|r|·(c·u + sqrt(1 − c²)·p): cosine c with r, as far toward `target` as c allows, same length.

    p is the unit part of `target` perpendicular to u = r/|r|; moving along p is the shortest way from u
    toward `target`, so for a fixed cosine with r this is the row closest to `target`."""
    length = np.linalg.norm(row.astype(np.float64))
    u = unit(row)
    p = unit(target - (target @ u) * u)
    return length * (cosine * u + np.sqrt(1 - cosine ** 2) * p)


def require_valid(original_w_dec, selected, edits: list[Edit]) -> None:
    """The control must follow the same rules as the LLM (mas_sae.llm_edit.edit_rules)."""
    problems = [f"feature {e.feature}: {p}" for e in edits for p in edit_problems(original_w_dec, selected, e.feature, e.row)]
    if problems:
        raise SystemExit("The mechanical edit breaks the edit rules:\n  - " + "\n  - ".join(problems))


def run_record(run_dir, sae_dir, edits, original_w_dec, edited, selected, d) -> dict:
    inputs = [SAE_DIR / "params.safetensors", SAE_DIR / "config.json", SANDBOX_INPUTS / "inputs.json", STUDY_CONFIG]
    return {
        "run_id": run_dir.name, "command": get_run_command(), "git_commit": get_git_commit(),
        "formula": "cookbooks/LLM_EDITED_SAE_PROMPT.md, Section 6 (adoption rows toward d, retention rows toward -d)",
        "target_cosine": TARGET_COSINE, "cosine_cap": COSINE_CAP, "cap_margin": CAP_MARGIN,
        "d_sha256": hashlib.sha256(np.ascontiguousarray(d, dtype=np.float64).tobytes()).hexdigest(),
        "inputs_sha256": {str(p): sha256_file(p) for p in inputs},
        "row_cosines_with_original": {str(e.feature): cosine_similarity(e.row, original_w_dec[e.feature]) for e in edits},
        "set_directions": direction_report(edited, original_w_dec, selected, d),
        "edited_sae_sha256": {f.name: sha256_file(f) for f in sorted(sae_dir.iterdir())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steer-test", action="store_true",
                        help="also run steer_test on 200 train episodes (GPU, about 10 minutes, no Bedrock)")
    run(parser.parse_args().steer_test)


if __name__ == "__main__":
    main()
