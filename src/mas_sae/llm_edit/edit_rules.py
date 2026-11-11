"""The rules every hand edit of the decoder must follow, and edits.py, the record of accepted edits.

An edit replaces one decoder row (one feature's direction). It is valid only if:
  - the feature is one of the selected adoption or retention features;
  - the new row has the original row's length (stage 6 normalizes rows before steering, so length
    carries no steering effect; keeping it keeps the edit clean);
  - the new row has cosine >= COSINE_CAP with the ORIGINAL row (not the previous edit), so repeated
    small edits cannot drift further than one large one.

Shared by the sandbox's host side, the harness, the controls and the audit. The container cannot
import mas_sae, so the kernel re-checks the same rules from rules.json, written by `rules_record`.
The host's check is the authoritative one: the saved SAE is always the original SAE plus the edits
in edits.py, replayed by `apply_edits`.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COSINE_CAP = 0.95          # decided in Phase 0 (LLM_EDITED_SAE_PROMPT.md, decision 5)
LENGTH_TOLERANCE = 1e-4    # relative: |new| / |original| must be within 1 ± this


@dataclass(frozen=True)
class Edit:
    number: int        # 1, 2, ... in the order the edits were made
    feature: int
    row: np.ndarray    # the new decoder row, float32, shape (input_dim,)
    why: str           # the LLM's stated reason


def rules_record(selected: dict[str, list[int]]) -> dict:
    """What the sandbox kernel needs to re-check edits (written to rules.json)."""
    return {"cosine_cap": COSINE_CAP, "length_tolerance": LENGTH_TOLERANCE, "selected": selected}


def edit_problems(original_w_dec: np.ndarray, selected: dict[str, list[int]], feature: int, row) -> list[str]:
    """Every rule the edit breaks; empty if it is valid. `original_w_dec` is latent × input."""
    allowed = set(selected["adoption"]) | set(selected["retention"])
    if feature not in allowed:
        return [f"feature {feature} is not one of the 20 selected features"]
    row = np.asarray(row, dtype=np.float64)
    original = original_w_dec[feature].astype(np.float64)
    if row.shape != original.shape:
        return [f"row has shape {row.shape}, expected {original.shape}"]
    if not np.all(np.isfinite(row)):
        return ["row has non-finite values"]

    problems = []
    length_ratio = np.linalg.norm(row) / np.linalg.norm(original)
    if abs(length_ratio - 1) > LENGTH_TOLERANCE:
        problems.append(f"row length changed by a factor of {length_ratio:.6f} (must stay within 1 ± {LENGTH_TOLERANCE})")
    cosine = cosine_similarity(row, original)
    if cosine < COSINE_CAP:
        problems.append(f"cosine with the original row is {cosine:.4f}, below the cap {COSINE_CAP}")
    return problems


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def apply_edits(original_w_dec: np.ndarray, edits: list[Edit]) -> np.ndarray:
    """The edited decoder: the original with each edit applied in order (a later edit of a row wins)."""
    w_dec = original_w_dec.copy()
    for edit in edits:
        w_dec[edit.feature] = edit.row
    return w_dec


# =====================================================================
# edits.py: the human-readable record of accepted edits
# =====================================================================
# Each row is stored exactly in rows/edit_NNN.npy; edits.py lists the edits in order with the LLM's
# reason as a comment and each row's sha256, so a reader sees what changed and why, and the audit
# can replay the file onto the original SAE.

def write_edits_py(edits: list[Edit], run_dir: Path) -> Path:
    rows_dir = Path(run_dir) / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    lines = ['"""Accepted hand edits of the decoder, in order. Replay with',
             'mas_sae.llm_edit.edit_rules.read_edits_py and apply_edits onto the original SAE."""', "",
             "EDITS = ["]
    for edit in edits:
        row_file = f"rows/edit_{edit.number:03d}.npy"
        np.save(Path(run_dir) / row_file, edit.row.astype(np.float32))
        lines.append(f"    # edit {edit.number}, feature {edit.feature}:")
        lines += [f"    #   {line}" for line in textwrap.wrap(edit.why or "(no reason given)", 90)]
        lines.append(f'    {{"edit": {edit.number}, "feature": {edit.feature}, "row_file": "{row_file}", '
                     f'"row_sha256": "{row_sha256(edit.row)}"}},')
    lines.append("]")
    path = Path(run_dir) / "edits.py"
    path.write_text("\n".join(lines) + "\n")
    return path


def read_edits_py(run_dir: Path) -> list[Edit]:
    """The edits in edits.py, each row checked against its recorded sha256. Parsed, never executed."""
    source = (Path(run_dir) / "edits.py").read_text()
    assignment = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.Assign) and node.targets[0].id == "EDITS")
    edits = []
    for entry in ast.literal_eval(assignment.value):
        row = np.load(Path(run_dir) / entry["row_file"])
        if row_sha256(row) != entry["row_sha256"]:
            raise ValueError(f"{entry['row_file']} does not match the sha256 recorded in edits.py")
        edits.append(Edit(entry["edit"], entry["feature"], row, why=""))
    return edits


def row_sha256(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest()
