"""What a run was made with, and the frozen run every worker must reproduce.

``frozen_run.json`` (written once, by prepare) holds the config, the study episode order, the feature
sets, the condition list and provenance, plus a hash of each. A worker recomputes all of them from
its own config and inputs and refuses to start if anything differs; it also refuses a different code
state, package versions or input file. The GPU model is recorded but not enforced: each episode's
baseline check on the worker is what guarantees comparability. Merge accepts only shards made for
this frozen run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from mas_sae.experiments.artifacts import (
    get_git_commit, get_package_version, get_run_command, get_untracked_files, get_untracked_sha256,
    sha256_file, sha256_json,
)
from mas_sae.intervention.experiment.representations import checkpoint_sha256

CODE_DIRS = ("src", "scripts", "configs")
PACKAGES = ("torch", "transformers", "numpy", "pandas", "safetensors")
# provenance that must be identical on every worker
IDENTITY_KEYS = ("code_tree_sha", "uncommitted_code_sha256", "untracked_code_sha256", "packages",
                 "solver", "input_sha256", "checkpoint_sha256")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout.decode()


def code_tree_sha() -> dict:
    """Git tree id of each code directory at HEAD: changes when code is committed, not when results are."""
    return {d: _git("rev-parse", f"HEAD:{d}").strip() for d in CODE_DIRS}


def uncommitted_code_sha256() -> str:
    return hashlib.sha256(_git("diff", "--no-ext-diff", "--binary", "HEAD", "--", *CODE_DIRS).encode()).hexdigest()


def input_sha256(config) -> dict:
    i = config.inputs
    files = [i.interactions, i.contexts, i.feature_sets,
             *(str(Path(i.stored_labels) / f) for f in ("run_settings.json", "summary.json", "raw_results.jsonl"))]
    return {f: sha256_file(f) for f in files}


def provenance(config, device_name: str | None) -> dict:
    return {
        "git_commit": get_git_commit(), "code_tree_sha": code_tree_sha(),
        "uncommitted_code_sha256": uncommitted_code_sha256(),
        "untracked_code_files": get_untracked_files(*CODE_DIRS),
        "untracked_code_sha256": get_untracked_sha256(*CODE_DIRS),
        "packages": {p: get_package_version(p) for p in PACKAGES},
        "device_name": device_name, "solver": config.solver, "command": get_run_command(),
        "input_sha256": input_sha256(config),
        "checkpoint_sha256": {name: checkpoint_sha256(spec) for name, spec in config.representations.items()},
    }


def device_name(solver_model) -> str | None:
    import torch
    device = solver_model.model.device
    return torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)


def frozen_content(config, episode_ids: list[str], choose_r_episode_ids: list[str], feature_sets: dict,
                   condition_keys: list[str]) -> dict:
    """Everything that defines the experiment, and a hash of each part."""
    parts = {"config": config.as_dict(), "episode_ids": episode_ids, "choose_r_episode_ids": choose_r_episode_ids,
             "feature_sets": feature_sets, "conditions": condition_keys}
    return {**json.loads(json.dumps(parts)), "hashes": {k: sha256_json(v) for k, v in parts.items()}}


def write_frozen_run(path: Path, content: dict, prov: dict, recorded: dict) -> str:
    """Write ``frozen_run.json`` once; a rerun of prepare must freeze the same content. Returns its sha256.

    ``recorded`` holds facts recorded at freezing that workers do not recompute (the edit size and
    the choose-r report's sha256: workers need not have choose-r's files)."""
    path = Path(path)
    if path.exists():
        require_same_experiment(json.loads(path.read_text()), content, prov, where=str(path))
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**content, "recorded": recorded, "provenance": prov}, indent=2, default=str))
    return sha256_file(path)


def require_same_experiment(frozen: dict, content: dict, prov: dict | None, where: str) -> None:
    """Refuse unless ``content`` equals what ``frozen`` recorded and, if ``prov`` is given (GPU
    workers), so does its identity: code, packages, Solver, inputs and checkpoints (not the GPU model).
    Merge passes no ``prov``: it only labels and analyses, on any machine."""
    changed = [k for k in content["hashes"] if frozen["hashes"].get(k) != content["hashes"][k]]
    if prov is not None:
        prov = json.loads(json.dumps(prov))
        changed += [k for k in IDENTITY_KEYS if frozen["provenance"].get(k) != prov.get(k)]
    if changed:
        raise RuntimeError(f"{where} was frozen with different {changed}; this machine cannot continue that run.")
