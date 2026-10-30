"""Reproducibility metadata for one collection run.

This module only constructs the ``provenance`` block of a resolved config
and the run identity a resumed collection is checked against; it does not
validate or write artifacts (``collection_artifacts``) and it
does not own generic run utilities such as git/package lookups
(``artifacts``). Missing runtime facts remain null.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any

import torch

from mas_sae.agents import critic as critic_prompts
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.data.musique import MUSIQUE_DATASET_ID
from mas_sae.experiments import artifacts
from mas_sae.experiments.conditions import (
    OMITTED,
    A2Presentation,
    resolve_conditions,
    resolve_presentations,
)
from mas_sae.experiments.reproducibility import config_sha256


def experiment_compatibility(resolved_config: dict[str, Any], candidate_sites: list[str]) -> dict[str, Any]:
    """Compare recorded experiment settings while allowing different worker hardware."""
    provenance = resolved_config["provenance"]
    return {
        "roles": {role: {k: metadata.get(k) for k in (
            "model_id", "requested_revision", "resolved_revision", "loader", "dtype",
            "quantization", "generation", "chat_template_kwargs")}
            for role, metadata in provenance["roles"].items()},
        **{k: provenance.get(k) for k in ("protocol_version", "capabilities", "active_conditions", "prompt_versions", "prompt_sha256")},
        "a2_presentations": provenance.get("a2_presentations", ["feedback"]),
        "activation_sites": candidate_sites,
    }


def role_metadata(agent: Any) -> dict[str, Any]:
    """Model identity, revision, precision and placement of one role."""
    spec = getattr(agent, "model_spec", None)
    if not isinstance(spec, dict):
        spec = {}
    model = getattr(agent, "model", None)
    config = getattr(model, "config", None)
    def scalar(value):
        return value if isinstance(value, (str, int, float, bool, list, dict)) else None
    dtype = getattr(model, "dtype", None)
    device = getattr(model, "device", None)
    return {
        "model_id": spec.get("id") or scalar(getattr(config, "_name_or_path", None)),
        "requested_revision": spec.get("revision"),
        "resolved_revision": scalar(getattr(config, "_commit_hash", None)),
        "architecture": scalar(getattr(config, "architectures", None)),
        "loader": spec.get("loader"),
        "dtype": str(dtype) if isinstance(dtype, torch.dtype) else None,
        "device": str(device) if isinstance(device, (str, torch.device)) else None,
        "device_map": scalar(getattr(model, "hf_device_map", None)),
        "requested_placement": spec.get("device_map", spec.get("device")),
        "quantization": scalar(getattr(config, "quantization_config", None)),
        "generation": spec.get("generation", {"max_new_tokens": agent.max_new_tokens, "do_sample": False}),
        "chat_template_kwargs": dict(getattr(agent, "chat_template_kwargs", None) or {}),
    }


def environment_metadata() -> dict[str, Any]:
    """Working-tree and GPU facts; ``git_dirty`` is null when git is unavailable."""
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        git_dirty = bool(result.stdout.strip()) if result.returncode == 0 else None
    except OSError:
        git_dirty = None
    return {
        "git_dirty": git_dirty,
        "cuda_version": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else [],
    }


def build_collection_progress_provenance(
    config: dict[str, Any],
    resolved_config: dict[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Run identity that every resume of a chunked collection must match.

    Built from the raw config and the ``provenance`` block of
    ``build_resolved_config``. Code identity is the HEAD commit plus hashes
    of tracked uncommitted changes and of untracked source files; environment identity is the package
    versions and GPUs, since either can change generated outputs.
    """
    resolved = resolved_config["provenance"]
    return {
        "created_at_utc": artifacts.utc_now().isoformat(),
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest_sha256,
        "dataset_revision": config["dataset"].get("revision"),
        "model_revisions": {
            role: {key: metadata.get(key) for key in ("requested_revision", "resolved_revision")}
            for role, metadata in resolved["roles"].items()
        },
        "git_commit": resolved["git_commit"],
        "git_diff_sha256": artifacts.get_git_diff_sha256(),
        "untracked_code_sha256": artifacts.get_untracked_sha256("src", "scripts"),
        "package_versions": resolved["package_versions"],
        "gpus": resolved["environment"]["gpus"],
        "hash_basis": {
            "config_sha256": "canonical JSON of the collection config without its output section",
            "manifest_sha256": "canonical JSON of ordered {question_id, experiment_split, hop_group[, candidate_order]}",
            "git_diff_sha256": "sha256 of `git diff --binary HEAD` (tracked files only)",
            "untracked_code_sha256": "sha256 of names and bytes of untracked, non-ignored files under src/ and scripts/",
        },
    }


def _critic_prompt_names(
    critic: Any,
    type_checked_target: bool,
) -> dict[str, str]:
    """Names of the critic prompts a run actually used, by capability."""
    names = {
        "critic_natural": (
            "NATURAL_BLIND_PROMPT+NATURAL_COMPARE_PROMPT"
            if getattr(critic, "blind_then_compare", False)
            else "NATURAL_PROMPT_V1"
        ),
        "critic_controlled": (
            "CONTROLLED_OWN_CONCLUSION_PROMPT"
            if getattr(critic, "controlled_as_own_conclusion", False)
            else "CONTROLLED_PROMPT_V1"
        ),
    }

    if type_checked_target:
        names["critic_distractor"] = "DISTRACTOR_PROMPT"

    return names


def _prompt_hashes(names: dict[str, str], solver: Any, validator: Any) -> dict[str, str]:
    """Fingerprint the runtime templates; critic templates live at module scope.

    Default class templates also support lightweight agents used by provenance
    callers. Instance overrides are used whenever the agent supplies them.
    """
    hashes = {}
    for role, name in names.items():
        if role.startswith("critic_"):
            templates = [getattr(critic_prompts, part) for part in name.split("+")]
            hashes[role] = artifacts.sha256_json(templates)
        else:
            agent, defaults = (validator, Validator) if role == "validator" else (solver, Solver)
            template = getattr(agent, name, getattr(defaults, name))
            if role == "solver_candidates":
                # Preserve the established candidate-template-and-label digest.
                labels = getattr(solver, "CANDIDATE_LABELS", Solver.CANDIDATE_LABELS)
                template += json.dumps(labels, sort_keys=True)
            hashes[role] = artifacts.sha256_text(template)
    return hashes


def build_resolved_config(
    config: dict[str, Any],
    model: Any,
    solver: Any,
    critic: Any,
    validator: Any,
    *,
    protocol_version: str | None = None,
    type_checked_target: bool = False,
) -> dict[str, Any]:
    """Add minimal reproducibility provenance to the run config.

    ``protocol_version`` is the caller's opaque run label, recorded as
    given when supplied. The prompt names and ``capabilities`` block are
    derived from the critic's capability flags and ``type_checked_target``,
    so a run records the behaviour that actually produced its feedback.
    """
    capabilities = {
        "blind_then_compare": bool(
            getattr(critic, "blind_then_compare", False)
        ),
        "controlled_as_own_conclusion": bool(
            getattr(critic, "controlled_as_own_conclusion", False)
        ),
        "type_checked_target": bool(type_checked_target),
    }
    label = (
        {} if protocol_version is None
        else {"protocol_version": protocol_version}
    )
    a2_presentations = config.get("collection", {}).get("a2_presentations", OMITTED)
    presentations = resolve_presentations(a2_presentations)
    presentation_record = {}
    if a2_presentations is not OMITTED:
        presentation_record["a2_presentations"] = [p.value for p in presentations]
    candidate_prompt = {}
    if A2Presentation.CANDIDATES in presentations:
        candidate_prompt = {"solver_candidates": "CANDIDATE_PROMPT"}
    prompt_versions = {
        "solver_solve": "SOLVE_PROMPT_V1",
        "solver_revise": "REVISE_PROMPT_V1",
        **candidate_prompt,
        **_critic_prompt_names(critic, type_checked_target),
        "validator": "VALIDATE_PROMPT",
    }

    return {
        **config,
        "provenance": {
            "created_at_utc": artifacts.utc_now().isoformat(),
            "git_commit": artifacts.get_git_commit(),
            "model_revision": (
                getattr(model.config, "_commit_hash", None) or "unknown"
            ),
            "dataset_id": "mixed-manifest" if "inputs" in config.get("dataset", {}) else MUSIQUE_DATASET_ID,
            "roles": {name: role_metadata(agent) for name, agent in
                      (("solver", solver), ("critic", critic), ("validator", validator))},
            "environment": environment_metadata(),
            "active_conditions": [c.value for c in resolve_conditions(config.get("collection", {}).get("active_conditions", OMITTED))],
            **presentation_record,
            "package_versions": {
                "torch": artifacts.get_package_version("torch"),
                "transformers": artifacts.get_package_version("transformers"),
                "datasets": artifacts.get_package_version("datasets"),
            },
            **label,
            "capabilities": capabilities,
            "prompt_versions": prompt_versions,
            "prompt_sha256": _prompt_hashes(prompt_versions, solver, validator),
            "generation": {
                "do_sample": (False if "roles" not in config else None),
                "solver_max_new_tokens": solver.max_new_tokens,
                "critic_max_new_tokens": critic.max_new_tokens,
                "validator_max_new_tokens": validator.max_new_tokens,
            },
        },
    }
