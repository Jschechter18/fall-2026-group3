from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import yaml

from mas_sae.data.activation_store import ActivationStore
from mas_sae.data.musique import MUSIQUE_DATASET_ID
from mas_sae.experiments.collection import stack_site_activations
from mas_sae.experiments.records import write_jsonl


def get_git_sha() -> str:
    """Return the current Git commit or 'unknown' outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def get_package_version(package: str) -> str:
    """Return an installed package version when available."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def ensure_output_available(
    result_root: str | Path,
    run_name: str,
    source_split: str,
) -> None:
    """Prevent an existing collection result from being overwritten."""
    output_dir = Path(result_root) / run_name / source_split

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists at {output_dir}. "
            "Remove it or choose a different run_name."
        )


def build_resolved_config(
    config: dict[str, Any],
    model: Any,
    solver: Any,
    critic: Any,
    validator: Any,
) -> dict[str, Any]:
    """Add minimal reproducibility provenance to the run config."""
    return {
        **config,
        "provenance": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_sha": get_git_sha(),
            "model_revision": (
                getattr(model.config, "_commit_hash", None) or "unknown"
            ),
            "dataset_id": MUSIQUE_DATASET_ID,
            "package_versions": {
                "torch": get_package_version("torch"),
                "transformers": get_package_version("transformers"),
                "datasets": get_package_version("datasets"),
            },
            "prompt_versions": {
                "solver_solve": "SOLVE_PROMPT_V1",
                "solver_revise": "REVISE_PROMPT_V1",
                "critic_natural": "NATURAL_PROMPT_V1",
                "critic_controlled": "CONTROLLED_PROMPT_V1",
                "validator": "VALIDATE_PROMPT",
            },
            "generation": {
                "do_sample": False,
                "solver_max_new_tokens": solver.max_new_tokens,
                "critic_max_new_tokens": critic.max_new_tokens,
                "validator_max_new_tokens": validator.max_new_tokens,
            },
        },
    }


def site_slug(module_name: str) -> str:
    """Convert a model layer module name to its storage directory name."""
    layer = module_name.rsplit(".", 1)[-1]

    try:
        return f"layer_{int(layer):02d}"
    except ValueError as error:
        raise ValueError(
            f"Could not determine layer number from {module_name!r}."
        ) from error


def _save_and_verify(
    store: ActivationStore,
    split: str,
    tensor: torch.Tensor,
) -> None:
    """Save one activation tensor and verify its local round trip."""
    store.save_activations(split, tensor)
    loaded = store.load_activations(split)

    if not torch.equal(tensor.detach().cpu(), loaded):
        raise RuntimeError(
            f"Activation round trip failed for split {split!r}."
        )


def _add_sae_indices(
    records: list[dict[str, Any]],
    num_attempt1: int,
) -> list[dict[str, Any]]:
    """Add row indices for the combined SAE activation tensor."""
    enriched = []

    for record in records:
        row = dict(record)
        row["sae_attempt1_index"] = int(row["attempt1_activation_index"])
        row["sae_attempt2_index"] = (
            num_attempt1 + int(row["attempt2_activation_index"])
        )
        enriched.append(row)

    return enriched


def save_collection_artifacts(
    *,
    activation_root: str | Path,
    result_root: str | Path,
    run_name: str,
    source_split: str,
    candidate_sites: list[str],
    attempt1_by_site: dict[str, list[torch.Tensor]],
    attempt2_by_site: dict[str, list[torch.Tensor]],
    records: list[dict[str, Any]],
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    """Save activation tensors, metadata, resolved config, and summary."""
    attempt1 = stack_site_activations(attempt1_by_site)
    attempt2 = stack_site_activations(attempt2_by_site)
    expected_sites = set(candidate_sites)

    if set(attempt1) != expected_sites:
        raise ValueError(
            "Attempt 1 activation sites do not match candidate sites."
        )
    if set(attempt2) != expected_sites:
        raise ValueError(
            "Attempt 2 activation sites do not match candidate sites."
        )

    num_attempt1: int | None = None
    num_attempt2: int | None = None
    site_shapes: dict[str, dict[str, list[int]]] = {}

    for site in candidate_sites:
        attempt1_tensor = attempt1[site]
        attempt2_tensor = attempt2[site]

        if attempt1_tensor.shape[1] != attempt2_tensor.shape[1]:
            raise ValueError(f"Hidden dimensions do not match for {site!r}.")

        if num_attempt1 is None:
            num_attempt1 = attempt1_tensor.shape[0]
            num_attempt2 = attempt2_tensor.shape[0]
        elif (
            attempt1_tensor.shape[0] != num_attempt1
            or attempt2_tensor.shape[0] != num_attempt2
        ):
            raise ValueError(
                "Activation row counts differ across candidate sites."
            )

        sae_tensor = torch.cat([attempt1_tensor, attempt2_tensor], dim=0)
        slug = site_slug(site)
        store = ActivationStore(Path(activation_root) / run_name / slug)

        _save_and_verify(
            store, f"{source_split}_attempt1", attempt1_tensor
        )
        _save_and_verify(
            store, f"{source_split}_attempt2", attempt2_tensor
        )
        _save_and_verify(store, source_split, sae_tensor)

        site_shapes[slug] = {
            "attempt1": list(attempt1_tensor.shape),
            "attempt2": list(attempt2_tensor.shape),
            "sae": list(sae_tensor.shape),
        }

    if num_attempt1 is None or num_attempt2 is None:
        raise ValueError("No activation sites were saved.")

    enriched_records = _add_sae_indices(records, num_attempt1)
    accepted = sum(
        row["solver_accepted_feedback"] is True for row in records
    )
    rejected = sum(
        row["solver_accepted_feedback"] is False for row in records
    )
    noncommittal = sum(
        row["solver_accepted_feedback"] is None for row in records
    )

    output_dir = Path(result_root) / run_name / source_split
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "interactions.jsonl", enriched_records)

    with (output_dir / "resolved_config.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(resolved_config, file, sort_keys=False)

    summary = {
        "run_name": run_name,
        "source_split": source_split,
        "num_questions": num_attempt1,
        "num_episodes": num_attempt2,
        "sae_rows_per_layer": num_attempt1 + num_attempt2,
        "accepted": accepted,
        "rejected": rejected,
        "unlabeled_noncommittal": noncommittal,
        "site_shapes": site_shapes,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    return summary
