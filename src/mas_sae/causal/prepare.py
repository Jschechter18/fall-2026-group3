"""Prepare replay label/reference artifacts using the existing production builders."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from mas_sae.causal import replay
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.probe.feature_selection import sae_files

logger = logging.getLogger(__name__)
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def prepare_inputs(cfg: replay.ReplayConfig) -> None:
    """Build missing artifacts, validate existing ones, never overwrite or load the Solver.

    Scientific input paths come only from the replay YAML. Builders retain their own
    frozen-partition, checkpoint, label-version and overwrite checks.
    """
    root = Path(cfg.capstone_data_root).resolve()
    prep = cfg.input_preparation
    collection = prep.get("collection_results")
    if collection:
        collection = root / collection
        expected = collection / cfg.partition / "interactions.jsonl"
        if expected.resolve() != (root / cfg.interactions_jsonl).resolve():
            raise ValueError("input_preparation.collection_results does not contain the configured replay interactions.")
        labels = root / cfg.labels_csv
        if labels.name != "labels.csv":
            raise ValueError("The production label builder writes labels.csv; use that filename in labels_csv.")
        if not labels.exists():
            if cfg.label_policy == "claude":
                from .semantic import read_source
                read_source(cfg)  # never generate rule-based substitutes for missing Claude outputs
                subprocess.run([sys.executable, str(SCRIPTS / "labels" / "export_final_labels_for_probes.py"),
                                "--claude-runs", str(root / cfg.claude["source_run"]),
                                "--full-results", str(collection), "--layout", "partitioned",
                                "--coverage", "subset", "--output", str(labels.parent)], cwd=root, check=True)
            else:
                subprocess.run([sys.executable, str(SCRIPTS / "prepare_behavior_v01.py"),
                                "--full-results", str(collection), "--layout", "partitioned",
                                "--coverage", "subset", "--label-policy", cfg.label_policy,
                                "--output", str(labels.parent)], cwd=root, check=True)
    # This verifies stored labels and their rule version before the more expensive reference encoding.
    episodes = replay.load_episodes(cfg)
    logger.info("Validated %d selected replay episodes under %s", len(episodes), cfg.label_policy)
    selection_config = prep.get("feature_selection_config")
    if cfg.random_controls == "contribution" or cfg.steer_strengths:
        if not cfg.feature_reference:
            raise ValueError("Contribution controls/steering need feature_reference.")
        reference = root / cfg.feature_reference
        summary = root / cfg.feature_selection_summary
        if reference.resolve() != (summary.parent / "feature_reference_activations.json").resolve():
            raise ValueError("The reference builder writes feature_reference_activations.json beside the summary.")
        if not reference.exists():
            if not selection_config:
                raise ValueError("Set input_preparation.feature_selection_config to build missing references.")
            subprocess.run([sys.executable, str(SCRIPTS / "feature_reference_activations.py"),
                            "--config", str(root / selection_config), "--summary", str(summary)],
                           cwd=root, check=True)
    hashes = {name: {str(p.relative_to(root)): sha256_file(p) for p in sae_files(spec, root)}
              for name, spec in cfg.saes.items()}
    features = replay.feature_sets(cfg, hashes)
    replay.load_probe_direction(cfg)
    logger.info("Inputs ready: %d representations, %d conditions per episode", len(features),
                len(replay.configured_conditions(cfg, features)))
