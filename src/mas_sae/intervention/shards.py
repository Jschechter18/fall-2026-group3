"""Splitting the study episodes across GPU workers, marking a shard complete, and checking shards at merge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mas_sae.experiments.artifacts import sha256_file
from mas_sae.intervention.results_file import read_rows, row_key


def split_into_shards(episode_ids: list[str], number_of_shards: int) -> list[list[str]]:
    """Contiguous, balanced pieces of the frozen episode order."""
    if number_of_shards > len(episode_ids):
        raise ValueError(f"{number_of_shards} shards for {len(episode_ids)} episodes.")
    return [list(piece) for piece in np.array_split(np.array(episode_ids, dtype=object), number_of_shards)]


def shard_rows_path(run_dir: Path, shard: int) -> Path:
    return Path(run_dir) / f"shard_{shard}.jsonl"


def shard_marker_path(run_dir: Path, shard: int) -> Path:
    return Path(run_dir) / f"shard_{shard}.complete.json"


def expected_keys(episode_ids: list[str], condition_keys: list[str]) -> set[tuple[str, str]]:
    return {(e, c) for e in episode_ids for c in condition_keys}


def missing_and_extra(rows: list[dict], episode_ids: list[str], condition_keys: list[str]) -> tuple[set, set, int]:
    """(missing keys, unexpected keys, number of duplicated keys)."""
    keys = [row_key(r) for r in rows]
    expected = expected_keys(episode_ids, condition_keys)
    return expected - set(keys), set(keys) - expected, len(keys) - len(set(keys))


def mark_shard_complete(run_dir: Path, shard: int, episode_ids: list[str], condition_keys: list[str],
                        frozen_run_sha256: str) -> dict:
    """Write the completion marker, only if every (episode, condition) of the shard is there exactly once."""
    path = shard_rows_path(run_dir, shard)
    missing, extra, duplicated = missing_and_extra(read_rows(path), episode_ids, condition_keys)
    if missing or extra or duplicated:
        raise RuntimeError(f"Shard {shard} is not complete: {len(missing)} missing, {len(extra)} unexpected, "
                           f"{duplicated} duplicated rows.")
    marker = {"shard": shard, "row_count": len(episode_ids) * len(condition_keys),
              "rows_sha256": sha256_file(path), "frozen_run_sha256": frozen_run_sha256}
    shard_marker_path(run_dir, shard).write_text(json.dumps(marker, indent=2))
    return marker


def read_complete_shards(run_dir: Path, shards: list[list[str]], condition_keys: list[str],
                         frozen_run_sha256: str) -> list[dict]:
    """Every shard's rows, after checking each marker and that every condition appears exactly once."""
    problems, rows = [], []
    for shard, episode_ids in enumerate(shards):
        marker_path, rows_path = shard_marker_path(run_dir, shard), shard_rows_path(run_dir, shard)
        if not marker_path.exists():
            problems.append(f"shard {shard} has no completion marker")
            continue
        marker = json.loads(marker_path.read_text())
        if marker["frozen_run_sha256"] != frozen_run_sha256:
            problems.append(f"shard {shard} was generated for a different frozen run")
        if marker["rows_sha256"] != sha256_file(rows_path):
            problems.append(f"shard {shard}'s rows changed after it was marked complete")
        shard_rows = read_rows(rows_path)
        if len(shard_rows) != marker["row_count"]:
            problems.append(f"shard {shard} has {len(shard_rows)} rows, its marker says {marker['row_count']}")
        rows += shard_rows
    all_ids = [e for ids in shards for e in ids]
    missing, extra, duplicated = missing_and_extra(rows, all_ids, condition_keys)
    if missing or extra or duplicated:
        problems.append(f"{len(missing)} conditions missing, {len(extra)} unexpected, {duplicated} duplicated")
    if problems:
        raise RuntimeError("Cannot merge:\n  - " + "\n  - ".join(problems))
    return rows
