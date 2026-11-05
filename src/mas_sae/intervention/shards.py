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


def shard_excluded_path(run_dir: Path, shard: int) -> Path:
    """{episode_id: why}: episodes whose baseline on this worker differed from prepare's."""
    return Path(run_dir) / f"shard_{shard}.excluded.json"


def read_excluded(run_dir: Path, shard: int) -> dict[str, str]:
    path = shard_excluded_path(run_dir, shard)
    return json.loads(path.read_text()) if path.exists() else {}


def record_excluded(run_dir: Path, shard: int, episode_id: str, reason: str) -> dict[str, str]:
    excluded = {**read_excluded(run_dir, shard), episode_id: reason}
    shard_excluded_path(run_dir, shard).write_text(json.dumps(excluded, indent=2))
    return excluded


def expected_keys(episode_ids: list[str], condition_keys: list[str]) -> set[tuple[str, str]]:
    return {(e, c) for e in episode_ids for c in condition_keys}


def missing_and_extra(rows: list[dict], episode_ids: list[str], condition_keys: list[str]) -> tuple[set, set, int]:
    """(missing keys, unexpected keys, number of duplicated keys)."""
    keys = [row_key(r) for r in rows]
    expected = expected_keys(episode_ids, condition_keys)
    return expected - set(keys), set(keys) - expected, len(keys) - len(set(keys))


def without_excluded(rows: list[dict], episode_ids: list[str], excluded: dict) -> tuple[list[dict], list[str]]:
    """Rows and episode ids with the excluded episodes removed (an excluded episode may have rows left
    from an earlier attempt of the worker, before its baseline stopped matching)."""
    return [r for r in rows if r["episode_id"] not in excluded], [e for e in episode_ids if e not in excluded]


def mark_shard_complete(run_dir: Path, shard: int, episode_ids: list[str], condition_keys: list[str],
                        frozen_run_sha256: str, excluded: dict | None = None) -> dict:
    """Write the completion marker, only if every (episode, condition) of the shard's episodes that
    were not excluded is there exactly once. The marker records the exclusions; merge reads them there."""
    excluded = dict(excluded or {})
    if unknown := sorted(set(excluded) - set(episode_ids)):
        raise RuntimeError(f"Shard {shard} excludes episodes that are not in it: {unknown[:3]}.")
    path = shard_rows_path(run_dir, shard)
    rows = read_rows(path)
    kept_rows, kept_ids = without_excluded(rows, episode_ids, excluded)
    missing, extra, duplicated = missing_and_extra(kept_rows, kept_ids, condition_keys)
    if missing or extra or duplicated:
        raise RuntimeError(f"Shard {shard} is not complete: {len(missing)} missing, {len(extra)} unexpected, "
                           f"{duplicated} duplicated rows.")
    marker = {"shard": shard, "row_count": len(rows), "rows_sha256": sha256_file(path),
              "frozen_run_sha256": frozen_run_sha256, "excluded": excluded}
    shard_marker_path(run_dir, shard).write_text(json.dumps(marker, indent=2))
    return marker


def read_complete_shards(run_dir: Path, shards: list[list[str]], condition_keys: list[str],
                         frozen_run_sha256: str) -> tuple[list[dict], dict[str, str]]:
    """(rows, {episode_id: why excluded}): every shard's rows without its excluded episodes, after
    checking each marker and that every condition of every other episode appears exactly once."""
    problems, rows, excluded = [], [], {}
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
        excluded.update(marker.get("excluded", {}))
        rows += shard_rows
    all_ids = [e for ids in shards for e in ids]
    rows, kept_ids = without_excluded(rows, all_ids, excluded)
    missing, extra, duplicated = missing_and_extra(rows, kept_ids, condition_keys)
    if missing or extra or duplicated:
        problems.append(f"{len(missing)} conditions missing, {len(extra)} unexpected, {duplicated} duplicated")
    if problems:
        raise RuntimeError("Cannot merge:\n  - " + "\n  - ".join(problems))
    return rows, excluded
