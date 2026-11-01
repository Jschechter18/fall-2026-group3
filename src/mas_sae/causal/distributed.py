"""Local map-reduce orchestration for causal replay.

Transport is deliberately outside this module. A coordinator and workers may
exchange the preflight bundle and immutable shard directories through S3, but
all scientific validation and generation happen against local files.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from mas_sae.experiments.artifacts import (
    ensure_output_available,
    file_facts,
    sha256_file,
    sha256_json,
    staged_output,
    write_json_atomic,
)

from . import replay


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
SHARDS_DIR = "shards"
COMPLETE_NAME = "complete.json"
PREFLIGHT_FILES = ("replay_config.json", "episodes.csv", "per_row.jsonl", "replay_check.json")


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def _condition_key(value: dict) -> tuple:
    return value["experiment"], value["condition"], value.get("draw")


def _with_manifest_hash(basis: dict) -> dict:
    return {**basis, "manifest_sha256": sha256_json(basis)}


def _preflight_files(preflight_dir: Path) -> dict:
    missing = [name for name in PREFLIGHT_FILES if not (preflight_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Preflight run {preflight_dir} is missing {missing}.")
    return {
        name: {"size": (preflight_dir / name).stat().st_size, "sha256": sha256_file(preflight_dir / name)}
        for name in PREFLIGHT_FILES
    }


def _validate_preflight(preflight_dir: Path) -> tuple[dict, pd.DataFrame, list[dict], dict]:
    """Return a complete, successful baseline replay or fail before sharding."""
    _preflight_files(preflight_dir)
    check = json.loads((preflight_dir / "replay_check.json").read_text())
    if not check.get("passed"):
        raise ValueError(f"Preflight replay check did not pass: {check.get('failures', [])}")

    meta = json.loads((preflight_dir / "replay_config.json").read_text())
    episodes = pd.read_csv(preflight_dir / "episodes.csv", dtype={"question_id": str})
    if episodes.question_id.duplicated().any():
        raise ValueError("Preflight episodes.csv contains duplicate question IDs.")
    if not (preflight_dir / "per_row.jsonl").read_bytes().endswith(b"\n"):
        raise ValueError("Preflight per_row.jsonl ends with an incomplete row.")
    rows = replay.read_rows(preflight_dir / "per_row.jsonl")
    baseline = [c for c in meta.get("conditions", []) if c.get("experiment") == replay.BASELINE]
    if len(baseline) != 1 or _condition_key(baseline[0]) != (replay.BASELINE, "no_sae", None):
        raise ValueError("Preflight metadata must contain exactly one no-SAE baseline condition.")
    condition_keys = [_condition_key(condition) for condition in meta["conditions"]]
    if len(condition_keys) != len(set(condition_keys)):
        raise ValueError("Preflight metadata contains duplicate condition keys.")

    expected = {(str(q), replay.BASELINE, "no_sae", None) for q in episodes.question_id}
    actual = [replay.key(row) for row in rows]
    duplicates = [key for key, count in Counter(actual).items() if count > 1]
    if duplicates or set(actual) != expected or len(rows) != len(expected):
        raise ValueError(
            "Preflight baseline coverage is not exact: "
            f"duplicates={len(duplicates)}, missing={len(expected - set(actual))}, "
            f"extra={len(set(actual) - expected)}."
        )
    if check.get("n_episodes") != len(episodes) or check.get("n_planned") != len(episodes):
        raise ValueError("Preflight replay check does not cover every frozen episode.")
    return meta, episodes, rows, check


def prepare_distributed_run(
    preflight_dir: str | Path,
    *,
    num_shards: int,
    output_dir: str | Path | None = None,
) -> Path:
    """Freeze balanced episode shards from one successful global preflight."""
    preflight_dir = Path(preflight_dir).resolve()
    meta, episodes, _, _ = _validate_preflight(preflight_dir)
    if type(num_shards) is not int or not 1 <= num_shards <= len(episodes):
        raise ValueError("num_shards must be between 1 and the preflight episode count.")

    output_dir = Path(output_dir).resolve() if output_dir else preflight_dir / "distributed"
    ensure_output_available(output_dir)
    question_ids = episodes.question_id.astype(str).tolist()
    width, remainder = divmod(len(question_ids), num_shards)
    shards = []
    start = 0
    for shard_id in range(num_shards):
        end = start + width + (shard_id < remainder)
        shards.append({"shard_id": shard_id, "question_ids": question_ids[start:end]})
        start = end

    provenance = meta.get("provenance") or {}
    basis = {
        "schema_version": SCHEMA_VERSION,
        # A relative path keeps a downloaded preflight/distributed bundle portable.
        "preflight_relative_path": os.path.relpath(preflight_dir, output_dir),
        "preflight_files": _preflight_files(preflight_dir),
        "scientific_identity": {key: provenance.get(key) for key in replay.IDENTITY_KEYS},
        "episode_order": question_ids,
        "conditions_sha256": sha256_json(meta["conditions"]),
        "features_sha256": sha256_json(meta["features"]),
        "shards": shards,
    }
    manifest = _with_manifest_hash(basis)
    output_dir.mkdir(parents=True)
    path = output_dir / MANIFEST_NAME
    write_json_atomic(path, manifest, overwrite=False)
    logger.info("prepared %d episodes in %d causal replay shards: %s", len(question_ids), num_shards, path)
    return path


def read_manifest(path: str | Path) -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text())
    basis = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != SCHEMA_VERSION or sha256_json(basis) != manifest.get("manifest_sha256"):
        raise ValueError(f"Distributed manifest changed or has an unsupported schema: {path}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Distributed manifest has no shards.")
    ids = [shard.get("shard_id") for shard in shards]
    if ids != list(range(len(shards))):
        raise ValueError("Distributed shard IDs must be contiguous from zero.")
    assigned = [str(q) for shard in shards for q in shard.get("question_ids", [])]
    expected = [str(q) for q in manifest.get("episode_order", [])]
    sizes = [len(shard.get("question_ids", [])) for shard in shards]
    if (any(not isinstance(q, str) for shard in shards for q in shard.get("question_ids", []))
            or assigned != expected or len(assigned) != len(set(assigned))):
        raise ValueError("Distributed shards must cover the frozen episode order exactly once.")
    if not sizes or min(sizes) <= 0 or max(sizes) - min(sizes) > 1:
        raise ValueError("Distributed shards must be nonempty and balanced within one episode.")
    return manifest


def preflight_directory(manifest_path: str | Path, manifest: dict | None = None) -> Path:
    manifest_path = Path(manifest_path).resolve()
    manifest = manifest or read_manifest(manifest_path)
    return (manifest_path.parent / manifest["preflight_relative_path"]).resolve()


def _verified_preflight(manifest_path: Path, manifest: dict) -> tuple[Path, dict, pd.DataFrame, list[dict]]:
    preflight_dir = preflight_directory(manifest_path, manifest)
    current = _preflight_files(preflight_dir)
    if current != manifest["preflight_files"]:
        raise ValueError("Preflight artifacts differ from the frozen distributed manifest.")
    meta, episodes, baseline, _ = _validate_preflight(preflight_dir)
    if episodes.question_id.astype(str).tolist() != manifest["episode_order"]:
        raise ValueError("Preflight episode order differs from the distributed manifest.")
    if sha256_json(meta["conditions"]) != manifest["conditions_sha256"]:
        raise ValueError("Preflight conditions differ from the distributed manifest.")
    if sha256_json(meta["features"]) != manifest["features_sha256"]:
        raise ValueError("Preflight features differ from the distributed manifest.")
    return preflight_dir, meta, episodes, baseline


def shard_directory(manifest_path: str | Path, shard_id: int) -> Path:
    return Path(manifest_path).resolve().parent / SHARDS_DIR / f"{shard_id:06d}"


def _shard(manifest: dict, shard_id: int) -> dict:
    if type(shard_id) is not int or not 0 <= shard_id < len(manifest["shards"]):
        raise ValueError(f"shard must be an integer in [0, {len(manifest['shards'])}).")
    return manifest["shards"][shard_id]


def _validate_row_keys(rows: list[dict], question_ids: list[str], condition_keys: list[tuple], *, complete: bool) -> set:
    expected = {(qid, *condition) for qid in question_ids for condition in condition_keys}
    keys = [replay.key(row) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    actual = set(keys)
    extra = actual - expected
    missing = expected - actual
    if duplicates or extra or (complete and missing):
        raise ValueError(
            "Shard row coverage is invalid: "
            f"duplicates={len(duplicates)}, missing={len(missing)}, extra={len(extra)}."
        )
    return expected


def _validate_condition_metadata(rows: list[dict], conditions: list[dict]) -> None:
    """Ensure a row's recorded intervention parameters match its frozen condition."""
    expected = {_condition_key(condition): condition for condition in conditions}
    fields = ("mode", "strength", "fraction", "edit_norm")
    for row in rows:
        condition = expected.get(_condition_key(row))
        if condition is None or any(row.get(field) != condition.get(field) for field in fields):
            raise ValueError(
                f"{row.get('question_id')}: row metadata differs from frozen condition "
                f"{_condition_key(row)}."
            )


def _validate_worker_baseline(rows: list[dict], preflight_rows: list[dict], question_ids: list[str]) -> dict:
    """Check that a worker reproduces the coordinator baseline before editing activations."""
    wanted = set(question_ids)
    worker = {row["question_id"]: row for row in rows
              if row["question_id"] in wanted and row["experiment"] == replay.BASELINE}
    preflight = {row["question_id"]: row for row in preflight_rows if row["question_id"] in wanted}
    if set(worker) != wanted or set(preflight) != wanted:
        raise ValueError("Worker baseline parity check is missing assigned episodes.")
    fields = ("input_ids_sha256", "solver_attempt_2", "solver_response", "first_token_id")
    mismatches = {
        question_id: [field for field in fields if worker[question_id].get(field) != preflight[question_id].get(field)]
        for question_id in question_ids
    }
    mismatches = {question_id: fields for question_id, fields in mismatches.items() if fields}
    if mismatches:
        sample = dict(list(mismatches.items())[:3])
        raise ValueError(
            f"Worker baseline differs from the successful preflight for {len(mismatches)} episodes, e.g. {sample}."
        )
    return {"passed": True, "n_episodes": len(question_ids), "matched_fields": list(fields)}


def _validate_completed_shard(root: Path, manifest: dict, shard: dict) -> dict:
    complete_path = root / COMPLETE_NAME
    if not complete_path.is_file():
        raise FileNotFoundError(f"Incomplete causal replay shard: {root}")
    completed = json.loads(complete_path.read_text())
    expected_identity = {
        "manifest_sha256": manifest["manifest_sha256"],
        "shard_id": shard["shard_id"],
        "question_ids": shard["question_ids"],
    }
    if any(completed.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"{root}: completion marker belongs to another manifest or shard.")
    if completed.get("files") != file_facts(root, exclude_names={COMPLETE_NAME}):
        raise ValueError(f"{root}: completed shard files changed.")
    worker = json.loads((root / "worker.json").read_text())
    if any(worker.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"{root}: worker identity disagrees with its completion marker.")
    if {key: worker.get("provenance", {}).get(key) for key in replay.IDENTITY_KEYS} != manifest["scientific_identity"]:
        raise ValueError(f"{root}: worker scientific identity differs from the manifest.")
    return completed


def run_shard(manifest_path: str | Path, shard_id: int, *, resume: bool = False) -> Path:
    """Map one frozen episode shard using the existing replay generation path."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    shard = _shard(manifest, shard_id)
    root = shard_directory(manifest_path, shard_id)
    if (root / COMPLETE_NAME).exists():
        _validate_completed_shard(root, manifest, shard)
        if resume:
            logger.info("causal replay shard %d is already complete", shard_id)
            return root
        raise FileExistsError(f"{root} is complete; use --resume to verify and skip it.")
    if root.exists() and not resume:
        raise FileExistsError(f"{root} exists; use --resume to continue it.")

    _, meta, _, baseline_rows = _verified_preflight(manifest_path, manifest)
    cfg = replay.ReplayConfig.from_dict(meta["config"], source=str(manifest_path))
    episodes = replay.load_episodes(cfg)
    if episodes.question_id.astype(str).tolist() != manifest["episode_order"]:
        raise ValueError("Current causal inputs do not reproduce the frozen episode order.")
    assigned = set(shard["question_ids"])
    shard_episodes = episodes[episodes.question_id.astype(str).isin(assigned)].copy()
    if shard_episodes.question_id.astype(str).tolist() != shard["question_ids"]:
        raise ValueError("Current causal inputs do not reproduce this shard's episode order.")

    comp = replay.load_components(cfg, shard_episodes)
    comp.info["label_policy_name"] = cfg.label_policy
    conditions = replay.configured_conditions(cfg, comp.features)
    if _jsonable(conditions) != meta["conditions"] or _jsonable(comp.features) != meta["features"]:
        raise ValueError("Worker resolved different conditions or feature sets from the preflight.")
    worker_provenance = replay.provenance(cfg, None, comp)
    replay.validate_provenance_identity(
        manifest["scientific_identity"], worker_provenance, context=f"causal replay shard {shard_id}"
    )

    identity = {
        "manifest_sha256": manifest["manifest_sha256"],
        "shard_id": shard_id,
        "question_ids": shard["question_ids"],
    }
    worker = {**identity, "provenance": worker_provenance}
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        worker_path = root / "worker.json"
        if not worker_path.exists():
            if any(root.iterdir()):
                raise ValueError(f"Cannot resume {root}: worker.json is missing from a nonempty shard directory.")
            write_json_atomic(worker_path, _jsonable(worker), overwrite=False)
        saved = json.loads(worker_path.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Cannot resume {root}: worker manifest or shard changed.")
        replay.validate_provenance_identity(
            saved.get("provenance") or {}, worker_provenance, context=f"causal replay shard {shard_id}"
        )
    else:
        root.mkdir()
        write_json_atomic(root / "worker.json", _jsonable(worker), overwrite=False)

    output = root / "per_row.jsonl"
    replay.repair_tail(output)
    rows = replay.read_rows(output)
    all_condition_keys = [_condition_key(condition) for condition in conditions]
    _validate_row_keys(rows, shard["question_ids"], all_condition_keys, complete=False)
    _validate_condition_metadata(rows, conditions)

    baseline_condition = next(condition for condition in conditions if condition["experiment"] == replay.BASELINE)
    done = {replay.key(row) for row in rows}
    baseline_todo = [
        (episode, baseline_condition)
        for episode in shard_episodes.to_dict("records")
        if (episode["question_id"], *_condition_key(baseline_condition)) not in done
    ]
    first = {replay.key(row): row.get("first_token_id") for row in rows}
    if baseline_todo:
        replay.generate_all(baseline_todo, comp, output, first, f"shard {shard_id} baseline")
    rows = replay.read_rows(output)
    baseline_check = _validate_worker_baseline(rows, baseline_rows, shard["question_ids"])
    write_json_atomic(root / "baseline_check.json", baseline_check)

    nonbaseline = [condition for condition in conditions if condition["experiment"] != replay.BASELINE]
    done = {replay.key(row) for row in rows}
    todo = [
        (episode, condition)
        for episode in shard_episodes.to_dict("records")
        for condition in nonbaseline
        if (episode["question_id"], *_condition_key(condition)) not in done
    ]
    logger.info(
        "shard %d: %d episodes x %d intervention conditions; %d done, %d to go",
        shard_id, len(shard_episodes), len(nonbaseline),
        sum(row["experiment"] != replay.BASELINE for row in rows), len(todo),
    )
    replay.generate_all(todo, comp, output, first, f"shard {shard_id} condition", cfg.max_hours)

    final_rows = replay.read_rows(output)
    expected = _validate_row_keys(final_rows, shard["question_ids"], all_condition_keys, complete=True)
    _validate_condition_metadata(final_rows, conditions)
    completed = {
        **identity,
        "n_rows": len(final_rows),
        "n_expected_rows": len(expected),
        "files": file_facts(root, exclude_names={COMPLETE_NAME}),
    }
    # This is the local/S3 commit marker: publish it only after every payload file is durable.
    write_json_atomic(root / COMPLETE_NAME, completed, overwrite=False)
    logger.info("completed causal replay shard %d: %s", shard_id, root)
    return root


def reduce_shards(manifest_path: str | Path, *, destination: str | Path | None = None) -> Path:
    """Reduce complete map shards into one validated, analysed replay run."""
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path)
    preflight_dir, meta, episodes, baseline = _verified_preflight(manifest_path, manifest)
    destination = Path(destination).resolve() if destination else manifest_path.parent / "merged"
    ensure_output_available(destination)

    expected_roots = {f"{shard['shard_id']:06d}" for shard in manifest["shards"]}
    shards_root = manifest_path.parent / SHARDS_DIR
    actual_roots = {path.name for path in shards_root.iterdir() if path.is_dir()} if shards_root.exists() else set()
    if actual_roots != expected_roots:
        raise ValueError(
            f"Shard set mismatch; missing={sorted(expected_roots - actual_roots)}, "
            f"extra={sorted(actual_roots - expected_roots)}"
        )

    all_condition_keys = [_condition_key(condition) for condition in meta["conditions"]]
    worker_rows: list[dict] = []
    worker_metadata = []
    for shard in manifest["shards"]:
        root = shard_directory(manifest_path, shard["shard_id"])
        complete = _validate_completed_shard(root, manifest, shard)
        rows = replay.read_rows(root / "per_row.jsonl")
        _validate_row_keys(rows, shard["question_ids"], all_condition_keys, complete=True)
        _validate_condition_metadata(rows, meta["conditions"])
        _validate_worker_baseline(rows, baseline, shard["question_ids"])
        worker_rows.extend(rows)
        worker_metadata.append({"complete": complete, "worker": json.loads((root / "worker.json").read_text())})

    rows = worker_rows
    _validate_row_keys(rows, manifest["episode_order"], all_condition_keys, complete=True)
    _validate_condition_metadata(rows, meta["conditions"])
    episode_order = {question_id: index for index, question_id in enumerate(manifest["episode_order"])}
    condition_order = {key: index for index, key in enumerate(all_condition_keys)}
    rows.sort(key=lambda row: (episode_order[row["question_id"]], condition_order[_condition_key(row)]))

    with staged_output(destination) as stage:
        for name in ("replay_config.json", "episodes.csv"):
            source = preflight_dir / name
            if source.exists():
                shutil.copy2(source, stage / name)
        for source_name, destination_name in (
            ("replay_check.json", "preflight_replay_check.json"),
            ("replay_check_mismatches.csv", "preflight_replay_check_mismatches.csv"),
        ):
            source = preflight_dir / source_name
            if source.exists():
                shutil.copy2(source, stage / destination_name)
        with (stage / "per_row.jsonl").open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        worker_check = replay.replay_check(
            pd.DataFrame(rows), episodes,
            meta["config"].get("min_replay_label_match", replay.ReplayConfig.min_replay_label_match),
        )
        if not worker_check["passed"]:
            raise ValueError(f"Merged worker replay check failed: {worker_check['failures']}")
        write_json_atomic(stage / "replay_check.json", worker_check)
        replay.replay_mismatches(pd.DataFrame(rows), episodes).to_csv(
            stage / "replay_check_mismatches.csv", index=False,
        )
        write_json_atomic(stage / "distributed_run.json", {
            "manifest_sha256": manifest["manifest_sha256"],
            "num_shards": len(manifest["shards"]),
            "n_episodes": len(episodes),
            "n_rows": len(rows),
            "workers": worker_metadata,
        })
        replay.analyse(stage)
    logger.info("reduced %d causal replay shards: %s", len(manifest["shards"]), destination)
    return destination
