"""Prepare one mixed question pool without assigning train/validation/test."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from mas_sae.data.musique import assign_experiment_splits
from mas_sae.experiments.artifacts import sha256_json
from mas_sae.experiments.conditions import OMITTED, A2Presentation, assign_candidate_orders, resolve_conditions, resolve_presentations
from mas_sae.models.roles import resolve_roles

def _read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("data", value.get("questions"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON list or JSONL records")
    return value


def build_master_dataset(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine versioned inputs without changing their question payloads.

    Each input names ``source``, ``version``, and ``path``. Optional
    ``source_split``, ``id_field`` and ``hop_field`` describe, rather than
    transform, the supplied data. Global ids are ``source:original_id``.

    Synthetic-input contract (not yet verified against a real release): the
    producer supplies schema-validated, deduplicated MuSiQue-compatible records.
    This importer checks identity/metadata, not schema or question quality.
    Synthetic input is optional; a MuSiQue-only input list is supported today.
    """
    rows: list[dict[str, Any]] = []
    seen_global: set[str] = set()
    input_records = []

    for spec in inputs:
        source = spec["source"]
        version = str(spec["version"]) if spec["version"] is not None else ""
        if (not isinstance(source, str) or not source.strip() or ":" in source
                or not isinstance(version, str) or not version.strip()):
            raise ValueError("input source/version must be non-empty and source may not contain ':'")

        path = Path(spec["path"])
        raw = path.read_bytes()
        records = _read_records(path)
        input_records.append({"source": source, "version": version, "path": str(path),
                              "sha256": hashlib.sha256(raw).hexdigest(), "count": len(records)})
        id_field = spec.get("id_field", "id")

        for source_index, payload in enumerate(records):
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: record {source_index} is not an object")
            original_id = str(payload.get(id_field) if payload.get(id_field) is not None else "")
            if not original_id:
                raise ValueError(f"{path}: record {source_index} has no {id_field!r}")
            global_id = f"{source}:{original_id}"
            if global_id in seen_global:
                raise ValueError(f"duplicate input id {global_id!r}")
            seen_global.add(global_id)
            hop_field = spec.get("hop_field")
            # No real synthetic release exists yet. MuSiQue-compatible schema
            # alone does not promise a hop_group field: agree on hop metadata
            # with the producer before enabling synthetic inputs.
            hop = payload.get(hop_field) if hop_field else spec.get("hop")
            if hop is None and source == "musique":
                # Parse before namespacing and persist the result; consumers never parse global ids.
                from mas_sae.data.musique import hop_group
                hop = hop_group(original_id)
            if hop is None:
                raise ValueError(f"{global_id}: explicit hop metadata is required")
            required = {"question": str, "paragraphs": list, "answer": str}
            if any(not isinstance(payload.get(k), t) for k, t in required.items()):
                raise ValueError(f"{global_id}: expected question, paragraphs, and answer collection fields")
            if any(not isinstance(p, dict) or not isinstance(p.get("title"), str)
                   or not isinstance(p.get("paragraph_text"), str) for p in payload["paragraphs"]):
                raise ValueError(f"{global_id}: paragraphs require title and paragraph_text strings")
            if not isinstance(payload.get("answer_aliases", []), (list, type(None))):
                raise ValueError(f"{global_id}: answer_aliases must be a list or null")
            group_field = spec.get("group_field", "split_group_id")
            group = payload.get(group_field, global_id)
            if not isinstance(group, str) or not group:
                raise ValueError(f"{global_id}: split group must be a nonempty string")
            generation = payload.get("generation_provenance", spec.get("generation_provenance"))
            row = {"question_id": global_id, "original_id": original_id,
                   "dataset_source": source, "dataset_version": version,
                   "source_split": str(payload.get("source_split", spec.get("source_split", ""))),
                   "hop_group": str(hop), "source_index": source_index, "split_group_id": group,
                   "payload": payload}
            if generation is not None:
                row["generation_provenance"] = generation
            for field in ("in_qc100", "in_development_data", "prior_development_exposure"):
                if field in payload:
                    row[field] = payload[field]
            rows.append(row)
    if not rows:
        raise ValueError("Dataset has no questions")
    rows.sort(key=lambda r: (r["dataset_source"], r["dataset_version"],
                             r["source_split"], r["original_id"]))
    for i, row in enumerate(rows): row["master_index"] = i
    return {"inputs": input_records, "questions": rows}


def collection_settings(config):
    """Settings that must agree across workers; paths and placement are operational."""
    roles = resolve_roles(config)
    roles = {name: {k: v for k, v in spec.items()
                    if k not in {"device", "device_map"}} for name, spec in roles.items()}
    collection = dict(config["collection"])
    collection.setdefault("protocol_version", "v1")
    collection["active_conditions"] = [v.value for v in resolve_conditions(collection.get("active_conditions", OMITTED))]
    collection["a2_presentations"] = [v.value for v in resolve_presentations(collection.get("a2_presentations", OMITTED))]
    if len(collection["active_conditions"]) != 1 or len(collection["a2_presentations"]) != 1:
        raise ValueError("Manifest collection requires one condition and one A2 presentation per question")
    return {"roles": roles, "collection": collection}


def assign_families(questions, proportions, seed, *, candidate_strata=False):
    """Allocate explicit families using the established shuffle and rounding rules."""
    families = defaultdict(list)
    for row in questions:
        families[row["split_group_id"]].append(row)
    representatives = [min(rows, key=lambda r: r["question_id"]) for _, rows in sorted(families.items())]
    groups = {r["question_id"]: json.dumps([r["dataset_source"], r["hop_group"],
              r.get("candidate_order") if candidate_strata else None]) for r in representatives}
    assignments = assign_experiment_splits(
        sorted(groups), proportions=proportions, seed=seed,
        supported_splits=tuple(proportions), hop_groups=groups)
    return {row["question_id"]: assignments[min(rows, key=lambda r: r["question_id"])["question_id"]]
            for rows in families.values() for row in rows}


def prepare_manifest(config, *, num_chunks):
    """Reserve intervention, then freeze randomness and work slices for every question."""
    settings = collection_settings(config)
    master = build_master_dataset(config["dataset"]["inputs"])
    questions = master["questions"]
    if type(num_chunks) is not int or not 1 <= num_chunks <= len(questions):
        raise ValueError("num_chunks must be between 1 and the question count")
    run_name = config["output"]["run_name"]
    if run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("run_name must be a single directory name")
    reservation = config["dataset"]["intervention"]
    seed = config["collection"]["seed"]
    if "question_ids_file" in reservation:
        ids = json.loads(Path(reservation["question_ids_file"]).read_text())
        if not isinstance(ids, list) or any(not isinstance(q, str) for q in ids) or len(ids) != len(set(ids)):
            raise ValueError("Intervention IDs must be a unique JSON string list")
        unknown = set(ids) - {r["question_id"] for r in questions}
        if unknown:
            raise ValueError(f"Unknown intervention IDs: {sorted(unknown)[:5]}")
        family_ids = {r["split_group_id"] for r in questions if r["question_id"] in ids}
        reserved = {r["question_id"] for r in questions if r["split_group_id"] in family_ids}
    else:
        fraction = reservation["fraction"]
        if fraction == 0:
            reserved = set()
        else:
            assignment = assign_families(questions, {"pool": 1-fraction, "intervention": fraction}, reservation.get("seed", seed))
            reserved = {q for q, p in assignment.items() if p == "intervention"}
    for row in questions:
        qid = row["question_id"]
        row["partition"] = "intervention" if qid in reserved else None
        row["original_source_split"] = row.pop("source_split")
        row["collection_seed"] = int.from_bytes(hashlib.sha256(f"{seed}:{qid}".encode()).digest()[:8], "big") % (2**31)
    if A2Presentation.CANDIDATES.value in settings["collection"]["a2_presentations"]:
        orders = assign_candidate_orders(sorted(r["question_id"] for r in questions), seed=seed,
            hop_groups={r["question_id"]: json.dumps([r["dataset_source"], r["hop_group"], r["partition"]]) for r in questions})
        for row in questions:
            row["candidate_order"] = orders[row["question_id"]]
    width, remainder = divmod(len(questions), num_chunks)
    chunks, start = [], 0
    for chunk_id in range(num_chunks):
        end = start + width + (chunk_id < remainder)
        chunks.append({"chunk_id": chunk_id, "start": start, "end": end})
        start = end
    basis = {"run_name": run_name, "inputs": master["inputs"], "questions": questions,
             "settings": settings, "chunks": chunks}
    return {**basis, "manifest_sha256": sha256_json(basis)}


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    basis = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_json(basis) != manifest.get("manifest_sha256"):
        raise ValueError("Prepared manifest changed; prepare a new run")
    return manifest


def select_chunks(manifest, chunk_ids=None):
    chunks = {c["chunk_id"]: c for c in manifest["chunks"]}
    selected = list(chunks) if chunk_ids is None else list(chunk_ids)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(chunks):
        raise ValueError("Select unique chunk IDs from the prepared manifest")
    return [chunks[i] for i in selected]
