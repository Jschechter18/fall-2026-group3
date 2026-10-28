"""Adapt historical membership and distributed runs to production manifests.

No labels are generated here. Membership is question-level; row addresses and
optional episode IDs describe this collection, not permanent dataset identity.
"""
from __future__ import annotations

import csv
from pathlib import Path

from mas_sae.data.production import (
    MANIFEST_FIELDS, PARTITIONS, _activation_index, _check_indices_are_unique,
    read_partitions,
)

SOURCE_FIELDS = ("dataset_source", "original_id", "original_source_split")
EXPOSURE_FIELDS = ("in_qc100", "in_development_data")


def optional_bool(value):
    """Unknown exposure history remains unknown rather than becoming false."""
    if value is None or value == "":
        return None
    if type(value) is bool:
        return value
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Invalid exposure flag {value!r}")


def import_historical_partitions(master, path, *, exposure_manifest=None):
    """Map validated historical MuSiQue CSV IDs to master IDs without recutting.

    A full historical split_manifest.csv can supply exposure flags directly;
    partitions.csv alone does not establish whether questions were inspected.
    """
    fixed = read_partitions(path)
    lookup = {(r["dataset_source"], r["original_id"]): r for r in master["questions"]}
    if len(lookup) != len(master["questions"]):
        raise ValueError("Master has ambiguous source/original-ID mappings")
    with Path(path).open(newline="") as file:
        history = {r["question_id"]: r for r in csv.DictReader(file)}
    if exposure_manifest is not None:
        with Path(exposure_manifest).open(newline="") as file:
            exposure_rows = list(csv.DictReader(file))
        seen = set()
        for row in exposure_rows:
            qid = row["question_id"]
            if qid in seen or qid not in fixed:
                raise ValueError(f"Exposure manifest has duplicate or unknown historical ID {qid}")
            seen.add(qid)
            history[qid].update({k: row[k] for k in EXPOSURE_FIELDS if k in row})
    imported = {}
    for original_id, assignment in fixed.items():
        question = lookup.get(("musique", original_id))
        if question is None:
            raise ValueError(f"Historical MuSiQue question {original_id} is missing from master")
        if (question["source_split"] != assignment["source_split"]
                or question["hop_group"] != assignment["hop_group"]):
            raise ValueError(f"Historical origin/hop metadata differs for {original_id}")
        flags = {k: optional_bool(history[original_id].get(k)) for k in EXPOSURE_FIELDS}
        # Being in QC proves exposure even if the broader flag was omitted.
        if flags["in_qc100"] is True:
            flags["in_development_data"] = True
        imported[question["question_id"]] = {
            **{k: question[k] for k in ("question_id", "original_id", "dataset_source",
                                        "dataset_version", "source_split", "hop_group", "payload_sha256")},
            "partition": assignment["partition"], **flags,
            "prior_development_exposure": flags["in_development_data"],
            "exposure_history_known": flags["in_development_data"] is not None,
        }
        for field in ("candidate_order", "collection_seed"):
            value = history[original_id].get(field)
            if value not in (None, ""):
                imported[question["question_id"]][field] = int(value) if field == "collection_seed" else value
    return imported


def canonical_partitions(questions):
    """Represent frozen membership using storage split plus explicit origin."""
    result = {}
    for row in questions:
        qid, partition = row["question_id"], row["partition"]
        if qid in result or partition not in PARTITIONS:
            raise ValueError(f"Duplicate ID or unknown partition for {qid}")
        exposure = optional_bool(row.get("in_development_data", row.get("prior_development_exposure")))
        qc = optional_bool(row.get("in_qc100"))
        if qc is True:
            exposure = True
        result[qid] = {
            **row, "source_split": partition,
            "original_source_split": row.get("original_source_split", row["source_split"]),
            "in_qc100": qc, "in_development_data": exposure,
            "exposure_history_known": exposure is not None,
        }
    return result


def build_collected_manifest(rows, frozen, *, exclusions=(), coverage="complete"):
    """Reuse production index semantics for one-condition, one-presentation runs."""
    if coverage not in {"complete", "subset"}:
        raise ValueError("coverage must be complete or subset")
    _check_indices_are_unique(rows, "full")
    excluded = {r["question_id"] for r in exclusions}
    if len(excluded) != len(exclusions) or not excluded <= set(frozen):
        raise ValueError("Duplicate or unknown excluded question")
    result = {}
    for row in rows:
        qid = row["question_id"]
        fixed = frozen.get(qid)
        if fixed is None or qid in result or qid in excluded:
            raise ValueError(f"Unknown, repeated, or excluded collected question {qid}")
        if fixed["partition"] == "intervention":
            raise ValueError("Intervention questions cannot enter ordinary collection")
        if row["source_split"] != fixed["partition"]:
            raise ValueError(f"{qid}: collection folder differs from frozen partition")
        record = dict.fromkeys(MANIFEST_FIELDS, "")
        record.update(question_id=qid, source_split=row["source_split"], partition=fixed["partition"],
                      hop_group=fixed["hop_group"], in_qc100=fixed.get("in_qc100"),
                      in_development_data=fixed.get("in_development_data"),
                      full_index=_activation_index(row), scan_index=None)
        record.update({k: v for k, v in fixed.items() if k not in record})
        record["episode_id"] = row.get("episode_id", "")
        result[qid] = record
    reserved = {q for q, r in frozen.items() if r["partition"] == "intervention"}
    if excluded & reserved:
        raise ValueError("Reserved intervention questions cannot be collection exclusions")
    missing = set(frozen) - set(result) - excluded - reserved
    if missing and coverage == "complete":
        raise ValueError(f"Missing collectible questions: {sorted(missing)[:5]}")
    return result


def write_distributed_tables(output_dir, questions, partitions):
    """Write membership, actual row addresses, and collection coverage without labels."""
    import json
    from collections import Counter
    from mas_sae.data.production import write_manifest, write_partitions
    from mas_sae.experiments.records import read_jsonl
    from mas_sae.experiments.artifacts import sha256_file, write_json_atomic

    root = Path(output_dir)
    fixed = canonical_partitions(questions)
    paths = [root/p/"interactions.jsonl" for p in partitions]
    rows = [r for path in paths for r in read_jsonl(path)]
    exclusions = [r for p in partitions for r in read_jsonl(root/p/"exclusions.jsonl")]
    manifest = build_collected_manifest(rows, fixed, exclusions=exclusions)
    write_manifest(root/"split_manifest.csv", manifest)
    write_partitions(root/"partitions.csv", fixed, layout="partitioned")
    reserved = [q for q, r in fixed.items() if r["partition"] == "intervention"]
    write_json_atomic(root/"run_layout.json", {"layout": "partitioned", "source_splits": list(partitions)})
    write_json_atomic(root/"collection_status.json", {
        "collected": list(manifest), "excluded": exclusions, "reserved": reserved,
        "exposure_history_unknown": [q for q, r in fixed.items() if not r["exposure_history_known"]],
    })
    (root/"intervention_question_ids.json").write_text(json.dumps(reserved, indent=2)+"\n")
    counts = Counter(f"{r['partition']} / {r['dataset_source']}" for r in manifest.values())
    write_json_atomic(root/"counts.json", {
        "source_files_sha256": {str(path.relative_to(root)): sha256_file(path) for path in paths},
        "collected_by_partition_and_source": dict(sorted(counts.items())),
        "num_excluded": len(exclusions), "num_reserved": len(reserved),
    })
