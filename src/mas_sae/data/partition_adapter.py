"""Question membership, origin metadata, and actual row addresses for mixed runs."""
from collections import Counter
from pathlib import Path

from mas_sae.data.production import MANIFEST_FIELDS, PARTITIONS, _activation_index, _check_indices_are_unique
from mas_sae.experiments.artifacts import sha256_file, write_json_atomic
from mas_sae.experiments.records import read_jsonl

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



def canonical_partitions(questions, *, storage_split=None):
    """Keep permanent membership separate from the storage address of this run."""
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
            **{k: v for k, v in row.items() if k != "payload"},
            "source_split": storage_split or partition,
            "in_qc100": qc, "in_development_data": exposure,
            "exposure_history_known": exposure is not None,
        }
    return result


def build_collected_manifest(rows, frozen, *, exclusions=(), coverage="complete"):
    """Construct addresses from actual records, never from master dataset positions."""
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
        if row["source_split"] != fixed["source_split"]:
            raise ValueError(f"{qid}: record storage differs from its manifest")
        record = dict.fromkeys(MANIFEST_FIELDS, "")
        record.update(fixed, full_index=_activation_index(row), scan_index=None,
                      episode_id=row.get("episode_id", ""))
        result[qid] = record
    missing = set(frozen) - set(result) - excluded
    if missing and coverage == "complete":
        raise ValueError(f"Missing collectible questions: {sorted(missing)[:5]}")
    return result


def write_run_tables(output_dir, frozen, partitions, exclusions):
    """Write final addresses and hashes after the exported records are normalized."""
    from mas_sae.data.production import write_manifest, write_partitions
    root = Path(output_dir)
    paths = [root/p/"interactions.jsonl" for p in partitions]
    rows = [r for path in paths for r in read_jsonl(path)]
    manifest = build_collected_manifest(rows, frozen, exclusions=exclusions)
    write_manifest(root/"split_manifest.csv", manifest)
    write_partitions(root/"partitions.csv", frozen, layout="partitioned")
    write_json_atomic(root/"run_layout.json", {"layout": "partitioned", "source_splits": list(partitions)})
    write_json_atomic(root/"collection_status.json", {"collected": list(manifest), "excluded": exclusions,
        "empty_partitions": [p for p in partitions if not any(r["partition"] == p for r in rows)]})
    write_json_atomic(root/"counts.json", {
        "source_files_sha256": {str(path.relative_to(root)): sha256_file(path) for path in paths},
        "collected_by_partition_and_source": dict(sorted(Counter(f"{r['partition']} / {r['dataset_source']}" for r in rows).items())),
        "membership_by_partition": dict(Counter(r["partition"] for r in frozen.values())),
        "collected_by_partition_hop_order": dict(sorted(Counter(
            f"{r['partition']} / {r['hop_group']} / {r.get('candidate_order', 'feedback')}" for r in rows).items())),
        "num_excluded": len(exclusions)})
    return manifest
