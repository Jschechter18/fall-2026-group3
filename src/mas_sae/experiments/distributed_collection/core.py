"""Frozen dataset, work-plan, and object transport for distributed collection.

Workers exchange immutable question packages and completed instances of the
existing :mod:`collection_progress` chunk format.  ActivationStore deliberately
remains local-only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable
import uuid
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from mas_sae.data.musique import SUPPORTED_PARTITIONS
from mas_sae.experiments.artifacts import sha256_file, sha256_json, write_json_atomic
from mas_sae.experiments import collection_progress
from mas_sae.experiments.collection_artifacts import save_collection_artifacts
from mas_sae.experiments.records import read_jsonl
from mas_sae.experiments.conditions import assign_candidate_orders
from mas_sae.agents.solver import Solver

MASTER_VERSION = 1
PLAN_VERSION = 1
REQUIRED_CHUNK_FILES = ("metadata.json", "records.jsonl", "exclusions.jsonl", "activations.pt")
COLLECTABLE_PARTITIONS = ("train", "validation", "test") # NOTE: WE MAY NEED TO THINK ABOUT THIS


def _canonical_payload(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


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
    seen_source_ids: set[tuple[str, str]] = set()
    input_records = []
    
    for spec in inputs:
        source, version = str(spec["source"]), str(spec["version"])
        if not source or ":" in source or not version:
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
            original_id = str(payload.get(id_field, ""))
            if not original_id:
                raise ValueError(f"{path}: record {source_index} has no {id_field!r}")
            key, global_id = (source, original_id), f"{source}:{original_id}"
            if key in seen_source_ids or global_id in seen_global:
                raise ValueError(f"duplicate input id {global_id!r}")
            seen_source_ids.add(key); seen_global.add(global_id)
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
            generation = payload.get("generation_provenance", spec.get("generation_provenance"))
            row = {"question_id": global_id, "original_id": original_id,
                   "dataset_source": source, "dataset_version": version,
                   "source_split": str(payload.get("source_split", spec.get("source_split", ""))),
                   "hop_group": str(hop), "source_index": source_index,
                   "payload_sha256": hashlib.sha256(_canonical_payload(payload)).hexdigest(),
                   "payload": payload}
            if generation is not None:
                row["generation_provenance"] = generation
            rows.append(row)
    rows.sort(key=lambda r: (r["dataset_source"], r["dataset_version"],
                             r["source_split"], r["original_id"]))
    for i, row in enumerate(rows): row["master_index"] = i
    basis = {"schema_version": MASTER_VERSION, "inputs": input_records, "questions": rows}
    # Local filenames are audit metadata, not dataset identity: moving identical versioned inputs to another machine must preserve the master digest.
    hash_basis = {**basis, "inputs": [{k: v for k, v in item.items() if k != "path"}
                                      for item in input_records]}
    return {**basis, "master_sha256": sha256_json(hash_basis)}


def _weighted_assignment(rows: list[dict[str, Any]], proportions: dict[str, float], seed: int) -> dict[str, str]:
    if set(proportions) != set(SUPPORTED_PARTITIONS) or abs(sum(proportions.values()) - 1) > 1e-9:
        raise ValueError(f"proportions must name {list(SUPPORTED_PARTITIONS)} and sum to 1")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[(row["dataset_source"], row["hop_group"])].append(row)
    result = {}
    for group, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda r: r["question_id"])
        random.Random(f"{seed}:{group[0]}:{group[1]}").shuffle(ordered)
        cumulative, previous = 0.0, 0
        for p_index, partition in enumerate(SUPPORTED_PARTITIONS):
            cumulative += proportions[partition]
            end = len(ordered) if p_index == len(SUPPORTED_PARTITIONS)-1 else round(len(ordered)*cumulative)
            for row in ordered[previous:end]: result[row["question_id"]] = partition
            previous = end
    return result

def freeze_partitions(master: dict[str, Any], *, proportions: dict[str, float], seed: int,
                      existing: dict[str, dict[str, Any]] | None = None,
                      musique_validation_policy: str = "validation") -> dict[str, Any]:
    """Keep old assignments and deterministically assign only new questions.

    Historical policy ``validation`` reserves every MuSiQue validation-source
    item for validation. ``stratify`` deliberately includes it in the general
    allocation; ``error`` forces callers to preprocess it explicitly.

    Parameters
    ----------
    master : dict[str, Any]
        The master dataset containing all questions and their metadata.
    proportions : dict[str, float]
        The desired proportions for each partition (train, validation, test, intervention).
    seed : int
        The random seed used for deterministic assignment of new questions.
    existing : dict[str, dict[str, Any]] | None, optional
        Previously existing assignments, by question ID, to be preserved. If None, no existing assignments are considered.
    musique_validation_policy : str, optional
        Policy for handling MuSiQue validation-source items. Can be "validation", "stratify", or "error". Defaults to "validation".

    Returns
    -------
    dict[str, Any]
        The updated master dataset with frozen and newly assigned partitions.

    Raises
    ------
    ValueError
        Raised if the musique_validation_policy is invalid.
    ValueError
        Raised if existing assignments contain unknown question IDs.
    ValueError
        Raised if a frozen question's metadata has changed.
    ValueError
        Raised if there is an inconsistency in the assignment process.
    """
    
    if musique_validation_policy not in {"validation", "stratify", "error"}:
        raise ValueError("musique_validation_policy must be validation, stratify, or error")
    existing = existing or {}
    by_id = {r["question_id"]: r for r in master["questions"]}
    unknown = set(existing) - set(by_id)
    if unknown: raise ValueError(f"existing assignments contain unknown ids: {sorted(unknown)[:3]}")
    frozen = {}
    new = []
    for row in master["questions"]:
        old = existing.get(row["question_id"])
        if old:
            for field in ("dataset_source", "dataset_version", "source_split", "hop_group", "payload_sha256"):
                if field in old and old[field] != row[field]:
                    raise ValueError(f"{row['question_id']}: frozen {field} changed")
            frozen[row["question_id"]] = {**old, **{k: row[k] for k in
                ("question_id", "dataset_source", "dataset_version", "source_split", "hop_group", "payload_sha256")}}
            frozen[row["question_id"]].setdefault(
                "collection_seed", int.from_bytes(hashlib.sha256(
                    f"{seed}:{row['question_id']}".encode()).digest()[:8], "big") % (2**31))
            frozen[row["question_id"]].setdefault("prior_development_exposure", False)
        else: new.append(row)
    forced, allocatable = [], []
    for row in new:
        is_mv = row["dataset_source"] == "musique" and row["source_split"] == "validation"
        if is_mv and musique_validation_policy == "error":
            raise ValueError("MuSiQue validation-source rows require an explicit policy")
        (forced if is_mv and musique_validation_policy == "validation" else allocatable).append(row)
    assigned = _weighted_assignment(allocatable, proportions, seed) if allocatable else {}
    for row in new:
        partition = "validation" if row in forced else assigned[row["question_id"]]
        prior = row["payload"].get("prior_development_exposure", False)
        frozen[row["question_id"]] = {k: row[k] for k in
            ("question_id", "dataset_source", "dataset_version", "source_split", "hop_group", "payload_sha256")}
        frozen[row["question_id"]].update(partition=partition,
            collection_seed=int.from_bytes(hashlib.sha256(f"{seed}:{row['question_id']}".encode()).digest()[:8], "big") % (2**31),
            prior_development_exposure=prior)
    ordered = [frozen[r["question_id"]] for r in master["questions"]]
    # Freeze presentation order once, independently of worker/chunk boundaries.
    # On extension, preserve existing orders and assign only missing ones.
    missing_orders = [r for r in ordered if "candidate_order" not in r]
    candidate_orders = assign_candidate_orders(
        [r["question_id"] for r in missing_orders], seed=seed,
        hop_groups={r["question_id"]: f"{r['partition']}:{r['hop_group']}" for r in missing_orders},
    )
    for row in ordered:
        if "candidate_order" not in row:
            row["candidate_order"] = candidate_orders[row["question_id"]]
        if row["candidate_order"] not in Solver.CANDIDATE_ORDERS:
            raise ValueError(f"{row['question_id']}: invalid frozen candidate_order")
    basis = {"schema_version": 1, "master_sha256": master["master_sha256"], "seed": seed,
             "proportions": proportions, "musique_validation_policy": musique_validation_policy,
             "questions": ordered}
    return {**basis, "partitions_sha256": sha256_json(basis)}


def plan_chunks(master: dict[str, Any], frozen: dict[str, Any], *, run_id: str,
                chunk_size: int, compatibility: dict[str, Any] | None = None) -> dict[str, Any]:
    """Plan chunks for distributed processing based on the frozen partitions and master dataset. Done in manifest order; intervention is never planned

    Args:
        master (dict[str, Any]): The master dataset containing all questions.
        frozen (dict[str, Any]): The frozen partitions of the dataset.
        run_id (str): The unique identifier for this run.
        chunk_size (int): The desired size of each chunk.
        compatibility (dict[str, Any]): Compatibility constraints for the chunks.

    Returns:
        dict[str, Any]: A manifest describing the planned chunks.
    """
    if chunk_size < 1: raise ValueError("chunk_size must be positive")
    if frozen["master_sha256"] != master["master_sha256"]:
        raise ValueError("partitions belong to a different master dataset")
    assignments = {r["question_id"]: r for r in frozen["questions"]}
    if len(assignments) != len(frozen["questions"]) or set(assignments) != {r["question_id"] for r in master["questions"]}:
        raise ValueError("partition question membership differs from master")
    for row in master["questions"]:
        if assignments[row["question_id"]]["payload_sha256"] != row["payload_sha256"]:
            raise ValueError(f"{row['question_id']}: partition payload differs from master")
    chunks = []
    for partition in COLLECTABLE_PARTITIONS:
        # Plan chunks for the current partition (train, val, test)
        members = [r for r in master["questions"] if assignments[r["question_id"]]["partition"] == partition]
        for start in range(0, len(members), chunk_size):
            group = members[start:start+chunk_size]
            chunk_id = f"{partition}-{start//chunk_size:06d}"
            membership = [{"question_id": r["question_id"], "collection_seed": assignments[r["question_id"]]["collection_seed"],
                           "payload_sha256": r["payload_sha256"]} for r in group]
            for member in membership:
                order = assignments[member["question_id"]].get("candidate_order")
                if order is not None:
                    member["candidate_order"] = order
            chunks.append({"chunk_id": chunk_id, "partition": partition, "partition_start": start,
                           "question_ids": [r["question_id"] for r in group],
                           "membership_sha256": sha256_json(membership), "questions": membership})
    basis = {"schema_version": PLAN_VERSION, "run_id": run_id,
             "master_sha256": master["master_sha256"], "partitions_sha256": frozen["partitions_sha256"],
             "compatibility": compatibility or {}, "chunks": chunks}
    return {**basis, "plan_sha256": sha256_json(basis)}


def write_work_packages(root: str | Path, master: dict[str, Any], plan: dict[str, Any]) -> None:
    """Write work packages for distributed collection based on the master dataset and plan.

    Parameters
    ----------
    root : str | Path
        The root directory where the work packages will be written.
    master : dict[str, Any]
        The master dataset containing all questions.
    plan : dict[str, Any]
        The plan describing how the dataset should be chunked.

    Raises
    ------
    FileExistsError
        Raised if the root directory already exists.
    """
    root = Path(root)
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    by_id = {r["question_id"]: r for r in master["questions"]}
    for chunk in plan["chunks"]:
        package = {"schema_version": 1, "run_id": plan["run_id"], "plan_sha256": plan["plan_sha256"],
                   "compatibility": plan["compatibility"], "chunk": chunk,
                   "examples": [{**by_id[qid]["payload"], "id": qid,
                                  "original_id": by_id[qid]["original_id"],
                                  "dataset_source": by_id[qid]["dataset_source"],
                                  "hop_group": by_id[qid]["hop_group"]} for qid in chunk["question_ids"]]}
        write_json_atomic(root / f"{chunk['chunk_id']}.json", package)


def chunk_file_manifest(chunk_dir: str | Path) -> dict[str, Any]:
    root = Path(chunk_dir)
    missing = [name for name in REQUIRED_CHUNK_FILES if not (root/name).is_file()]
    if missing: raise ValueError(f"chunk is missing required files: {missing}")
    files = {name: {"sha256": sha256_file(root/name), "size": (root/name).stat().st_size}
             for name in REQUIRED_CHUNK_FILES}
    return {"files": files, "package_sha256": sha256_json(files)}


@dataclass
class S3Transport:
    """Transport class for handling S3 interactions for distributed collection work packages."""
    bucket: str
    prefix: str
    client: Any = None

    def __post_init__(self):
        if not self.bucket: raise ValueError("bucket is required")
        self.prefix = self.prefix.strip("/")
        if self.client is None:
            import boto3  # optional: normal AWS credential chain
            self.client = boto3.client("s3")

    def _key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.prefix, *parts) if p.strip("/"))

    def check_reachable(self, *, timeout: float = 5.0) -> None:
        """Check HTTPS reachability without testing credentials or writing objects.

        Use the SDK's configured endpoint, including its region or custom URL.
        HTTP errors such as 403 still prove that the endpoint responded.
        """
        endpoint = self.client.meta.endpoint_url.rstrip("/")
        url = f"{endpoint}/{quote(self.bucket, safe='')}"
        try:
            with urlopen(Request(url, method="HEAD"), timeout=timeout):
                pass
        except HTTPError as error:
            error.close()
        except OSError as error:
            raise ConnectionError(
                f"S3 endpoint is unreachable at {url}: {error}. "
                "Check DNS/network connectivity and retry; local results are unchanged."
            ) from error

    def upload_work_package(self, run_id: str, chunk_id: str, path: str | Path) -> dict[str, Any]:
        """Upload a work package to the remote S3 storage.

        Parameters
        ----------
        run_id : str
            The unique identifier for the current run.
        chunk_id : str
            The unique identifier for the chunk being uploaded.
        path : str | Path
            The local path to the work package file to be uploaded.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the S3 key, SHA-256 digest, and size of the uploaded work package.

        Raises
        ------
        ValueError
            Raised if the work package identity does not match the destination.
        RuntimeError
            Raised if there is a mismatch between the expected and actual upload size.
        """
        path = Path(path); body = path.read_bytes()
        package = json.loads(body)
        if package.get("run_id") != run_id or package.get("chunk", {}).get("chunk_id") != chunk_id:
            raise ValueError("work package identity does not match destination")
        key = self._key("inputs", run_id, f"{chunk_id}.json")
        digest = hashlib.sha256(body).hexdigest()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body,
                               ContentType="application/json", Metadata={"sha256": digest})
        head = self.client.head_object(Bucket=self.bucket, Key=key)
        if int(head["ContentLength"]) != len(body): raise RuntimeError("work-package upload size mismatch")
        return {"key": key, "sha256": digest, "size": len(body)}

    def download_work_package(self, run_id: str, chunk_id: str, destination: str | Path) -> Path:
        """Download a work package from the remote storage to a local destination.

        Parameters
        ----------
        run_id : str
            The unique identifier for the current run.
        chunk_id : str
            The unique identifier for the chunk being downloaded.
        destination : str | Path
            The local path where the downloaded work package should be saved.

        Returns
        -------
        Path
            The path to the downloaded work package file.

        Raises
        ------
        FileExistsError
            Raised if the destination file already exists.
        RuntimeError
            Raised if there is a mismatch between the expected and actual work package identity or checksum.
        """
        destination = Path(destination)
        if destination.exists(): raise FileExistsError(destination)
        key = self._key("inputs", run_id, f"{chunk_id}.json")
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read()
        expected = (response.get("Metadata") or {}).get("sha256")
        if expected and hashlib.sha256(body).hexdigest() != expected:
            raise RuntimeError("work-package checksum mismatch")
        package = json.loads(body)
        if package.get("run_id") != run_id or package.get("chunk", {}).get("chunk_id") != chunk_id:
            raise RuntimeError("downloaded work-package identity mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        return destination

    def upload_attempt(self, run_id: str, chunk_id: str, chunk_dir: str | Path,
                       attempt_id: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
        """Upload an attempt for a specific chunk to the remote storage.

        Parameters
        ----------
        run_id : str
            The unique identifier for the current run.
        chunk_id : str
            The unique identifier for the chunk being uploaded.
        chunk_dir : str | Path
            The local directory containing the chunk files to be uploaded.
        attempt_id : str | None, optional
            The unique identifier for the attempt. If not provided, a new UUID will be generated, by default None.
        hardware : dict[str, Any] | None, optional
            Information about the hardware used for the attempt, by default None.

        Returns
        -------
        dict[str, Any]
            Metadata about the uploaded attempt, including run_id, chunk_id, attempt_id, and hardware information.

        Raises
        ------
        RuntimeError
            Raised if there is a mismatch between the expected and actual uploaded file sizes.
        """

        attempt_id = attempt_id or str(uuid.uuid4())
        manifest = chunk_file_manifest(chunk_dir)
        base = self._key("results", run_id, chunk_id, "attempts", attempt_id)
        for name, expected in manifest["files"].items():
            self.client.upload_file(str(Path(chunk_dir)/name), self.bucket, f"{base}/{name}")
            head = self.client.head_object(Bucket=self.bucket, Key=f"{base}/{name}")
            if int(head["ContentLength"]) != expected["size"]:
                raise RuntimeError(f"uploaded size mismatch for {name}")
        marker = {**manifest, "run_id": run_id, "chunk_id": chunk_id, "attempt_id": attempt_id,
                  "hardware": hardware or {}}
        self.client.put_object(Bucket=self.bucket, Key=f"{base}/complete.json",
                               Body=_canonical_payload(marker), ContentType="application/json")
        return marker

    def accept_attempt(self, marker: dict[str, Any]) -> dict[str, Any]:
        """Accept once using an S3 conditional create; identical retries are idempotent."""
        key = self._key("results", marker["run_id"], marker["chunk_id"], "accepted.json")
        accepted = {k: marker[k] for k in ("run_id", "chunk_id", "attempt_id", "package_sha256")}
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=_canonical_payload(accepted),
                                   ContentType="application/json", IfNoneMatch="*")
            return accepted
        except Exception:
            try:
                current = json.loads(self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read())
            except Exception: raise
            if current["package_sha256"] != accepted["package_sha256"]:
                raise RuntimeError(f"conflicting completed attempts for chunk {marker['chunk_id']}")
            return current

    def download_accepted(self, run_id: str, chunk_id: str, destination: str | Path) -> Path:
        """Download the accepted attempt for a specific chunk to a local destination.

        Parameters
        ----------
        run_id : str
            The unique identifier for the current run.
        chunk_id : str
            The unique identifier for the chunk to be downloaded.
        destination : str | Path
            The local directory where the accepted chunk files will be downloaded.

        Returns
        -------
        Path
            The local path to the downloaded chunk directory.

        Raises
        ------
        RuntimeError
            Raised if the downloaded chunk's package hash does not match the accepted package hash.
        """
        accepted_key = self._key("results", run_id, chunk_id, "accepted.json")
        accepted = json.loads(self.client.get_object(Bucket=self.bucket, Key=accepted_key)["Body"].read())
        base = self._key("results", run_id, chunk_id, "attempts", accepted["attempt_id"])
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for name in REQUIRED_CHUNK_FILES:
            self.client.download_file(self.bucket, f"{base}/{name}", str(destination/name))
        actual = chunk_file_manifest(destination)
        if actual["package_sha256"] != accepted["package_sha256"]:
            raise RuntimeError(f"corrupt accepted package for {chunk_id}")
        write_json_atomic(destination/"download_receipt.json", accepted)
        return destination


def collection_report(frozen: dict[str, Any], records: Iterable[dict[str, Any]],
                      exclusions: Iterable[dict[str, Any]], eligible_ids: Iterable[str] = ()) -> dict[str, Any]:
    """Collection success and consumer eligibility are intentionally separate."""
    lookup = {r["question_id"]: r for r in frozen["questions"]}
    successful = {str(r["question_id"]) for r in records}
    excluded = {str(r["question_id"]) for r in exclusions}
    eligible = set(map(str, eligible_ids))
    def counts(ids):
        return dict(sorted(Counter(f"{lookup[q]['partition']} / {lookup[q]['dataset_source']}" for q in ids).items()))
    return {"assigned": counts(lookup), "collected": counts(successful),
            "excluded": counts(excluded), "eligible": counts(eligible)}

def finalize_local_chunks(*, plan: dict[str, Any], chunk_dirs: dict[str, str | Path],
                          candidate_sites: list[str], activation_root: str | Path,
                          result_root: str | Path, run_name: str,
                          resolved_config: dict[str, Any]) -> dict[str, Any]:
    """Validate and merge accepted chunks in frozen order, partition by partition.

    The existing merger reconstructs all four indices from saved successful row
    counts. The existing artifact writer then creates ``A1``, ``A2`` and
    ``cat(A1, A2)`` separately for every site; combined per-chunk SAE tensors
    are never consumed.

    Parameters
    ----------
    plan : dict[str, Any]
        Mapping from chunk IDs to their corresponding local directories containing chunk files.
    chunk_dirs : dict[str, str  |  Path]
        Mapping from chunk IDs to their corresponding local directories containing chunk files.
    candidate_sites : list[str]
        List of candidate sites for the collection process.
    activation_root : str | Path
        Root directory for activation files.
    result_root : str | Path
        Root directory for result files.
    run_name : str
        Name of the current run.
    resolved_config : dict[str, Any]
        Resolved configuration for the collection process.

    Returns
    -------
    dict[str, Any]
        Summary of the finalized local chunks, typically including status and any relevant metadata.

    Raises
    ------
    ValueError
        Raised if candidate_sites is empty.
    ValueError
        Raised if there is a mismatch between the expected and provided chunk directories.
    FileExistsError
        Raised if the final run output already exists.
    ValueError
        Raised if there is an issue with the chunk validation or merging process.
    ValueError
        Raised if there is an issue with the resolved configuration.
    RuntimeError
        Raised if there is a corruption detected in the accepted chunks.
    """
    
    if not candidate_sites:
        raise ValueError("candidate_sites must not be empty.")
    expected = {c["chunk_id"]: c for c in plan["chunks"]}
    if set(chunk_dirs) != set(expected):
        missing, extra = set(expected)-set(chunk_dirs), set(chunk_dirs)-set(expected)
        raise ValueError(f"chunk set mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    activation_root, result_root = Path(activation_root), Path(result_root)
    final_a, final_r = activation_root/run_name, result_root/run_name
    if final_a.exists() or final_r.exists(): raise FileExistsError("final run output already exists")
    stage_name = f".{run_name}.staging-{uuid.uuid4()}"
    summaries = {}
    worker_provenance = []
    try:
        with tempfile.TemporaryDirectory() as progress_parent:
            for partition in COLLECTABLE_PARTITIONS:
                chunks = [c for c in plan["chunks"] if c["partition"] == partition]
                if not chunks: continue
                progress = Path(progress_parent)/partition; progress.mkdir()
                question_ids = []
                for chunk in chunks:
                    source = Path(chunk_dirs[chunk["chunk_id"]])
                    meta = json.loads((source/"metadata.json").read_text())
                    required = {"chunk_id": chunk["chunk_id"], "run_id": plan["run_id"],
                                "plan_sha256": plan["plan_sha256"],
                                "membership_sha256": chunk["membership_sha256"],
                                "question_ids": chunk["question_ids"], "start": chunk["partition_start"]}
                    if any(meta.get(k) != v for k, v in required.items()):
                        raise ValueError(f"{chunk['chunk_id']}: metadata is incompatible with work plan")
                    if meta.get("candidate_sites") != candidate_sites:
                        raise ValueError(f"{chunk['chunk_id']}: activation sites differ")
                    actual = chunk_file_manifest(source)
                    receipt = source/"download_receipt.json"
                    if receipt.is_file():
                        downloaded = json.loads(receipt.read_text())
                        if (downloaded.get("run_id") != plan["run_id"]
                                or downloaded.get("chunk_id") != chunk["chunk_id"]
                                or downloaded.get("package_sha256") != actual["package_sha256"]):
                            raise ValueError(f"{chunk['chunk_id']}: downloaded files changed")
                    seeds = {r["question_id"]: r["collection_seed"] for r in chunk["questions"]}
                    orders = {r["question_id"]: r.get("candidate_order") for r in chunk["questions"]}
                    for name in ("records.jsonl", "exclusions.jsonl"):
                        for row in read_jsonl(source/name):
                            qid = row.get("question_id")
                            if qid not in seeds or type(row.get("seed")) is not int or row["seed"] != seeds[qid]:
                                raise ValueError(f"{chunk['chunk_id']}: question seed differs from plan")
                            if row.get("a2_presentation") == "candidates":
                                if orders[qid] not in Solver.CANDIDATE_ORDERS or row.get("candidate_order") != orders[qid]:
                                    raise ValueError(f"{chunk['chunk_id']}: candidate order differs from plan")
                    worker_provenance.append({"chunk_id": chunk["chunk_id"],
                        "progress": meta.get("provenance"), "resolved_config": meta.get("resolved_config")})
                    shutil.copytree(source, progress/f"chunk_{chunk['partition_start']:08d}")
                    question_ids.extend(chunk["question_ids"])
                # Merge/write one captured site at a time. Chunk files contain
                # all sites, but tensors from earlier sites are released before
                # the next site is materialized, bounding peak tensor memory to
                # roughly one partition-layer plus one chunk container.
                reference_records = reference_exclusions = None
                summary: dict[str, Any] = {}
                site_shapes = {}
                for site in candidate_sites:
                    merged = collection_progress.merge_chunks(
                        progress, question_ids=question_ids, candidate_sites=[site])
                    if reference_records is None:
                        reference_records, reference_exclusions = merged["records"], merged["exclusions"]
                    elif (merged["records"] != reference_records
                          or merged["exclusions"] != reference_exclusions):
                        raise RuntimeError("record metadata changed while streaming activation sites")
                    current = save_collection_artifacts(
                        activation_root=activation_root, result_root=result_root,
                        run_name=stage_name, source_split=partition, candidate_sites=[site],
                        attempt1_by_site=merged["attempt1_by_site"], attempt2_by_site=merged["attempt2_by_site"],
                        records=merged["records"], exclusions=merged["exclusions"],
                        resolved_config=resolved_config,
                        sampled_questions=[{"question_id": q, "hop_group": "explicit"} for q in question_ids])
                    site_shapes.update(current["site_shapes"])
                    summary = current
                    del merged
                summary["site_shapes"] = site_shapes
                write_json_atomic(result_root/stage_name/partition/"summary.json", summary)
                summaries[partition] = summary
        staged_a, staged_r = activation_root/stage_name, result_root/stage_name
        # Rename only after every partition has validated and been written.
        if staged_a.exists(): staged_a.replace(final_a)
        staged_r.replace(final_r)
        write_json_atomic(final_r/"worker_provenance.json", {"workers": worker_provenance})
        write_json_atomic(final_r/"distributed_summary.json", {"plan_sha256": plan["plan_sha256"], "partitions": summaries})
        return summaries
    except Exception:
        for path in (activation_root/stage_name, result_root/stage_name):
            if path.exists(): shutil.rmtree(path)
        raise
