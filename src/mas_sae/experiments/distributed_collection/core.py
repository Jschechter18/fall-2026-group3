"""Local collection, merge, and export. AWS CLI handles file transport."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

from mas_sae.data.partition_adapter import canonical_partitions, build_collected_manifest, write_run_tables
from mas_sae.data.production import export_partition_activations, write_partitions
from mas_sae.experiments import collection_progress as progress
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file, write_json_atomic
from mas_sae.experiments.collection import collect_examples
from mas_sae.experiments.collection_artifacts import save_collection_artifacts
from mas_sae.experiments.conditions import OMITTED
from mas_sae.experiments.provenance import experiment_compatibility
from mas_sae.experiments.records import read_jsonl, write_jsonl
from .dataset import assign_families, collection_settings, read_manifest, split_settings


def check_config(manifest, config):
    if "split" not in manifest:
        raise ValueError("Prepared manifest has no frozen split settings; prepare a new run")
    if split_settings(config) != manifest["split"]:
        raise ValueError("Split settings differ from the prepared manifest")
    if collection_settings(config) != manifest["settings"]:
        raise ValueError("Collection settings differ from the prepared manifest")
    if config["output"]["run_name"] != manifest["run_name"]:
        raise ValueError("Run name differs from the prepared manifest")


def chunk_directory(manifest_path, chunk):
    return Path(manifest_path).parent / "chunks" / f"{chunk['chunk_id']:06d}"


def _file_facts(root):
    return {str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"complete.json", "collection.json"}
            and not any(part.endswith(".tmp") for part in p.relative_to(root).parts)}


def _check_completed(root, manifest, chunk):
    complete = json.loads((root/"complete.json").read_text())
    if (complete.get("manifest_sha256") != manifest["manifest_sha256"]
            or complete.get("chunk_id") != chunk["chunk_id"]
            or complete.get("files") != _file_facts(root)):
        raise ValueError(f"{root}: completed chunk identity or files changed")
    return complete


def collect_manifest_chunks(manifest_path, manifest, chunks, *, config, resolved_config,
                            model, solver, critic, validator, candidate_sites, type_checked_target,
                            resume=False):
    """One model load can process several work chunks, with durable batch checkpoints."""
    check_config(manifest, config)
    experiment = experiment_compatibility(resolved_config, candidate_sites)
    for chunk in chunks:
        root = chunk_directory(manifest_path, chunk)
        questions = manifest["questions"][chunk["start"]:chunk["end"]]
        ids = [r["question_id"] for r in questions]
        identity = {"manifest_sha256": manifest["manifest_sha256"], "chunk_id": chunk["chunk_id"],
                    "experiment": experiment, "resolved_config": resolved_config}
        if (root/"complete.json").exists():
            if not resume:
                raise FileExistsError(f"{root}: already complete; use --resume to skip")
            saved = _check_completed(root, manifest, chunk)
            if saved["experiment"] != experiment:
                raise ValueError("Completed chunk experiment differs from this worker")
            continue
        if root.exists():
            if not resume:
                raise FileExistsError(f"{root}: use --resume to continue")
            saved = json.loads((root/"provenance.json").read_text())
            if any(saved.get(k) != identity[k] for k in ("manifest_sha256", "chunk_id", "experiment")):
                raise ValueError("Cannot resume: manifest or experiment changed")
            if saved["resolved_config"] != resolved_config:
                saved.setdefault("resumed_workers", []).append(resolved_config)
                write_json_atomic(root/"provenance.json", saved)
        else:
            progress.initialize(root, identity)
        position = progress.resume_position(root, ids)
        batch_size = config["output"].get("chunk_size", 250)
        for start in range(position, len(ids), batch_size):
            batch = questions[start:start+batch_size]
            result = collect_examples(
                examples=[{**r["payload"], "id": r["question_id"]} for r in batch], source_split="all",
                model=model, solver=solver, critic=critic, validator=validator,
                candidate_sites=candidate_sites, base_seed=config["collection"]["seed"],
                question_offset=chunk["start"]+start,
                question_seeds={r["question_id"]: r["collection_seed"] for r in batch},
                candidate_orders={r["question_id"]: r["candidate_order"] for r in batch if "candidate_order" in r},
                active_conditions=config["collection"].get("active_conditions", OMITTED),
                a2_presentations=config["collection"].get("a2_presentations", OMITTED),
                type_checked_target=type_checked_target)
            progress.write_chunk(root, start=start, question_ids=ids[start:start+batch_size],
                                 candidate_sites=candidate_sites, result=result)
        if progress.resume_position(root, ids) != len(ids):
            raise ValueError("Work chunk is incomplete")
        write_json_atomic(root/"complete.json", {"manifest_sha256": manifest["manifest_sha256"],
            "chunk_id": chunk["chunk_id"], "experiment": experiment, "files": _file_facts(root)}, overwrite=False)


@contextmanager
def staged_output(destination):
    """Publish activations and records together by renaming their single parent."""
    destination = Path(destination)
    ensure_output_available(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        root = Path(tmp)/"bundle"
        root.mkdir()
        yield root
        ensure_output_available(destination)
        root.rename(destination)


def merge_run(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    expected = {f"{c['chunk_id']:06d}" for c in manifest["chunks"]}
    chunks_root = manifest_path.parent/"chunks"
    actual = {p.name for p in chunks_root.iterdir() if p.is_dir()} if chunks_root.exists() else set()
    if expected != actual:
        raise ValueError(f"Chunk set mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    checkpoints, workers = [], []
    experiment = None
    by_id = {r["question_id"]: r for r in manifest["questions"]}
    for chunk in manifest["chunks"]:
        root = chunk_directory(manifest_path, chunk)
        completed = _check_completed(root, manifest, chunk)
        worker = json.loads((root/"provenance.json").read_text())
        current = experiment_compatibility(worker["resolved_config"], completed["experiment"]["activation_sites"])
        if worker.get("manifest_sha256") != manifest["manifest_sha256"] or worker.get("chunk_id") != chunk["chunk_id"]:
            raise ValueError("Worker belongs to another manifest or chunk")
        if worker["experiment"] != current or current != completed["experiment"]:
            raise ValueError("Worker experiment metadata disagrees")
        check_config(manifest, worker["resolved_config"])
        if experiment is not None and experiment != current:
            raise ValueError("Workers used different experiment settings")
        experiment = current
        workers.append(worker)
        ids = [r["question_id"] for r in manifest["questions"][chunk["start"]:chunk["end"]]]
        paths, position = progress._committed_prefix(root, ids)
        if position != len(ids):
            raise ValueError(f"{root}: incomplete checkpoint coverage")
        for path in paths:
            for name in ("records.jsonl", "exclusions.jsonl"):
                for row in read_jsonl(path/name):
                    fixed = by_id.get(row.get("question_id"))
                    if fixed is None or type(row.get("seed")) is not int or row["seed"] != fixed["collection_seed"]:
                        raise ValueError("Collected question seed differs from manifest")
                    if name == "records.jsonl" and (row.get("question") != fixed["payload"]["question"]
                            or row.get("source_split") != "all"):
                        raise ValueError("Collected question or storage differs from manifest")
                    if name == "records.jsonl" and fixed.get("candidate_order") != row.get("candidate_order"):
                        raise ValueError("Collected candidate order differs from manifest")
        checkpoints.extend(paths)
    destination = manifest_path.parent/"merged"
    ids = list(by_id)
    sites = experiment["activation_sites"]
    if not sites or len(set(sites)) != len(sites):
        raise ValueError("Worker activation sites must be nonempty and unique")
    with staged_output(destination) as stage:
        reference = None
        shapes = {}
        for site in sites:
            merged = progress.merge_checkpoint_paths(checkpoints, question_ids=ids, candidate_sites=[site])
            rows, exclusions = merged["records"], merged["exclusions"]
            if len({r["episode_id"] for r in rows}) != len(rows) or len({r["question_id"] for r in rows}) != len(rows):
                raise ValueError("Manifest collection requires exactly one unique episode per successful question")
            for row in rows:
                fixed = by_id[row["question_id"]]
                row.update({k: v for k, v in fixed.items() if k not in {"payload", "master_index", "source_index"}})
            if reference is not None and reference != (rows, exclusions):
                raise ValueError("Records changed between activation sites")
            reference = (rows, exclusions)
            summary = save_collection_artifacts(
                activation_root=stage/"activations", result_root=stage/"results", run_name="raw", source_split="all",
                candidate_sites=[site], records=rows, exclusions=exclusions, resolved_config=workers[0]["resolved_config"],
                sampled_questions=[{k: v for k, v in r.items() if k != "payload"} for r in manifest["questions"]],
                attempt1_by_site=merged["attempt1_by_site"], attempt2_by_site=merged["attempt2_by_site"])
            shapes.update(summary["site_shapes"])
            del merged
        summary["site_shapes"] = shapes
        write_json_atomic(stage/"results/raw/all/summary.json", summary)
        write_json_atomic(stage/"collection.json", {"manifest_sha256": manifest["manifest_sha256"],
            "sites": sites, "files": _file_facts(stage), "collected": [r["question_id"] for r in rows], "excluded": exclusions,
            "workers": workers})
    return destination


def split_run(manifest_path, config):
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    check_config(manifest, config)
    merged = manifest_path.parent/"merged"
    info = json.loads((merged/"collection.json").read_text())
    if info["manifest_sha256"] != manifest["manifest_sha256"] or info["files"] != _file_facts(merged):
        raise ValueError("Merged run identity or files changed")
    split = manifest["split"]
    assignment = assign_families([r for r in manifest["questions"] if r["partition"] != "intervention"],
        split["proportions"], split["seed"], candidate_strata=True)
    questions = [{**r, "partition": "intervention" if r["partition"] == "intervention"
                  else assignment[r["question_id"]]} for r in manifest["questions"]]
    raw_fixed = canonical_partitions(questions, storage_split="all")
    rows = read_jsonl(merged/"results/raw/all/interactions.jsonl")
    raw_manifest = build_collected_manifest(rows, raw_fixed, exclusions=info["excluded"])
    layers = config["collection"]["layers"]
    destination = manifest_path.parent/"prepared"
    with staged_output(destination) as stage:
        fixed = canonical_partitions(questions)
        write_partitions(stage/"partitions.csv", fixed, layout="partitioned")
        write_json_atomic(stage/"split.json", {"manifest_sha256": manifest["manifest_sha256"], "split": split})
        for suffix, partitions in (("", ["train", "validation", "test"]), ("_intervention", ["intervention"])):
            run_name = manifest["run_name"]+suffix
            group_fixed = {q: r for q, r in fixed.items() if r["partition"] in partitions}
            exclusions = [r for r in info["excluded"] if r["question_id"] in group_fixed]
            present = [p for p in partitions if any(raw_manifest[r["question_id"]]["partition"] == p for r in rows)]
            export_partition_activations(rows, raw_manifest, merged/"activations/raw", run="full", layers=layers,
                export_activation_root=stage/"activations", export_result_root=stage/"results", export_run_name=run_name,
                partitions=present)
            result_root = stage/"results"/run_name
            for partition in partitions:
                path = result_root/partition/"interactions.jsonl"
                exported = read_jsonl(path) if path.exists() else []
                for record in exported:
                    record["collection_activation_index"] = record.pop("source_activation_index")
                    record["source_split"] = partition
                write_jsonl(path, exported)
                write_jsonl(result_root/partition/"exclusions.jsonl", [r for r in exclusions if fixed[r["question_id"]]["partition"] == partition])
            write_run_tables(result_root, group_fixed, partitions, exclusions)
            reserved_ids = sorted(q for q, r in fixed.items() if r["partition"] == "intervention")
            (result_root/"intervention_question_ids.json").write_text(json.dumps(reserved_ids, indent=2)+"\n")
            if suffix:
                write_jsonl(result_root/"contexts.jsonl", [
                    {"question_id": r["question_id"], "payload": r["payload"]}
                    for r in questions if r["partition"] == "intervention"])
    return destination
