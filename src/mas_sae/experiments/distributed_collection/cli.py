"""Argument definitions and dispatch for the distributed collection CLI.

Kept alongside the collection implementation so argument definitions and
command dispatch have one home. Parsing help does not load model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mas_sae.experiments.artifacts import ensure_output_available, write_json_atomic


def parser():
    """Describe preparation, transfer, and merge arguments in CLI help."""
    p = argparse.ArgumentParser(
        description="Prepare frozen question chunks, exchange them through S3, and merge results.",
        epilog="Run COMMAND --help for its arguments. See docs/distributed_collection.md for the workflow.",
    )
    sub = p.add_subparsers(dest="command", required=True,
                          help="Workflow step to execute; each command runs separately.")

    q = sub.add_parser("master", help="Combine versioned question sources into a master dataset.")
    q.add_argument("--inputs", type=Path, required=True,
                   help="JSON/YAML file with an 'inputs' list of source, version, path, and optional split/hop metadata. Synthetic input is optional.")
    q.add_argument("--output", type=Path, required=True,
                   help="New master dataset JSON file; must not already exist.")

    q = sub.add_parser("freeze", help="Freeze each question's partition and collection seed.")
    q.add_argument("--master", type=Path, required=True,
                   help="Master dataset JSON produced by the master command.")
    q.add_argument("--config", type=Path, required=True,
                   help="JSON/YAML with top-level proportions and seed, plus optional musique_validation_policy; not a collection-model config.")
    q.add_argument("--existing", type=Path,
                   help="Previous frozen JSON/YAML or historical MuSiQue partition CSV; preserve its assignments.")
    q.add_argument("--exposure-manifest", type=Path,
                   help="Optional historical split_manifest.csv supplying QC/development flags when --existing is a CSV.")
    q.add_argument("--output", type=Path, required=True,
                   help="New frozen partition manifest JSON file; must not already exist.")

    q = sub.add_parser("plan", help="Create train/validation/test chunks and local worker packages.")
    q.add_argument("--master", type=Path, required=True,
                   help="Master dataset JSON containing the question payloads to package.")
    q.add_argument("--partitions", type=Path, required=True,
                   help="Frozen partition manifest JSON produced by freeze; intervention questions are not scheduled.")
    q.add_argument("--run-id", required=True,
                   help="Run identifier embedded in the plan and packages; use the same value for their S3 transfers.")
    q.add_argument("--chunk-size", type=int, required=True,
                   help="Positive maximum number of questions assigned to each chunk, not the number of activation rows.")
    q.add_argument("--compatibility", type=Path,
                   help="Optional JSON/YAML mapping of expected top-level resolved-config fields. Omit to manage experiment settings through the team's shared collection YAML.")
    q.add_argument("--output", type=Path, required=True,
                   help="New work-plan JSON file containing ordered chunk membership; must not already exist.")
    q.add_argument("--packages", type=Path, required=True,
                   help="New local directory for one question-package JSON file per chunk; must not already exist.")

    q = sub.add_parser("finalize", help="Merge local completed chunks into final activation and record files.")
    q.add_argument("--plan", type=Path, required=True,
                   help="Work-plan JSON defining expected chunks and their ordering.")
    q.add_argument("--chunks", type=Path, required=True,
                   help="Local parent directory containing <chunk-id>/metadata.json, records.jsonl, exclusions.jsonl, and activations.pt for each planned chunk.")
    q.add_argument("--sites", required=True,
                   help="Comma-separated full model-module names, in recorded order, e.g. model.layers.8,model.layers.17.")
    q.add_argument("--resolved-config", type=Path, required=True,
                   help="JSON/YAML resolved collection configuration to save with final artifacts; currently not verified against saved worker provenance.")
    q.add_argument("--activation-root", type=Path, default=Path("data/activations"),
                   help="Local parent directory for <run-name>/layer_NN/<partition>*.pt (default: data/activations).")
    q.add_argument("--result-root", type=Path, default=Path("results/collection"),
                   help="Local parent directory for <run-name>/<partition>/ records and summaries (default: results/collection).")
    q.add_argument("--run-name", required=True,
                   help="Final output directory name beneath both roots; those run directories must not already exist. Separate from the plan's run ID.")

    transfer_help = {
        "upload-work": "Upload a prepared question package to S3.",
        "download-work": "Download one question package from S3 for a worker.",
        "upload": "Upload a completed chunk as an isolated S3 attempt.",
        "accept": "Register an uploaded attempt as the accepted result for its chunk.",
        "download": "Download and checksum-verify a chunk's accepted result.",
    }
    for name, description in transfer_help.items():
        q = sub.add_parser(name, help=description)
        q.add_argument("--bucket", required=True,
                       help="Shared S3 bucket name, without s3://; credentials use the normal AWS credential chain.")
        q.add_argument("--prefix", required=True,
                       help="Project key prefix inside the bucket, e.g. experiments/collection; inputs and results are stored below it.")
        q.add_argument("--run-id", required=True,
                       help=("Required by the current parser, but accept uses the run_id inside --marker instead."
                             if name == "accept" else "Run identifier used in S3 paths; must match the prepared plan/package."))
        q.add_argument("--chunk-id", required=True,
                       help=("Required by the current parser, but accept uses the chunk_id inside --marker instead."
                             if name == "accept" else "Exact chunk identifier from the work plan, e.g. train-000000."))
        if name == "upload-work":
            q.add_argument("--package", type=Path, required=True,
                           help="Local question-package JSON file generated by plan for this run and chunk.")
        elif name == "download-work":
            q.add_argument("--output", type=Path, required=True,
                           help="New local JSON file for the downloaded question package; must not already exist.")
        elif name == "upload":
            q.add_argument("--chunk-dir", type=Path, required=True,
                           help="Completed local chunk directory containing metadata, records, exclusions, and activations.")
            q.add_argument("--marker", type=Path, required=True,
                           help="New local JSON file to receive the uploaded attempt's completion metadata; pass this file to accept.")
            q.add_argument("--attempt-id",
                           help="Identifier for this upload attempt; defaults to a generated UUID. Use a fresh ID for a new attempt.")
        elif name == "accept":
            q.add_argument("--marker", type=Path, required=True,
                           help="Local completion-marker JSON produced by upload; its embedded IDs determine which S3 chunk is accepted.")
        else:
            q.add_argument("--output", type=Path, required=True,
                           help="New local directory for the accepted chunk files, normally <chunks>/<chunk-id>; must not already exist.")
    return p


def _load(path: Path):
    with path.open() as f:
        return yaml.safe_load(f) if path.suffix in {".yaml", ".yml"} else json.load(f)


def _write_new(path: Path, value):
    ensure_output_available(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, value, overwrite=False)



def main():
    """Parse a command and dispatch to the distributed collection operations."""
    a = parser().parse_args()
    from mas_sae.experiments.distributed_collection import (
        S3Transport, build_master_dataset, freeze_partitions, plan_chunks,
        write_work_packages, finalize_local_chunks,
    )
    if a.command == "master": _write_new(a.output, build_master_dataset(_load(a.inputs)["inputs"])); return
    if a.command == "freeze":
        c = _load(a.config)
        master = _load(a.master)
        if a.existing and a.existing.suffix == ".csv":
            from mas_sae.data.partition_adapter import import_historical_partitions
            old = import_historical_partitions(master, a.existing, exposure_manifest=a.exposure_manifest)
        else:
            if a.exposure_manifest:
                raise ValueError("--exposure-manifest requires historical CSV --existing")
            existing = _load(a.existing) if a.existing else None
            old = ({r["question_id"]: r for r in existing["questions"]} if existing else None)
        _write_new(a.output, freeze_partitions(master, proportions=c["proportions"], seed=c["seed"], existing=old, musique_validation_policy=c.get("musique_validation_policy", "validation"))); return
    if a.command == "plan":
        master = _load(a.master); plan = plan_chunks(master, _load(a.partitions), run_id=a.run_id, chunk_size=a.chunk_size, compatibility=_load(a.compatibility) if a.compatibility else {})
        _write_new(a.output, plan); write_work_packages(a.packages, master, plan); return
    if a.command == "finalize":
        plan = _load(a.plan); chunk_dirs = {c["chunk_id"]: a.chunks/c["chunk_id"] for c in plan["chunks"]}
        summary = finalize_local_chunks(plan=plan, chunk_dirs=chunk_dirs,
            candidate_sites=[s.strip() for s in a.sites.split(",") if s.strip()],
            activation_root=a.activation_root, result_root=a.result_root,
            run_name=a.run_name, resolved_config=_load(a.resolved_config))
        print(json.dumps(summary, indent=2)); return
    transport = S3Transport(a.bucket, a.prefix)
    try:
        transport.check_reachable()
    except ConnectionError as error:
        raise SystemExit(str(error)) from error
    if a.command == "upload-work": print(json.dumps(transport.upload_work_package(a.run_id, a.chunk_id, a.package), indent=2)); return
    if a.command == "download-work": transport.download_work_package(a.run_id, a.chunk_id, a.output); return
    if a.command == "upload": _write_new(a.marker, transport.upload_attempt(a.run_id, a.chunk_id, a.chunk_dir, a.attempt_id)); return
    if a.command == "accept": print(json.dumps(transport.accept_attempt(_load(a.marker)), indent=2)); return
    transport.download_accepted(a.run_id, a.chunk_id, a.output)

