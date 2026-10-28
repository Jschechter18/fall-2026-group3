"""Prepare frozen distributed collection inputs and move verified chunks via S3.

Examples use placeholders; see docs/distributed_collection.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mas_sae.experiments.distributed_collection import (
    S3Transport, build_master_dataset, freeze_partitions, plan_chunks,
    write_work_packages, finalize_local_chunks,
)


def _load(path: Path):
    with path.open() as f:
        return yaml.safe_load(f) if path.suffix in {".yaml", ".yml"} else json.load(f)


def _write_new(path: Path, value):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("master"); q.add_argument("--inputs", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("freeze"); q.add_argument("--master", type=Path, required=True); q.add_argument("--config", type=Path, required=True); q.add_argument("--existing", type=Path); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("plan"); q.add_argument("--master", type=Path, required=True); q.add_argument("--partitions", type=Path, required=True); q.add_argument("--run-id", required=True); q.add_argument("--chunk-size", type=int, required=True); q.add_argument("--compatibility", type=Path, required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--packages", type=Path, required=True)
    q = sub.add_parser("finalize"); q.add_argument("--plan", type=Path, required=True); q.add_argument("--chunks", type=Path, required=True); q.add_argument("--sites", required=True); q.add_argument("--resolved-config", type=Path, required=True); q.add_argument("--activation-root", type=Path, default=Path("data/activations")); q.add_argument("--result-root", type=Path, default=Path("results/collection")); q.add_argument("--run-name", required=True)
    for name in ("upload-work", "download-work", "upload", "accept", "download"):
        q = sub.add_parser(name); q.add_argument("--bucket", required=True); q.add_argument("--prefix", required=True); q.add_argument("--run-id", required=True); q.add_argument("--chunk-id", required=True)
        if name == "upload-work": q.add_argument("--package", type=Path, required=True)
        elif name == "download-work": q.add_argument("--output", type=Path, required=True)
        elif name == "upload": q.add_argument("--chunk-dir", type=Path, required=True); q.add_argument("--marker", type=Path, required=True); q.add_argument("--attempt-id")
        elif name == "accept": q.add_argument("--marker", type=Path, required=True)
        else: q.add_argument("--output", type=Path, required=True)
    return p


def main():
    a = parser().parse_args()
    if a.command == "master": _write_new(a.output, build_master_dataset(_load(a.inputs)["inputs"])); return
    if a.command == "freeze":
        c = _load(a.config); existing = _load(a.existing) if a.existing else None
        old = ({r["question_id"]: r for r in existing["questions"]} if existing else None)
        _write_new(a.output, freeze_partitions(_load(a.master), proportions=c["proportions"], seed=c["seed"], existing=old, musique_validation_policy=c.get("musique_validation_policy", "validation"))); return
    if a.command == "plan":
        master = _load(a.master); plan = plan_chunks(master, _load(a.partitions), run_id=a.run_id, chunk_size=a.chunk_size, compatibility=_load(a.compatibility))
        _write_new(a.output, plan); write_work_packages(a.packages, master, plan); return
    if a.command == "finalize":
        plan = _load(a.plan); chunk_dirs = {c["chunk_id"]: a.chunks/c["chunk_id"] for c in plan["chunks"]}
        summary = finalize_local_chunks(plan=plan, chunk_dirs=chunk_dirs,
            candidate_sites=[s.strip() for s in a.sites.split(",") if s.strip()],
            activation_root=a.activation_root, result_root=a.result_root,
            run_name=a.run_name, resolved_config=_load(a.resolved_config))
        print(json.dumps(summary, indent=2)); return
    transport = S3Transport(a.bucket, a.prefix)
    if a.command == "upload-work": print(json.dumps(transport.upload_work_package(a.run_id, a.chunk_id, a.package), indent=2)); return
    if a.command == "download-work": transport.download_work_package(a.run_id, a.chunk_id, a.output); return
    if a.command == "upload": _write_new(a.marker, transport.upload_attempt(a.run_id, a.chunk_id, a.chunk_dir, a.attempt_id)); return
    if a.command == "accept": print(json.dumps(transport.accept_attempt(_load(a.marker)), indent=2)); return
    transport.download_accepted(a.run_id, a.chunk_id, a.output)


if __name__ == "__main__": main()
