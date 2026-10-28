"""Distributed episode collection preparation, transport, and finalization. This module is 
primarily intended for ensuring our data can be reliably collected, transported, and finalized across distributed environments.

Public exports preserve imports from ``mas_sae.experiments.distributed_collection``.
Command-line parsing and dispatch live in :mod:`.cli`.
"""
from .core import (
    COLLECTABLE_PARTITIONS,
    MASTER_VERSION,
    PLAN_VERSION,
    REQUIRED_CHUNK_FILES,
    S3Transport,
    build_master_dataset,
    chunk_file_manifest,
    collection_report,
    finalize_local_chunks,
    freeze_partitions,
    plan_chunks,
    write_work_packages,
)

__all__ = [
    "COLLECTABLE_PARTITIONS",
    "MASTER_VERSION",
    "PLAN_VERSION",
    "REQUIRED_CHUNK_FILES",
    "S3Transport",
    "build_master_dataset",
    "chunk_file_manifest",
    "collection_report",
    "finalize_local_chunks",
    "freeze_partitions",
    "plan_chunks",
    "write_work_packages",
]
