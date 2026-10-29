"""Mixed-dataset collection with deterministic work chunks and post-collection splits."""
from .dataset import build_master_dataset, prepare_manifest, read_manifest, select_chunks
from .core import collect_manifest_chunks, merge_run, split_run

__all__ = ["build_master_dataset", "prepare_manifest", "read_manifest", "select_chunks",
           "collect_manifest_chunks", "merge_run", "split_run"]
