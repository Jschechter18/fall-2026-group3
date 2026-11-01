"""Verified access to the approved PR #153 candidate labels for Claude validation runs.

The frozen ``partitions.csv`` is the authority for membership: label files whose partitions
disagree with it, or whose checksum is not the approved PR #153 value, are refused.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from mas_sae.experiments.records import read_jsonl

PARTITIONS = ("train", "validation", "test", "intervention")
CLASSES = ("adoption", "retention", "agreement", "out")


def read_frozen_partitions(path: Path) -> dict[str, str]:
    """``{question_id: partition}`` from a frozen partitions.csv; duplicates and unknown partitions raise."""
    frozen: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qid, partition = row["question_id"], row["partition"]
            if qid in frozen:
                raise ValueError(f"{path}: question {qid} appears more than once")
            if partition not in PARTITIONS:
                raise ValueError(f"{path}: question {qid} has unknown partition {partition!r}")
            frozen[qid] = partition
    return frozen


def check_label_partitions(labels: list[dict], frozen: dict[str, str]) -> None:
    """Every label's partition must equal the frozen one (``None`` = absent from the frozen file)."""
    bad = [(r["question_id"], r["partition"], frozen.get(r["question_id"])) for r in labels
           if r["partition"] != frozen.get(r["question_id"])]
    if bad:
        raise ValueError(f"{len(bad)} label partition(s) disagree with the frozen partitions, "
                         f"e.g. {bad[:3]} (question, labels file, frozen)")


def load_verified_labels(labels_path: Path, counts_path: Path, partitions_path: Path
                         ) -> tuple[list[dict], dict[str, str], dict]:
    """PR #153 labels checked against the approved checksum and the frozen partitions.

    Returns ``(labels, frozen, approved_counts)``; raises ``SystemExit`` / ``ValueError`` otherwise.
    """
    approved = json.loads(Path(counts_path).read_text())
    sha = hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()
    if sha != approved["labels_sha256"]:
        raise SystemExit(f"{labels_path} sha256 {sha[:12]}… is not the approved PR #153 labels "
                         f"({approved['labels_sha256'][:12]}…)")
    labels = read_jsonl(labels_path)
    frozen = read_frozen_partitions(partitions_path)
    check_label_partitions(labels, frozen)
    return labels, frozen, approved


def read_run_sources(run_dir: Path) -> dict[str, dict]:
    """Candidates-run interactions by question_id (train and validation source splits)."""
    return {row["question_id"]: row for split in ("train", "validation")
            for row in read_jsonl(Path(run_dir) / split / "interactions.jsonl")}
