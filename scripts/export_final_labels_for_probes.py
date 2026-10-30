"""Write the approved PR #153 labels as a behavior package the probe loader reads (issue #155, for #161).

Same package as ``prepare_behavior_v01.py`` (``labels.csv``, ``split_manifest.csv``, ``partitions.csv``,
``counts.json``) built with the same production functions, so ``layer_selection`` and
``behavior_data.align_episodes`` read it unchanged. No label is recomputed: every collected episode takes
its PR #153 label from ``final_labels.jsonl``, whose checksum must equal ``labels_sha256`` in the
``counts.json`` of the same build (the approved build, or ``build_final_labels_partitioned.py``), and
whose partition must equal the manifest's.

labels.csv: question_id, source_split, partition, lexical_v2_label, then ``primary_target`` (1 adoption,
0 retention, blank otherwise), ``eligible_primary`` (adoption or retention), ``behavior_label`` (all four
classes), ``label_reason``, ``label_step``, ``previous_label``, ``flags`` and ``policy_version``. PR #153
defines none of the v0.x flag columns ``behavior_labels.read_labels`` requires, so causal replay keeps its
own package.

A collected episode whose question has no frozen partition (two in the candidates run) cannot enter the
manifest. It is written to ``excluded_episodes.json`` with its reason, for the loader's ``excluded_episodes``.

Usage, the candidates run (its 202 collection failures are frozen questions without an episode):
    PYTHONPATH=src python scripts/export_final_labels_for_probes.py \\
        --labels <build>/final_labels.jsonl \\
        --counts results/experiments/candidate_final_labels_20261027/counts.json \\
        --full-results results/collection/natural_4b_full9_candidates \\
        --frozen-partitions results/behavior/behavior_v012_official_20261011/partitions.csv \\
        --coverage subset --output <new folder>

A fresh partitioned run: ``--layout partitioned --full-results <prepared>/results/<run>``, with the
labels and counts from ``build_final_labels_partitioned.py``; the run's own partitions.csv is used.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mas_sae.data.production import (
    COVERAGE_MODES,
    build_partition_manifest,
    interaction_paths,
    read_partitions,
    read_run_interactions,
    write_package_tables,
)
from mas_sae.evaluation.behavior_labels import write_labels
from mas_sae.experiments.artifacts import (
    ensure_output_available,
    get_git_commit,
    get_git_diff_sha256,
    sha256_file,
    write_json_atomic,
)
from mas_sae.experiments.records import read_jsonl

TARGETS = {"adoption": 1, "retention": 0}
UNPARTITIONED = "collected, but the question has no frozen partition (PR #153 label: out)"


def load_build(labels_path: Path, counts_path: Path) -> tuple[dict[str, dict], dict]:
    """PR #153 label records by question, after checking them against their build's counts.json."""
    counts = json.loads(Path(counts_path).read_text())
    if sha256_file(labels_path) != counts.get("labels_sha256"):
        raise SystemExit(f"{labels_path} does not match labels_sha256 in {counts_path}")
    records = read_jsonl(labels_path)
    labels = {r["question_id"]: r for r in records}
    if len(labels) != len(records):
        raise SystemExit(f"{labels_path}: a question appears more than once")
    return labels, counts


def probe_label(record: dict, policy_version: str) -> dict:
    """The labels.csv fields of one PR #153 label record."""
    return {"primary_target": TARGETS.get(record["label"]), "eligible_primary": record["label"] in TARGETS,
            "behavior_label": record["label"], "label_reason": record["reason"], "label_step": record["step"],
            "previous_label": record["previous_label"], "flags": record["flags"], "policy_version": policy_version}


def attach_labels(rows: list[dict], manifest: dict[str, dict], labels: dict[str, dict], policy_version: str) -> None:
    """Give each collected row its partition and PR #153 label; any disagreement with the build raises."""
    for row in rows:
        qid, partition = row["question_id"], manifest[row["question_id"]]["partition"]
        record = labels.get(qid)
        if record is None or not record["collected"]:
            raise SystemExit(f"{qid}: no collected PR #153 label in the build")
        if record["partition"] != partition:
            raise SystemExit(f"{qid}: PR #153 build has partition {record['partition']!r}, the manifest {partition!r}")
        row["partition"] = partition
        row["label"] = probe_label(record, policy_version)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True, help="final_labels.jsonl of a PR #153 build")
    parser.add_argument("--counts", type=Path, required=True, help="counts.json of the same build")
    parser.add_argument("--full-results", type=Path, required=True, help="the run's results folder")
    parser.add_argument("--frozen-partitions", type=Path, help="default for --layout partitioned: the run's partitions.csv")
    parser.add_argument("--coverage", choices=COVERAGE_MODES, default="complete")
    parser.add_argument("--layout", choices=("legacy", "partitioned"), default="legacy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.frozen_partitions is None:
        if args.layout != "partitioned":
            parser.error("--frozen-partitions is required for the legacy layout")
        args.frozen_partitions = args.full_results / "partitions.csv"
    ensure_output_available(args.output)

    labels, build = load_build(args.labels, args.counts)
    inputs = [args.labels, args.counts, args.frozen_partitions, *interaction_paths(args.full_results, layout=args.layout)]
    hashes_before = {str(path): sha256_file(path) for path in inputs}
    frozen = read_partitions(args.frozen_partitions, layout=args.layout)
    rows = read_run_interactions(args.full_results, layout=args.layout)
    excluded = {r["question_id"]: UNPARTITIONED for r in rows if r["question_id"] not in frozen}
    rows = [r for r in rows if r["question_id"] not in excluded]
    exclusions = (json.loads((args.full_results / "collection_status.json").read_text())["excluded"]
                  if args.layout == "partitioned" else [])
    manifest = build_partition_manifest(rows, frozen=frozen, coverage=args.coverage, layout=args.layout,
                                        exclusions=exclusions)
    attach_labels(rows, manifest, labels, build["policy_version"])

    args.output.mkdir(parents=True)
    write_labels(args.output / "labels.csv", rows)
    coverage = write_package_tables(args.output, manifest, frozen, layout=args.layout, exclusions=exclusions)
    write_json_atomic(args.output / "excluded_episodes.json", excluded)
    if {str(path): sha256_file(path) for path in inputs} != hashes_before:
        raise RuntimeError("an input file changed while the package was written")
    by_partition = {}
    for row in rows:
        by_partition.setdefault(row["partition"], Counter())[row["label"]["behavior_label"]] += 1
    write_json_atomic(args.output / "counts.json", {
        "label_policy": {"name": "PR #153 final", "version": build["policy_version"],
                         "labels_sha256": build["labels_sha256"]},
        "target": {"column": "primary_target", "eligible": "eligible_primary", "1": "adoption", "0": "retention"},
        "labels_by_partition": {p: dict(c) for p, c in sorted(by_partition.items())},
        "binary_by_partition": {p: {k: c[k] for k in TARGETS} for p, c in sorted(by_partition.items())},
        "excluded_episodes": len(excluded),
        "partition": {"frozen_partitions": str(args.frozen_partitions), "coverage": args.coverage,
                      "questions": {k: coverage[k] for k in ("expected", "collected", "missing")}},
        "code": {"git_commit": get_git_commit(), "uncommitted_changes_sha256": get_git_diff_sha256()},
        "source_files_sha256": hashes_before,
    })
    print(json.dumps({p: {k: c[k] for k in TARGETS} for p, c in sorted(by_partition.items())}))


if __name__ == "__main__":
    main()
