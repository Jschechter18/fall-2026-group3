"""Build the behavior label package from the finished production run.

Writes three things into a new folder:

    labels.csv           one row per question: the new labels next to lexical_v2
    split_manifest.csv   one row per question: which partition it belongs to
    counts.json          label counts, the split audit and the source file hashes

The collection run folders are only read, never changed.

    PYTHONPATH=src python scripts/prepare_behavior_v01.py \\
        --artifacts /home/ubuntu/capstone-artifacts \\
        --output /home/ubuntu/capstone-artifacts/<new folder> \\
        --train-proportion 0.8 --test-proportion 0.1 --intervention-proportion 0.1 \\
        --split-seed 42

The partition rule: every MuSiQue-validation question is ``validation``;
MuSiQue-train questions are cut by question id, stratified by hop count, into
train / test / intervention with the proportions given here. The team agreed
0.8 / 0.1 / 0.1 with seed 42 on 2026-10-06. That cut was made once; the result
is the committed ``partitions.csv`` and every later run keeps it:

    PYTHONPATH=src python scripts/prepare_behavior_v01.py \\
        --full-results <artifacts>/<rerun>/results \\
        --output <artifacts>/<new folder> \\
        --frozen-partitions results/behavior/behavior_v011_split80_10_10_20261006/partitions.csv \\
        --coverage complete

With --frozen-partitions membership is read from the file and never recomputed;
only the activation indices are rebuilt for the new run. A question the file
does not know stops the run. A run that deliberately covers part of the
question set, such as a layer scan, passes --coverage subset and the uncovered
ids are written next to the package.

--label-policy picks the rule set that labels every episode (see
``mas_sae.evaluation.behavior_labels.POLICIES``). The default is the frozen
first rule set, v0.1.1, until the team adopts another under issue #108. The
policy's version and the hash of its full definition go into counts.json, so a
package always says which rules produced it.

Human QC is finished: no policy writes an annotation packet. The packets that
were issued, and their annotations, stay unchanged in the packages they came
from.
"""
import argparse
import json
from pathlib import Path

from mas_sae.data.production import (
    COVERAGE_MODES,
    interaction_paths,
    build_partition_manifest,
    check_scan_repeats_full_run,
    read_partitions,
    read_run_interactions,
    split_audit,
    write_package_tables,
)
from mas_sae.evaluation import behavior, behavior_labels, feedback_coherence, scoring
from mas_sae.evaluation.behavior_labels import (
    DEFAULT_POLICY_NAME,
    POLICIES,
    classify_candidate,
    label_counts,
    write_labels,
)
from mas_sae.experiments.artifacts import (
    ensure_output_available,
    get_git_commit,
    get_git_diff_sha256,
    sha256_file,
    write_json_atomic,
)

# Where the first production run, the layer scan and the first QC sample live
# inside the artifacts folder. Defaults only; any run can be passed explicitly.
FULL_RESULTS = "natural_4b_full/results"
SCAN_RESULTS = "issue69_natural_4b_layer_scan/results/collection/natural_4b_layer_scan"
FIRST_QC_SAMPLE = "issue55_human_qc/sample_manifest.json"

# The modules whose code decides the labels; their hashes go into counts.json.
# Candidate rules also write the feedback-coherence diagnostic.
RULE_MODULES = (behavior_labels, behavior, scoring)
CANDIDATE_RULE_MODULES = (feedback_coherence,)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path,
                        help="folder holding the collection runs; gives the historical defaults for "
                             "--full-results, --scan-results and --first-qc-sample")
    parser.add_argument("--full-results", type=Path,
                        help="the run's results folder, with train/ and validation/ interactions.jsonl "
                             f"(default: <artifacts>/{FULL_RESULTS})")
    parser.add_argument("--scan-results", type=Path,
                        help="a layer scan's results folder to record alongside the run "
                             f"(default: <artifacts>/{SCAN_RESULTS} when that folder exists)")
    parser.add_argument("--first-qc-sample", type=Path,
                        help="sample_manifest.json of the first 100-row QC sample "
                             f"(default: <artifacts>/{FIRST_QC_SAMPLE} when that file exists)")
    parser.add_argument("--output", type=Path, required=True,
                        help="new folder for the package")
    parser.add_argument("--frozen-partitions", type=Path,
                        help="partitions.csv to take every question's partition from, never recomputing "
                             "membership; use this for every run after the first")
    parser.add_argument("--label-policy", choices=sorted(POLICIES), default=DEFAULT_POLICY_NAME,
                        help="rule set that labels every episode; the default is the frozen first "
                             "rule set until the team adopts another (issue #108)")
    parser.add_argument("--coverage", choices=COVERAGE_MODES, default="complete",
                        help="with --frozen-partitions: 'complete' fails if any frozen question is "
                             "missing from the run; 'subset' records the missing ids instead")
    parser.add_argument("--train-proportion", type=float,
                        help="first derivation only: share of MuSiQue-train questions that become train")
    parser.add_argument("--test-proportion", type=float,
                        help="first derivation only: share of MuSiQue-train questions that become test")
    parser.add_argument("--intervention-proportion", type=float,
                        help="first derivation only: share reserved for the causal experiment")
    parser.add_argument("--split-seed", type=int,
                        help="first derivation only: seed of the hop-stratified cut")
    parser.add_argument("--layout", choices=("legacy", "partitioned"), default="legacy",
                        help="Input directory layout; partitioned uses the distributed finalizer's manifests and test folder.")
    args = parser.parse_args()
    if args.layout == "partitioned" and args.frozen_partitions is None:
        if args.full_results is None:
            parser.error("--layout partitioned requires --full-results")
        args.frozen_partitions = args.full_results / "partitions.csv"

    deriving = args.frozen_partitions is None
    derivation = (args.train_proportion, args.test_proportion, args.intervention_proportion, args.split_seed)
    if deriving and any(value is None for value in derivation):
        parser.error("without --frozen-partitions, --train-proportion, --test-proportion, "
                     "--intervention-proportion and --split-seed are all required")
    if not deriving and any(value is not None for value in derivation):
        parser.error("--frozen-partitions keeps the partitions as they are; do not pass proportions or a seed")
    if args.full_results is None and args.artifacts is None:
        parser.error("pass --full-results, or --artifacts for the historical layout")

    full_dir = args.full_results or args.artifacts / FULL_RESULTS
    scan_dir = args.scan_results
    if scan_dir is None and args.artifacts is not None and (args.artifacts / SCAN_RESULTS).is_dir():
        scan_dir = args.artifacts / SCAN_RESULTS
    qc_sample = args.first_qc_sample
    if qc_sample is None and args.artifacts is not None and (args.artifacts / FIRST_QC_SAMPLE).is_file():
        qc_sample = args.artifacts / FIRST_QC_SAMPLE
    policy = POLICIES[args.label_policy]
    ensure_output_available(args.output)

    # Remember the input files' hashes so we can confirm they were not changed.
    if args.layout == "partitioned" and scan_dir is not None:
        parser.error("Partitioned full runs do not use the historical scan-merge path")
    input_files = [path for run_dir in (full_dir, scan_dir) if run_dir is not None
                   for path in interaction_paths(run_dir, layout=args.layout)]
    hashes_before = {str(path): sha256_file(path) for path in input_files}

    # 1. Read the run(s) and give each question its partition.
    full_rows = read_run_interactions(full_dir, layout=args.layout)
    if not full_rows:
        parser.error("No collected episodes to label; inspect collection_status.json for exclusions")
    scan_rows = read_run_interactions(scan_dir) if scan_dir is not None else []
    if scan_rows:
        check_scan_repeats_full_run(full_rows, scan_rows, policy.input_fields)
    qc100_ids = []
    if qc_sample is not None:
        qc100_ids = [sample["question_id"] for sample in json.loads(qc_sample.read_text())["samples"]]

    if deriving:
        proportions = {"train": args.train_proportion, "test": args.test_proportion,
                       "intervention": args.intervention_proportion}
        frozen = None
        manifest = build_partition_manifest(full_rows, scan_rows, qc100_ids,
                                            proportions=proportions, seed=args.split_seed)
    else:
        frozen = read_partitions(args.frozen_partitions, layout=args.layout)
        if args.layout == "partitioned":
            # Historical QC files use original MuSiQue IDs; only that source
            # inherits their exposure, even if synthetic IDs share the suffix.
            musique_ids = {r["original_id"]: q for q, r in frozen.items() if r["dataset_source"] == "musique"}
            for original_id in qc100_ids:
                qid = original_id if original_id in frozen else musique_ids.get(original_id)
                if qid is not None:
                    frozen[qid].update(in_qc100=True, in_development_data=True,
                                       prior_development_exposure=True, exposure_history_known=True)
        exclusions = (json.loads((full_dir/"collection_status.json").read_text())["excluded"]
                      if args.layout == "partitioned" else [])
        manifest = build_partition_manifest(full_rows, scan_rows, qc100_ids,
                                            frozen=frozen, coverage=args.coverage,
                                            layout=args.layout, exclusions=exclusions)

    # 2. Label every episode of the full run.
    for row in full_rows:
        row["partition"] = manifest[row["question_id"]]["partition"]
        row["label"] = classify_candidate(row, policy)

    # 3. Write the tables and the counts. partitions.csv is the frozen mapping
    #    in full whenever one is in use, however much of it this run covers.
    args.output.mkdir(parents=True)
    write_labels(args.output / "labels.csv", full_rows)
    coverage = write_package_tables(args.output, manifest, frozen, layout=args.layout,
                                    exclusions=exclusions if not deriving else ())
    if deriving:
        partition_record = {"proportions_of_source_train": proportions, "seed": args.split_seed,
                            "rule": "MuSiQue-validation -> validation; MuSiQue-train cut by sorted "
                                    "question id, stratified by hop group"}
    else:
        partition_record = {"frozen_partitions": str(args.frozen_partitions),
                            "frozen_partitions_sha256": sha256_file(args.frozen_partitions),
                            "coverage": args.coverage,
                            "questions": {k: coverage[k] for k in ("expected", "collected", "missing")},
                            "rule": "membership taken from the frozen mapping; only activation "
                                    "indices were rebuilt for this run"}

    train_rows = [row for row in full_rows if row["source_split"] == "train"]
    validation_rows = [row for row in full_rows if row["source_split"] == "validation"]
    counts = {
        "rule_version": full_rows[0]["label"]["rule_version"],
        "label_policy": {
            "name": args.label_policy,
            "version": policy.version,
            "definition_sha256": policy.definition_sha256(),
            "overrides": {"premise": list(policy.premise_overrides),
                          "refusal": list(policy.refusal_overrides),
                          "a2_nonanswer": list(policy.a2_nonanswer_overrides)},
            "candidate_rules": (None if policy.candidate_rules is None
                                else policy.candidate_rules.definition()),
        },
        "code": {
            "git_commit": get_git_commit(),
            "uncommitted_changes_sha256": get_git_diff_sha256(),
            "rule_files_sha256": {Path(module.__file__).name: sha256_file(module.__file__)
                                  for module in RULE_MODULES + (
                                      CANDIDATE_RULE_MODULES if policy.candidate_rules else ())},
        },
        "source_files_sha256": hashes_before,
        "inputs": {"full_results": str(full_dir),
                   "scan_results": None if scan_dir is None else str(scan_dir),
                   "first_qc_sample": None if qc_sample is None else str(qc_sample)},
        "partition": partition_record,
        "all": label_counts(full_rows),
        "source_train": label_counts(train_rows),
        "source_validation": label_counts(validation_rows),
        "by_storage_split": {split: label_counts([r for r in full_rows if r["source_split"] == split])
                             for split in sorted({r["source_split"] for r in full_rows})},
        "layout": args.layout,
        "split_audit": split_audit(manifest),
    }
    write_json_atomic(args.output / "counts.json", counts)

    # 4. Confirm the inputs are untouched and print a short summary.
    if {str(path): sha256_file(path) for path in input_files} != hashes_before:
        raise RuntimeError("A source file changed while the package was being built")
    summary = {
        "output": str(args.output),
        "primary_target_by_split": counts["all"]["primary_target_by_split"],
        "split_audit": counts["split_audit"],
    }
    if policy.candidate_rules is not None:
        for name in ("strict_target_by_split", "first_token_target_by_split"):
            summary[name] = counts["all"].get(name, {})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
