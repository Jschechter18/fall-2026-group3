"""Audit Critic feedback quality against a behavior-label release (issue #115).

Writes one row per episode of one development partition: whether the delivered
Critic feedback was coherent, what the release's label rules say the Solver did,
answer-level warnings, and correctness (recorded only). Eligible rows get an
automatic candidate status (rule_pass / uncertain / invalid) for review; it is
a screen, not a verified label. The release and the interaction files are only
read; their hashes are checked before and after.

Only the interaction file that can hold the partition is opened: a train run
never opens validation/. Test and intervention rows share the train file; they
are dropped as soon as their id is read and are also refused by the partition
guard.

Train runs develop the coherence rules and can write a blind review packet.
The validation partition is read once, after the whole procedure is frozen:
pass the frozen hash with --expect-procedure-sha256. It covers the rule
settings and the code of the parser, matching and rules, so any change to
them invalidates the freeze.

Usage:
    PYTHONPATH=src python scripts/audit_feedback_quality.py \\
        --results ~/capstone-artifacts/natural_4b_full/results \\
        --labels ~/capstone-artifacts/behavior_v012_official_20261011_release \\
        --partition train \\
        --human-qc ~/capstone-artifacts/qc_v011_review/human_qc_559_unified.csv \\
        --human-notes ~/capstone-artifacts/issue55_human_evidence/annotator_a_completed_original.csv \\
        --review-packet 5 --review-packet-classes non_answers \\
        --already-reviewed ~/capstone-artifacts/issue115_feedback_audit_20261013/train/review_packet/private_key.json \\
        --output ~/capstone-artifacts/issue115_feedback_audit_20261014/train
"""
import argparse
import csv
import json
from pathlib import Path

import yaml

from mas_sae.data.production import authorize_rows, interaction_paths, load_labeled_rows
from mas_sae.evaluation import feedback_audit, feedback_coherence
from mas_sae.evaluation.behavior_qc import source_paragraphs, write_annotator_files
from mas_sae.experiments.artifacts import (
    ensure_output_available,
    get_git_commit,
    get_git_diff_sha256,
    sha256_file,
    write_json_atomic,
)

PURPOSE = {"train": "fit", "validation": "tune"}


def read_human(qc_path, notes_path):
    """{question_id: human review} from the adjudicated QC file and annotator notes."""
    with open(qc_path, newline="", encoding="utf-8-sig") as f:
        adjudicated = list(csv.DictReader(f))
    notes = {}
    if notes_path is not None:
        with open(notes_path, newline="", encoding="utf-8-sig") as f:
            notes = {r["review_id"]: r["notes"] for r in csv.DictReader(f)}
    return {
        r["question_id"]: {
            "review_id": r["review_id"],
            "eligible_primary": r["eligible_primary"],
            "feedback_type": r["feedback_type"],
            "solver_response": r["solver_response"],
            "contradiction_noted": "contradict" in notes.get(r["review_id"], "").lower(),
        }
        for r in adjudicated
    }


def write_csv(path, records):
    fields = list(records[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: ";".join(v) if isinstance(v, list) else ("" if v is None else v)
                             for k, v in r.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True,
                        help="run results folder with train/ and validation/ interactions.jsonl")
    parser.add_argument("--labels", type=Path, required=True,
                        help="label release folder with labels.csv and split_manifest.csv (read only)")
    parser.add_argument("--partition", choices=sorted(PURPOSE), required=True)
    parser.add_argument("--output", type=Path, required=True, help="new folder for the audit")
    parser.add_argument("--human-qc", type=Path, help="adjudicated human QC CSV (review_id, question_id, ...)")
    parser.add_argument("--human-notes", type=Path, help="annotator CSV whose notes mark contradictions")
    parser.add_argument("--review-packet", type=int, metavar="PER_GROUP",
                        help="train only: write a blind review packet with this many rows per group")
    parser.add_argument("--review-packet-classes", choices=feedback_audit.REVIEW_CLASSES,
                        default="conflicts", help="which rows the review packet samples")
    parser.add_argument("--already-reviewed", type=Path, nargs="+", default=[],
                        metavar="PRIVATE_KEY",
                        help="private_key.json of earlier packets; their questions are not sampled again")
    parser.add_argument("--seed", type=int, default=42, help="review packet sampling seed")
    parser.add_argument("--expect-procedure-sha256",
                        help="frozen audit procedure hash; required for the validation partition")
    parser.add_argument("--layout", choices=("legacy", "partitioned"), default="legacy",
                        help="Results directory layout; distributed results use partitioned.")
    args = parser.parse_args()

    procedure_sha = feedback_audit.procedure_sha256()
    if args.partition == "validation":
        if args.expect_procedure_sha256 is None:
            parser.error("validation is read once, with a frozen procedure: pass --expect-procedure-sha256")
        if args.review_packet is not None:
            parser.error("review packets come from the train partition only")
    if args.expect_procedure_sha256 is not None and args.expect_procedure_sha256 != procedure_sha:
        parser.error(f"audit procedure hash is {procedure_sha}, expected {args.expect_procedure_sha256}")
    ensure_output_available(args.output)

    input_files = interaction_paths(args.results, {args.partition}, layout=args.layout)
    input_files += [args.labels / "labels.csv", args.labels / "split_manifest.csv"]
    hashes_before = {str(p): sha256_file(p) for p in input_files}

    rows, manifest = load_labeled_rows(args.results, args.labels, partitions={args.partition}, layout=args.layout)
    authorize_rows(rows, manifest, purpose=PURPOSE[args.partition])

    records = [feedback_audit.audit_row(r) for r in rows]
    human = read_human(args.human_qc, args.human_notes) if args.human_qc else {}
    for r in records:
        h = human.get(r["question_id"], {})
        r["human_review_id"] = h.get("review_id")
        r["human_eligible_primary"] = h.get("eligible_primary")
        r["human_feedback_type"] = h.get("feedback_type")
        r["human_solver_response"] = h.get("solver_response")
        r["human_contradiction_noted"] = h.get("contradiction_noted")

    args.output.mkdir(parents=True)
    write_csv(args.output / "audit.csv", records)
    counts = {
        "partition": args.partition,
        "procedure_sha256": procedure_sha,
        "coherence_rules_sha256": feedback_coherence.rules_sha256(),
        "procedure_frozen": args.expect_procedure_sha256 is not None,
        "code": {
            "git_commit": get_git_commit(),
            "uncommitted_changes_sha256": get_git_diff_sha256(),
            "procedure_files_sha256": {p.name: sha256_file(p) for p in feedback_audit.procedure_files()},
        },
        "inputs": {"results": str(args.results), "labels": str(args.labels),
                   "human_qc": None if args.human_qc is None else str(args.human_qc),
                   "human_notes": None if args.human_notes is None else str(args.human_notes)},
        "source_files_sha256": hashes_before,
        "label_rule_version": sorted({r["label_rule_version"] for r in records}),
        "audit": feedback_audit.audit_counts(records),
        "human": feedback_audit.human_comparison(records, human) if human else None,
    }

    if args.review_packet is not None:
        packet_dir = args.output / f"review_packet_{args.review_packet_classes}"
        packet_dir.mkdir()
        reviewed = set(human)
        for key_path in args.already_reviewed:
            reviewed |= {entry["question_id"] for entry in json.loads(key_path.read_text())}
        chosen, key, sampling = feedback_audit.choose_review_rows(
            records, reviewed, per_group=args.review_packet, seed=args.seed,
            classes=args.review_packet_classes)
        sampling["already_reviewed_keys"] = [str(p) for p in args.already_reviewed]
        by_id = {r["question_id"]: r for r in rows}
        chosen_rows = [by_id[r["question_id"]] for r in chosen]
        run_config = yaml.safe_load((args.results / "train" / "resolved_config.yaml").read_text())
        paragraphs = source_paragraphs([r["question_id"] for r in chosen_rows],
                                       run_config["dataset"]["revision"])
        write_annotator_files(packet_dir, chosen_rows, key, paragraphs)
        (packet_dir / "private_key.json").write_text(json.dumps(key, indent=2) + "\n")
        write_json_atomic(packet_dir / "sampling.json", sampling)
        counts["review_packet"] = sampling

    write_json_atomic(args.output / "counts.json", counts)
    if {str(p): sha256_file(p) for p in input_files} != hashes_before:
        raise RuntimeError("An input file changed while the audit ran")
    print(json.dumps({"output": str(args.output), "procedure_sha256": procedure_sha,
                      "strict_negatives": counts["audit"]["strict_negatives"]}, indent=2))


if __name__ == "__main__":
    main()
