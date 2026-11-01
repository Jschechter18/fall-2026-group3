"""Write a seeded, hop-proportional sample of one partition's questions as a JSON id list.

For a collection config with ``sampling.strategy: listed``. The partition is
read from a Behavior package manifest, which is first checked against the
committed frozen mapping. Questions flagged ``in_development_data`` are never
drawn, and neither is any MuSiQue id found in an ``--exclude`` file (any
format: QC packets, review keys, earlier sample lists).

    PYTHONPATH=src python scripts/data/select_partition_questions.py \\
        --package /home/ubuntu/capstone-artifacts/behavior_v012_official_20261011 \\
        --partition train --num-questions 300 --seed 20261023 \\
        --exclude <files with question ids already used> \\
        --output configs/collection/v3/pilot_question_ids.json
"""
import argparse
import json
from pathlib import Path

from mas_sae.data.production import (
    PARTITIONS,
    PARTITIONS_FILE,
    check_manifest_matches_frozen,
    question_ids_in,
    read_manifest,
    read_partitions,
    sample_partition_questions,
)
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file, write_json_atomic

FROZEN_PARTITIONS = (Path(__file__).resolve().parents[2] / "results" / "behavior"
                     / "behavior_v011_split80_10_10_20261006" / PARTITIONS_FILE)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True,
                        help="Behavior package folder holding split_manifest.csv")
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--num-questions", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-partitions", type=Path, default=FROZEN_PARTITIONS)
    parser.add_argument("--layout", choices=("legacy", "partitioned"), default="legacy",
                        help="Partitioned uses namespaced master IDs; --exclude files must then be JSON ID lists.")
    args = parser.parse_args()

    ensure_output_available(args.output)
    manifest = read_manifest(args.package / "split_manifest.csv")
    frozen = read_partitions(args.frozen_partitions, layout=args.layout)
    check_manifest_matches_frozen(manifest, frozen)
    excluded = set()
    for path in args.exclude:
        if args.layout == "partitioned":
            ids = json.loads(path.read_text())
            if not isinstance(ids, list) or any(not isinstance(qid, str) for qid in ids):
                raise ValueError("Partitioned exclusion files must be JSON lists of namespaced question IDs")
            unknown = set(ids) - set(frozen)
            if unknown:
                raise ValueError(f"{path}: unknown namespaced IDs: {sorted(unknown)[:5]}")
            excluded.update(ids)
        else:
            excluded.update(question_ids_in(path))
    ids = sample_partition_questions(manifest, partition=args.partition,
                                     num_questions=args.num_questions, seed=args.seed, exclude=excluded)

    args.output.write_text(json.dumps(ids, indent=2) + "\n")
    write_json_atomic(args.output.with_suffix(".provenance.json"), {
        "package_manifest_sha256": sha256_file(args.package / "split_manifest.csv"),
        "frozen_partitions_sha256": sha256_file(args.frozen_partitions),
        "partition": args.partition, "num_questions": args.num_questions, "seed": args.seed,
        "excluded_files": {str(path): sha256_file(path) for path in args.exclude},
        "excluded_ids": len(excluded),
        "rule": "hop-proportional seeded sample; in_development_data and excluded ids never drawn",
        "output_sha256": sha256_file(args.output),
    })
    print(f"{len(ids)} {args.partition} questions -> {args.output}")


if __name__ == "__main__":
    main()
