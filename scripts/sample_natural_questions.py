"""Draw Natural train samples for Claude validation, proportional to the pool.

Membership comes from the frozen partition file via
``adoption_experiment.training_natural_rows`` (Natural only, partition ``train``).
Questions in any ``--exclude`` file are never drawn or counted.  Selection never
looks at A2, behavior or lexical labels.  Writes the unmodified source records
plus a ``sample.json`` with the seed, input hashes, allocation and question IDs.

Without ``--cohort``: a fresh sample of ``--size`` questions, proportional by hop.
With ``--cohort``: ``--size`` NEW questions such that cohort + new is proportional
across ``adoption_experiment.stratum`` (hop x A1 correct x Critic correct, with
``unknown`` where the flag is missing).  Only the new records are written, so the
cohort is never labeled twice.

Usage:
    PYTHONPATH=src python scripts/sample_natural_questions.py \\
        --interactions results/collection/natural_4b_full9_partitioned/train/interactions.jsonl \\
        --exclude results/adoption_experiment/sample_100q/sampled_episodes.jsonl \\
        --output results/claude_validation/samples/natural_hop100_seed20261029

    # grow that cohort to 500
    PYTHONPATH=src python scripts/sample_natural_questions.py \\
        --interactions results/collection/natural_4b_full9_partitioned/train/interactions.jsonl \\
        --exclude results/adoption_experiment/sample_100q/sampled_episodes.jsonl \\
        --cohort results/claude_validation/samples/natural_hop100_seed20261029/sampled_episodes.jsonl \\
        --size 400 \\
        --output results/claude_validation/samples/natural_topup400_to500_seed20261029
"""

import argparse
import json
from pathlib import Path

from mas_sae.experiments.adoption_experiment import stratum, training_natural_rows
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.hop_sample import proportional_hop_sample, require_known_hops, stratified_top_up
from mas_sae.experiments.records import read_jsonl, write_jsonl

PROTOCOL = "claude_validation_natural_hop_proportional_v1"
TOP_UP_PROTOCOL = "claude_validation_natural_stratified_top_up_v1"
DEFAULT_PARTITIONS = Path("results/behavior/behavior_v011_split80_10_10_20261006/partitions.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[],
                        help="JSONL files whose question_ids must not be drawn (e.g. earlier samples).")
    parser.add_argument("--cohort", type=Path, nargs="*", default=[],
                        help="JSONL files of the existing cohort to grow (top-up mode).")
    parser.add_argument("--size", type=int, default=100, help="Number of NEW questions to draw.")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    ensure_output_available(args.output)

    pool = training_natural_rows(read_jsonl(args.interactions), args.partitions)
    exclude_ids = {r["question_id"] for path in args.exclude for r in read_jsonl(path)}
    cohort_ids = {r["question_id"] for path in args.cohort for r in read_jsonl(path)}
    if cohort_ids & exclude_ids:
        raise SystemExit("A cohort question is also in an --exclude file.")

    if args.cohort:
        require_known_hops(pool)  # stratum() would put unparseable hops in "unknown"
        selected, allocation = stratified_top_up(pool, stratum, cohort_ids, args.size, args.seed, exclude_ids)
        protocol = TOP_UP_PROTOCOL
        selection = ("cohort + new proportional across adoption_experiment.stratum "
                     "(hop x A1 correct x Critic correct); never uses A2, behavior or lexical labels")
    else:
        selected, allocation = proportional_hop_sample(pool, args.size, args.seed, exclude_ids)
        protocol = PROTOCOL
        selection = "proportional by hop_group; never uses A2, behavior or lexical labels"

    packet = {
        "protocol": protocol,
        "seed": args.seed,
        "size": args.size,
        "cohort_size": len(cohort_ids),
        "cumulative_size": len(cohort_ids) + len(selected),
        "selection": selection,
        "pool_natural_train": len(pool),
        "excluded_prior_questions": len(exclude_ids),
        "allocation": allocation,
        "input_hashes": {str(p): sha256_file(p)
                         for p in (args.interactions, args.partitions, *args.exclude, *args.cohort)},
        "question_ids": [r["question_id"] for r in selected],
    }
    args.output.mkdir(parents=True)
    (args.output / "sample.json").write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output / "sampled_episodes.jsonl", selected)
    print(json.dumps({k: packet[k] for k in ("protocol", "seed", "pool_natural_train", "excluded_prior_questions",
                                             "cohort_size", "cumulative_size", "allocation")}, indent=2))
    print(f"Wrote {len(selected)} episodes to {args.output}")


if __name__ == "__main__":
    main()
