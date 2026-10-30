"""Label rates for one PR #153 label build, by partition and hop type (#174).

Run it once on the approved MuSiQue labels and once on the synthetic pilot, then compare the
two tables. Rows cover every partition x hop-type scope, with ``all`` for either side (so the
first row is the whole build). Each reports the four-class counts and shares, the
Adoption+Retention share, Adoption's share of Adoption+Retention, and A1 accuracy, each with a
95% Wilson interval. Small scopes are kept; ``n`` shows how far to trust them.

Every label row needs ``question_id``, ``label``, ``collected`` and ``partition`` (train,
validation, test, intervention, or null for a question outside the frozen partitions, reported
as ``unpartitioned``). A collected question must have exactly one interaction; an uncollected
one (a collection failure, labeled Out) must have none, so it counts in the four-class shares
but not in A1 accuracy. The hop type is the question-id prefix after any ``source:`` namespace.
Inputs are hashed before reading and rechecked before writing.

Usage:
    PYTHONPATH=src python scripts/experiments/compare_label_rates.py \\
        --labels <build>/final_labels.jsonl \\
        --interactions <run>/train/interactions.jsonl <run>/validation/interactions.jsonl \\
        --output <dir>
"""

import argparse
import json
from pathlib import Path

from mas_sae.data.musique import hop_type
from mas_sae.evaluation.claude_agreement import wilson
from mas_sae.evaluation.claude_validator import VALID_LABELS
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.records import read_jsonl, write_csv

PARTITIONS = ("train", "validation", "test", "intervention")
UNPARTITIONED = "unpartitioned"
ALL = "all"


def by_question(rows: list[dict], source: str) -> dict[str, dict]:
    """Index rows by ``question_id``, refusing repeats."""
    indexed = {}
    for row in rows:
        question_id = row["question_id"]
        if question_id in indexed:
            raise ValueError(f"{source}: repeated question_id {question_id!r}")
        indexed[question_id] = row
    return indexed


def partition_of(question_id: str, row: dict) -> str:
    """The label's partition, ``unpartitioned`` for null; refuses a missing or unknown one."""
    if "partition" not in row:
        raise ValueError(f"{question_id!r}: label has no partition field")
    if row["partition"] is None:
        return UNPARTITIONED
    if row["partition"] not in PARTITIONS:
        raise ValueError(f"{question_id!r}: unknown partition {row['partition']!r}")
    return row["partition"]


def check_inputs(labels: dict[str, dict], interactions: dict[str, dict]) -> None:
    """Refuse unknown labels or partitions, unmatched rows and non-boolean A1 correctness."""
    unknown = sorted({row["label"] for row in labels.values()} - set(VALID_LABELS))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    for question_id, row in labels.items():
        partition_of(question_id, row)
    unlabeled = sorted(interactions.keys() - labels.keys())
    if unlabeled:
        raise ValueError(f"{len(unlabeled)} interactions without a label, e.g. {unlabeled[:3]}")
    for question_id, row in labels.items():
        if row["collected"] is not (question_id in interactions):
            raise ValueError(f"{question_id!r}: collected={row['collected']!r} disagrees with the interactions")
    for question_id, row in interactions.items():
        if not isinstance(row["solver_attempt_1_correct"], bool):
            raise ValueError(f"{question_id!r}: solver_attempt_1_correct is not a boolean")


def hop_of(question_id: str) -> str:
    """Hop type such as ``3hop1``, ignoring a ``source:`` namespace."""
    return hop_type(question_id.split(":", 1)[-1])


def share(k: int, n: int) -> dict:
    """Share and 95% Wilson interval for k of n (None when n is 0)."""
    low, high = wilson(k, n) or (None, None)
    return {"share": k / n if n else None, "ci_low": low, "ci_high": high}


def scope_row(partition: str, hop: str, question_ids: list[str], labels: dict, interactions: dict) -> dict:
    """One output row: counts and shares for the given questions."""
    n = len(question_ids)
    counts = {label: sum(labels[q]["label"] == label for q in question_ids) for label in VALID_LABELS}
    binary = counts["adoption"] + counts["retention"]
    collected = [q for q in question_ids if q in interactions]
    correct = sum(interactions[q]["solver_attempt_1_correct"] for q in collected)
    row = {"partition": partition, "hop_type": hop, "n": n}
    for label in VALID_LABELS:
        row[label] = counts[label]
        row.update({f"{label}_{key}": value for key, value in share(counts[label], n).items()})
    row["binary"] = binary
    row.update({f"binary_{key}": value for key, value in share(binary, n).items()})
    row.update({f"adoption_of_binary_{key}": value
                for key, value in share(counts["adoption"], binary).items()})
    row["collected"] = len(collected)
    row["a1_correct"] = correct
    row.update({f"a1_accuracy_{key}": value for key, value in share(correct, len(collected)).items()})
    return row


def label_rates(labels: dict[str, dict], interactions: dict[str, dict]) -> list[dict]:
    """One row per partition x hop-type scope, ``all`` first on each side, partitions in order."""
    check_inputs(labels, interactions)
    scopes: dict[tuple[str, str], list[str]] = {}
    for question_id, row in labels.items():
        partition, hop = partition_of(question_id, row), hop_of(question_id)
        for key in ((ALL, ALL), (partition, ALL), (ALL, hop), (partition, hop)):
            scopes.setdefault(key, []).append(question_id)
    order = (ALL, *PARTITIONS, UNPARTITIONED)
    keys = sorted(scopes, key=lambda key: (order.index(key[0]), key[1] != ALL, key[1]))
    return [scope_row(partition, hop, scopes[partition, hop], labels, interactions) for partition, hop in keys]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    paths = [args.labels, *args.interactions]
    hashes = {str(path): sha256_file(path) for path in paths}
    labels = by_question(read_jsonl(args.labels), str(args.labels))
    interactions = by_question([row for path in args.interactions for row in read_jsonl(path)], "interactions")
    rows = label_rates(labels, interactions)
    changed = [path for path in paths if sha256_file(path) != hashes[str(path)]]
    if changed:
        raise RuntimeError(f"inputs changed while reading: {[str(path) for path in changed]}")

    args.output.mkdir(parents=True)
    write_csv(args.output / "label_rates.csv", rows)
    (args.output / "summary.json").write_text(
        json.dumps({"inputs": hashes, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        if ALL not in (row["partition"], row["hop_type"]):
            continue                                    # partition x hop rows are in the CSV
        print(f"{row['partition']:>13} {row['hop_type']:>6}  n={row['n']:>6}  "
              + "  ".join(f"{label} {row[f'{label}_share']:.1%}" for label in VALID_LABELS)
              + f"  A+R {row['binary_share']:.1%}  A1 acc {row['a1_accuracy_share'] or 0:.1%}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
