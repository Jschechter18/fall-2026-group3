"""Label rates for one label set, by source, partition and hop type (#174).

Run it on two label sets made the same way, e.g. Claude-labeled MuSiQue and synthetic pilot
episodes, then compare the tables. Rows cover every source x partition x hop-type scope, with
``all`` for any side (so the first row is the whole build). The source is the question-id
namespace (``musique``, ``synthetic``; ``unnamespaced`` for legacy ids), so a mixed pilot is
compared within one table. Each reports the four-class counts and shares, the
Adoption+Retention share, Adoption's share of Adoption+Retention, and A1 accuracy, each with a
95% Wilson interval. Small scopes are kept; ``n`` shows how far to trust them.

Every label row needs ``question_id``, ``label``, ``collected`` and ``partition`` (train,
validation, test, intervention, or null for a question outside the frozen partitions, reported
as ``unpartitioned``). A collected question must have exactly one interaction; an uncollected
one (a collection failure, labeled Out) must have none, so it counts in the four-class shares
but not in A1 accuracy. The hop type is the question-id prefix after any ``source:`` namespace.
Inputs are hashed before reading and rechecked before writing.

Two label sources:
- ``--labels``: a PR #153 ``final_labels.jsonl``, which includes collection failures.
- ``--package``: a probe package from ``export_final_labels_for_probes.py``, either PR #153 or Claude.
  Each ``labels.csv`` row is a collected episode with its ``behavior_label``. Episodes listed in
  ``excluded_episodes.json`` (e.g. Claude technical failures) are counted and left out, never relabeled.

Usage:
    PYTHONPATH=src python scripts/experiments/compare_label_rates.py \\
        --package <probe package> \\
        --interactions <run>/train/interactions.jsonl <run>/validation/interactions.jsonl \\
        --output <dir>
"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from mas_sae.data.musique import hop_type
from mas_sae.evaluation.claude_agreement import wilson
from mas_sae.evaluation.claude_validator import VALID_LABELS
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.records import read_jsonl, write_csv

PARTITIONS = ("train", "validation", "test", "intervention")
UNPARTITIONED = "unpartitioned"
ALL = "all"
UNNAMESPACED = "unnamespaced"


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


def source_of(question_id: str) -> str:
    """The ``source:`` namespace of a question id (``musique``, ``synthetic``), else ``unnamespaced``."""
    return question_id.split(":", 1)[0] if ":" in question_id else UNNAMESPACED


def hop_of(question_id: str) -> str:
    """Hop type such as ``3hop1``, ignoring a ``source:`` namespace."""
    return hop_type(question_id.split(":", 1)[-1])


def share(k: int, n: int) -> dict:
    """Share and 95% Wilson interval for k of n (None when n is 0)."""
    low, high = wilson(k, n) or (None, None)
    return {"share": k / n if n else None, "ci_low": low, "ci_high": high}


def scope_row(source: str, partition: str, hop: str, question_ids: list[str], labels: dict,
              interactions: dict) -> dict:
    """One output row: counts and shares for the given questions."""
    n = len(question_ids)
    counts = {label: sum(labels[q]["label"] == label for q in question_ids) for label in VALID_LABELS}
    binary = counts["adoption"] + counts["retention"]
    collected = [q for q in question_ids if q in interactions]
    correct = sum(interactions[q]["solver_attempt_1_correct"] for q in collected)
    row = {"source": source, "partition": partition, "hop_type": hop, "n": n}
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
    """One row per source x partition x hop-type scope, ``all`` first in each, partitions in order."""
    check_inputs(labels, interactions)
    scopes: dict[tuple[str, str, str], list[str]] = {}
    for question_id, row in labels.items():
        values = (source_of(question_id), partition_of(question_id, row), hop_of(question_id))
        for key in product(*((ALL, value) for value in values)):
            scopes.setdefault(key, []).append(question_id)
    order = (ALL, *PARTITIONS, UNPARTITIONED)
    keys = sorted(scopes, key=lambda key: (key[0] != ALL, key[0], order.index(key[1]), key[2] != ALL, key[2]))
    return [scope_row(*key, scopes[key], labels, interactions) for key in keys]


def read_package(folder: Path) -> tuple[list[dict], dict[str, str]]:
    """Label rows (each a collected episode) and excluded episodes of a probe package."""
    with (folder / "labels.csv").open(newline="", encoding="utf-8") as fh:
        rows = [{"question_id": row["question_id"], "label": row["behavior_label"], "collected": True,
                 "partition": row["partition"] or None} for row in csv.DictReader(fh)]
    return rows, json.loads((folder / "excluded_episodes.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels", type=Path, help="a PR #153 final_labels.jsonl")
    source.add_argument("--package", type=Path, help="a probe package folder (PR #153 or Claude labels)")
    parser.add_argument("--interactions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    label_files = ([args.labels] if args.labels
                   else [args.package / "labels.csv", args.package / "excluded_episodes.json"])
    paths = [*label_files, *args.interactions]
    hashes = {str(path): sha256_file(path) for path in paths}
    excluded = {}
    if args.labels:
        labels = by_question(read_jsonl(args.labels), str(args.labels))
    else:
        package_rows, excluded = read_package(args.package)
        labels = by_question(package_rows, str(args.package / "labels.csv"))
        labeled_excluded = sorted(set(excluded) & set(labels))
        if labeled_excluded:
            raise ValueError(f"excluded episodes also have labels, e.g. {labeled_excluded[:3]}")
    interactions = by_question([row for path in args.interactions for row in read_jsonl(path)], "interactions")
    interactions = {qid: row for qid, row in interactions.items() if qid not in excluded}
    rows = label_rates(labels, interactions)
    changed = [path for path in paths if sha256_file(path) != hashes[str(path)]]
    if changed:
        raise RuntimeError(f"inputs changed while reading: {[str(path) for path in changed]}")

    args.output.mkdir(parents=True)
    write_csv(args.output / "label_rates.csv", rows)
    (args.output / "summary.json").write_text(
        json.dumps({"inputs": hashes, "excluded_episodes": excluded, "rows": rows}, indent=2) + "\n",
        encoding="utf-8")
    for row in rows:
        if sum(row[k] != ALL for k in ("source", "partition", "hop_type")) > 1:
            continue                                    # crossed scopes are in the CSV
        a1 = "n/a" if row["a1_accuracy_share"] is None else f"{row['a1_accuracy_share']:.1%}"
        print(f"{row['source']:>12} {row['partition']:>13} {row['hop_type']:>6}  n={row['n']:>6}  "
              + "  ".join(f"{label} {row[f'{label}_share']:.1%}" for label in VALID_LABELS)
              + f"  A+R {row['binary_share']:.1%}  A1 acc {a1}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
