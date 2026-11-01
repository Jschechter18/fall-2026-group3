"""Apply the approved PR #153 label rules to a partitioned (distributed) collection export.

The rules are ``label_episode`` and ``failure_record`` of
``results/experiments/candidate_final_labels_20261027/final_labels.py``, imported unchanged: the build
refuses to run unless that file and ``proposal_snapshot.py`` are byte-identical to the ones the approved
``counts.json`` records. Only the input side differs from the approved build. Membership comes from the
prepared run's ``partitions.csv`` (``layout="partitioned"``, namespaced IDs), and every partition folder
(``train``/``validation``/``test`` of the ordinary run, ``intervention`` of the ``_intervention`` run)
must account for exactly that partition's questions. Writes ``final_labels.jsonl`` and ``counts.json``
in the approved format. The team reference labels are keyed by the old plain IDs, so they never match
new episodes (they were never applied anyway).

Usage:
    PYTHONPATH=src python scripts/evaluation/build_final_labels_partitioned.py \\
        --partitions <prepared>/partitions.csv \\
        --runs <prepared>/results/<run> <prepared>/results/<run>_intervention \\
        --output <dir>
"""

import argparse
import json
import sys
from pathlib import Path

from mas_sae.data.production import read_partitions
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file
from mas_sae.experiments.records import read_jsonl

POLICY_DIR = Path(__file__).resolve().parents[2] / "results/experiments/candidate_final_labels_20261027"
RULE_FILES = ("final_labels.py", "proposal_snapshot.py")
RUN_FILES = ("interactions.jsonl", "exclusions.jsonl")


def load_rules(policy_dir: Path = POLICY_DIR):
    """The approved ``final_labels`` module, after checking its rule files against ``counts.json``."""
    approved = json.loads((policy_dir / "counts.json").read_text())["sources"]
    for name in RULE_FILES:
        key = next((k for k in approved if k.endswith("/" + name)), None)
        if key is None or sha256_file(policy_dir / name) != approved[key]:
            raise SystemExit(f"{policy_dir / name} is not the approved PR #153 rule file")
    if str(policy_dir) not in sys.path:
        sys.path.insert(0, str(policy_dir))
    import final_labels

    return final_labels


def run_partitions(run_dir: Path) -> list[str]:
    """Partition folders a partitioned run declares in ``run_layout.json``."""
    layout = json.loads((run_dir / "run_layout.json").read_text())
    if layout.get("layout") != "partitioned":
        raise SystemExit(f"{run_dir} is not a partitioned run")
    return layout["source_splits"]


def unique_rows(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    by_id = {row["question_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"{path}: a question appears more than once")
    return by_id


def build(rules, frozen: dict[str, dict], run_dirs: list[Path]) -> list[dict]:
    """One record per question of every partition folder, in ``partitions.csv`` order."""
    records, seen = [], set()
    for run_dir in run_dirs:
        for partition in run_partitions(run_dir):
            if partition in seen:
                raise SystemExit(f"partition {partition} appears in more than one run")
            seen.add(partition)
            folder = run_dir / partition
            expected = [q for q, r in frozen.items() if r["partition"] == partition]
            collected, failed = (unique_rows(folder / name) for name in RUN_FILES)
            if set(collected) & set(failed) or set(collected) | set(failed) != set(expected):
                raise SystemExit(f"{folder}: collected and failed IDs do not account for its {partition} questions")
            for qid in expected:
                if qid in collected:
                    record = rules.label_episode(collected[qid], partition)
                elif failed[qid].get("stage") in rules.COLLECTION_MODES:
                    record = rules.failure_record(failed[qid], partition)
                else:
                    raise SystemExit(f"{qid}: unknown collection failure stage {failed[qid].get('stage')!r}")
                records.append(dict(record, source_split=partition))
    missing = {r["partition"] for r in frozen.values()} - seen
    if missing:
        raise SystemExit(f"partitions.csv has questions in {sorted(missing)} but no run folder for them")
    return records


def input_hashes(partitions: Path, run_dirs: list[Path]) -> dict[str, str]:
    """sha256 of every file the build reads, keyed as ``counts.json`` records it."""
    files = {str(p): p for p in (*(POLICY_DIR / name for name in RULE_FILES), partitions)}
    for run in run_dirs:
        files[f"{run.name}/run_layout.json"] = run / "run_layout.json"
        files |= {f"{run.name}/{p}/{name}": run / p / name for p in run_partitions(run) for name in RUN_FILES}
    return {key: sha256_file(path) for key, path in files.items()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partitions", type=Path, required=True, help="the prepared run's partitions.csv")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="ordinary and intervention result runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    sources = input_hashes(args.partitions, args.runs)
    rules = load_rules()
    records = build(rules, read_partitions(args.partitions, layout="partitioned"), args.runs)
    if input_hashes(args.partitions, args.runs) != sources:
        raise RuntimeError("an input file changed during the build")
    args.output.mkdir(parents=True)
    (args.output / "final_labels.jsonl").write_text(rules.labels_jsonl(records), encoding="utf-8")
    manifest = dict(rules.counts(records), labels_sha256=rules.labels_sha256(records), sources=sources)
    (args.output / "counts.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["labels"]))


if __name__ == "__main__":
    main()
