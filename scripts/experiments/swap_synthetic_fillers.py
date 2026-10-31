"""Swap only the filler paragraphs of existing synthetic records for related ones (#174 paired ablation).

Every record keeps its id, question, answer, aliases, decomposition, supporting paragraphs and generated
distractors byte-identical, at the same positions. Each filler paragraph is replaced in place by a
related pool paragraph (``ParagraphPool.related``). The generated distractors are identified from the
generation run's ``provenance.jsonl``, i.e. the titles in the response the record was built from.

Writes, in the given question order:
- ``accepted.jsonl``: the edited records, for ``dataset.path``;
- ``question_ids.json``: the order, for ``dataset.sampling.question_ids_file`` with ``strategy: listed``.
  Collecting in this order reproduces the original per-question seeds and candidate orders.
- ``swap_report.json``: input and output hashes, settings, and the per-record checks.

Refuses repeated ids, a record whose non-filler content changed, a filler that mentions a hop answer
or alias (after ``normalize_answer``) or names a supporting/distractor title, and inputs that change
while they are read.

Usage:
    PYTHONPATH=src python scripts/experiments/swap_synthetic_fillers.py \\
        --accepted <generation run>/accepted.jsonl --provenance <generation run>/provenance.jsonl \\
        --question-ids <pilot>/train/sampled_questions.json --pool data/MuSiQue/clean/train.json \\
        --seed 42 --output <new folder>
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from mas_sae.data.synthetic import validate_candidate
from mas_sae.data.synthetic.generation import ParagraphPool
from mas_sae.evaluation.scoring import normalize_answer
from mas_sae.experiments.artifacts import ensure_output_available, sha256_file, write_json_atomic
from mas_sae.experiments.records import read_jsonl, write_jsonl


def by_id(rows: list[dict], key: str, source: str) -> dict[str, dict]:
    """Index rows by ``key``, refusing repeats."""
    indexed = {}
    for row in rows:
        if row[key] in indexed:
            raise SystemExit(f"{source}: repeated {key} {row[key]!r}")
        indexed[row[key]] = row
    return indexed


def distractor_titles(provenance: dict[str, dict]) -> dict[str, set[str]]:
    """Generated distractor titles per record, from the response the record was built from."""
    titles = {}
    for record_id, row in provenance.items():
        built = [a for a in row["attempts"] if a["error"] is None and not a["precheck_errors"]]
        if not built:
            raise SystemExit(f"{record_id}: no successful attempt in the provenance")
        titles[record_id] = {d["title"] for d in json.loads(built[-1]["raw_text"]).get("distractors", [])}
    return titles


def read_question_ids(path: Path) -> list[str]:
    """A JSON list of ids, or of ``{"question_id": ...}`` objects (``sampled_questions.json``)."""
    ids = [q["question_id"] if isinstance(q, dict) else q for q in json.loads(path.read_text(encoding="utf-8"))]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: repeated question id")
    return ids


def swap_fillers(record: dict, distractors: set[str], pool: ParagraphPool, rng: random.Random) -> tuple[dict, list[int]]:
    """A copy of ``record`` with each filler replaced in place by a related pool paragraph."""
    supporting = [p for p in record["paragraphs"] if p["is_supporting"]]
    keep = {p["title"] for p in record["paragraphs"] if p["is_supporting"] or p["title"] in distractors}
    slots = [i for i, p in enumerate(record["paragraphs"]) if not p["is_supporting"] and p["title"] not in distractors]
    query = " ".join([record["question"], *(p["paragraph_text"] for p in supporting)])
    answers = [record["answer"], *record["answer_aliases"], *(s["answer"] for s in record["question_decomposition"])]
    fillers = pool.related(len(slots), rng, query=query, exclude_titles=keep, exclude_answers=answers)
    edited = json.loads(json.dumps(record))
    for slot, filler in zip(slots, fillers):
        edited["paragraphs"][slot] = {"idx": record["paragraphs"][slot]["idx"], "title": filler["title"],
                                      "paragraph_text": filler["paragraph_text"], "is_supporting": False}
    return edited, slots


def check(original: dict, edited: dict, slots: list[int], distractors: set[str]) -> None:
    """Refuse any change outside the filler slots, and any filler that leaks an answer or a name."""
    qid = original["id"]
    unchanged = {k: v for k, v in original.items() if k != "paragraphs"}
    if {k: v for k, v in edited.items() if k != "paragraphs"} != unchanged:
        raise SystemExit(f"{qid}: a field other than the filler paragraphs changed")
    for i, (old, new) in enumerate(zip(original["paragraphs"], edited["paragraphs"])):
        if i not in slots and old != new:
            raise SystemExit(f"{qid}: paragraph {i} is not a filler but changed")
    answers = {normalize_answer(a) for a in [original["answer"], *original["answer_aliases"],
                                             *(s["answer"] for s in original["question_decomposition"])]}
    names = {normalize_answer(p["title"]) for p in original["paragraphs"]
             if p["is_supporting"] or p["title"] in distractors}
    for i in slots:
        text = f" {normalize_answer(edited['paragraphs'][i]['paragraph_text'])} "
        leaked = sorted(n for n in answers | names if n and f" {n} " in text)
        if leaked:
            raise SystemExit(f"{qid}: filler {i} mentions {leaked}")
    report = validate_candidate(edited, expected_paragraphs=len(original["paragraphs"]))
    if not report.valid:
        raise SystemExit(f"{qid}: the edited record is invalid: {report.errors}")


def seed_for(seed: int, question_id: str) -> int:
    """A per-record seed from the run seed and the id, independent of the record's position."""
    return int(hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()[:16], 16)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True, help="the MuSiQue JSONL the generator drew fillers from")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_output_available(args.output)

    inputs = [args.accepted, args.provenance, args.question_ids, args.pool]
    hashes = {str(path): sha256_file(path) for path in inputs}
    records = by_id(read_jsonl(args.accepted), "id", str(args.accepted))
    provenance = by_id(read_jsonl(args.provenance), "record_id", str(args.provenance))
    question_ids = read_question_ids(args.question_ids)
    missing = [q for q in question_ids if q not in records or q not in provenance]
    if missing:
        raise SystemExit(f"{len(missing)} question id(s) have no record or provenance, e.g. {missing[:3]}")
    titles = distractor_titles({q: provenance[q] for q in question_ids})
    pool = ParagraphPool.from_jsonl(args.pool, filler="related")

    edited, checks = [], []
    for qid in question_ids:
        new, slots = swap_fillers(records[qid], titles[qid], pool, random.Random(seed_for(args.seed, qid)))
        check(records[qid], new, slots, titles[qid])
        edited.append(new)
        checks.append({"question_id": qid, "fillers_replaced": len(slots),
                       "generated_distractors": len(titles[qid])})
    if {str(path): sha256_file(path) for path in inputs} != hashes:
        raise RuntimeError("an input file changed while the records were built")

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "accepted.jsonl", edited)
    (args.output / "question_ids.json").write_text(json.dumps(question_ids, indent=2) + "\n", encoding="utf-8")
    write_json_atomic(args.output / "swap_report.json", {
        "inputs_sha256": hashes, "seed": args.seed, "filler": pool.filler, "related_top": pool.related_top,
        "records": len(edited), "fillers_replaced": sum(c["fillers_replaced"] for c in checks),
        "outputs_sha256": {name: sha256_file(args.output / name) for name in ("accepted.jsonl", "question_ids.json")},
        "per_record": checks,
    })
    print(f"{len(edited)} records, {sum(c['fillers_replaced'] for c in checks)} fillers replaced -> {args.output}")


if __name__ == "__main__":
    main()
