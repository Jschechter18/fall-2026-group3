"""Validate saved candidates without making model calls."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from .deduplication import DuplicateIndex
from .validation import validate_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Candidate JSONL")
    parser.add_argument("--original", type=Path, nargs="+", required=True,
                        help="Original MuSiQue JSONL files; include every available split")
    parser.add_argument("--index", type=Path, required=True, help="Persistent SQLite index")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory (default: data/MuSiQue/synthetic/runs/<run-id>/validation)")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex}"
        args.output_dir = Path("data/MuSiQue/synthetic/runs") / run_id / "validation"
    
    paths = [args.input, *args.original]
    for path in paths:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if args.index.resolve() in {p.resolve() for p in paths}:
        parser.error("Index must not overwrite an input file")
    if args.output_dir.exists():
        parser.error("Output directory must be new to avoid overwriting prior results")
        
    args.output_dir.mkdir(parents=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    
    counts: Counter = Counter(accepted=0, rejected=0)
    reasons: Counter = Counter()
    
    with DuplicateIndex(args.index) as index:
        for original in args.original:
            index.import_jsonl(original, source=str(original.resolve()))
        with args.input.open(encoding="utf-8") as stream, \
                (args.output_dir / "accepted.jsonl").open("w", encoding="utf-8") as accepted, \
                (args.output_dir / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
            for line_number, line in enumerate(stream, 1):
                try:
                    candidate = json.loads(line)
                except ValueError as exc:
                    rejection = {"line": line_number, "raw": line.rstrip("\n"), "errors": [
                        {"code": "json", "field": "", "message": str(exc)}]}
                else:
                    validation = validate_candidate(candidate)
                    rejection = {"line": line_number, "candidate": candidate,
                                 "errors": [asdict(error) for error in validation.errors]}
                    if validation.valid:
                        duplicate = index.register_question(candidate["id"], candidate["question"],
                                                            source=str(args.input.resolve()))
                        if duplicate:
                            rejection["errors"] = [{"code": "duplicate", "field": "question",
                                                     "message": "Normalized question already exists."}]
                            rejection["duplicate"] = asdict(duplicate)
                        else:
                            accepted.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                            accepted.flush()
                            counts["accepted"] += 1
                            continue
                rejected.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                counts["rejected"] += 1
                reasons.update({error["code"] for error in rejection["errors"]})
    summary = {**counts, "output_dir": str(args.output_dir.resolve()),
               "rejections_by_code": dict(reasons),
               "quality_scope": "structural validation and normalized exact uniqueness only"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
