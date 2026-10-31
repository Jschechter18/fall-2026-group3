"""Paired filler ablation (#174): only filler paragraphs change, in place; everything else is identical."""
import importlib.util
import json
import random
from pathlib import Path

import pytest

from mas_sae.data.synthetic.generation import ParagraphPool, assemble_candidate
from test_synthetic_generation import RAW_2HOP, TOPICAL, UNRELATED

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/experiments/swap_synthetic_fillers.py"


@pytest.fixture
def swap():
    spec = importlib.util.spec_from_file_location("swap_synthetic_fillers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def run(tmp_path):
    """Two records built with random fillers, their provenance, a pool file and an id order."""
    records = [assemble_candidate(RAW_2HOP, "2hop", ParagraphPool(UNRELATED), random.Random(i)) for i in range(2)]
    provenance = [{"record_id": r["id"], "attempts": [
        {"error": "stop_reason=max_tokens", "precheck_errors": [], "raw_text": "{}"},
        {"error": None, "precheck_errors": [], "raw_text": json.dumps(RAW_2HOP)}]} for r in records]
    pool = write_jsonl(tmp_path / "pool.jsonl", [{"paragraphs": TOPICAL + UNRELATED}])
    order = [records[1]["id"], records[0]["id"]]
    (tmp_path / "ids.json").write_text(json.dumps([{"question_id": q} for q in order]))
    return {"accepted": write_jsonl(tmp_path / "accepted.jsonl", records),
            "provenance": write_jsonl(tmp_path / "provenance.jsonl", provenance),
            "pool": pool, "ids": tmp_path / "ids.json", "records": {r["id"]: r for r in records}, "order": order}


def args(run, output):
    return ["--accepted", str(run["accepted"]), "--provenance", str(run["provenance"]),
            "--question-ids", str(run["ids"]), "--pool", str(run["pool"]), "--seed", "42", "--output", str(output)]


def test_only_fillers_change_in_place_and_order_is_kept(swap, run, tmp_path):
    swap.main(args(run, tmp_path / "out"))
    edited = [json.loads(l) for l in open(tmp_path / "out/accepted.jsonl")]
    assert [r["id"] for r in edited] == run["order"]
    assert json.loads((tmp_path / "out/question_ids.json").read_text()) == run["order"]
    distractors = {d["title"] for d in RAW_2HOP["distractors"]}
    for new in edited:
        old = run["records"][new["id"]]
        assert {k: v for k, v in new.items() if k != "paragraphs"} == {k: v for k, v in old.items() if k != "paragraphs"}
        for before, after in zip(old["paragraphs"], new["paragraphs"]):
            if before["is_supporting"] or before["title"] in distractors:
                assert after == before
            else:
                assert after["idx"] == before["idx"] and after["title"].startswith("Observatory")
    report = json.loads((tmp_path / "out/swap_report.json").read_text())
    assert report["fillers_replaced"] == 2 * 16 and report["filler"] == "related" and len(report["inputs_sha256"]) == 4


def test_same_seed_gives_the_same_records(swap, run, tmp_path):
    swap.main(args(run, tmp_path / "a"))
    swap.main(args(run, tmp_path / "b"))
    assert (tmp_path / "a/accepted.jsonl").read_bytes() == (tmp_path / "b/accepted.jsonl").read_bytes()


def test_check_refuses_changed_evidence_and_leaking_fillers(swap, run):
    record = next(iter(run["records"].values()))
    distractors = {d["title"] for d in RAW_2HOP["distractors"]}
    slots = [i for i, p in enumerate(record["paragraphs"]) if not p["is_supporting"] and p["title"] not in distractors]
    supporting = next(i for i, p in enumerate(record["paragraphs"]) if p["is_supporting"])
    changed = json.loads(json.dumps(record))
    changed["paragraphs"][supporting]["paragraph_text"] += " Edited."
    with pytest.raises(SystemExit, match="is not a filler but changed"):
        swap.check(record, changed, slots, distractors)
    leaking = json.loads(json.dumps(record))
    leaking["paragraphs"][slots[0]]["paragraph_text"] = "The town of Selin is nearby."
    with pytest.raises(SystemExit, match="mentions"):
        swap.check(record, leaking, slots, distractors)
    renamed = dict(json.loads(json.dumps(record)), answer="1843")
    with pytest.raises(SystemExit, match="other than the filler"):
        swap.check(record, renamed, slots, distractors)


def test_refuses_repeated_or_unknown_ids(swap, run, tmp_path):
    rows = [json.loads(l) for l in open(run["accepted"])]
    write_jsonl(run["accepted"], rows + rows[:1])
    with pytest.raises(SystemExit, match="repeated id"):
        swap.main(args(run, tmp_path / "a"))
    write_jsonl(run["accepted"], rows)
    run["ids"].write_text(json.dumps(run["order"] + ["2hop__synthetic_missing"]))
    with pytest.raises(SystemExit, match="no record or provenance"):
        swap.main(args(run, tmp_path / "b"))
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
