"""Paired filler ablation (#174): only filler paragraphs change, in place; everything else is identical."""
import importlib.util
import json
import random
from pathlib import Path

import pytest

from mas_sae.data.synthetic.generation import ParagraphPool, assemble_candidate
from test_related_fillers import TOPICAL, UNRELATED
from test_synthetic_generation import RAW_2HOP

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


def make_run(tmp_path, raws, pool_paragraphs=TOPICAL + UNRELATED):
    """One record per raw response, built with random fillers, plus provenance, a pool file and an id order."""
    records = [assemble_candidate(raw, "2hop", ParagraphPool(UNRELATED), random.Random(i)) for i, raw in enumerate(raws)]
    provenance = [{"record_id": r["id"], "attempts": [
        {"error": "stop_reason=max_tokens", "precheck_errors": [], "raw_text": "{}"},
        {"error": None, "precheck_errors": [], "raw_text": json.dumps(raw)}]} for r, raw in zip(records, raws)]
    pool = write_jsonl(tmp_path / "pool.jsonl", [{"paragraphs": pool_paragraphs}])
    order = [records[1]["id"], records[0]["id"]]
    (tmp_path / "ids.json").write_text(json.dumps([{"question_id": q} for q in order]))
    return {"accepted": write_jsonl(tmp_path / "accepted.jsonl", records),
            "provenance": write_jsonl(tmp_path / "provenance.jsonl", provenance),
            "pool": pool, "ids": tmp_path / "ids.json", "records": {r["id"]: r for r in records}, "order": order}


@pytest.fixture
def run(tmp_path):
    return make_run(tmp_path, [RAW_2HOP, RAW_2HOP])


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
    with pytest.raises(swap.BadQuestion, match="mentions"):
        swap.check(record, leaking, slots, distractors)
    renamed = dict(json.loads(json.dumps(record)), answer="1843")
    with pytest.raises(SystemExit, match="other than the filler"):
        swap.check(record, renamed, slots, distractors)


# Every topical (observatory) paragraph names this record's distractor title, as a college "in Halden"
# did for a question whose distractor is titled "Halden".
NAMED = dict(RAW_2HOP, distractors=[RAW_2HOP["distractors"][0], {
    "title": "Refracting Telescope", "paragraph_text": "Refracting Telescope is a fictional entity."}])


def test_fillers_that_name_the_questions_entities_are_never_drawn(swap, tmp_path):
    run = make_run(tmp_path, [RAW_2HOP, NAMED])
    swap.main(args(run, tmp_path / "out"))
    edited = {json.loads(l)["id"]: json.loads(l) for l in open(tmp_path / "out/accepted.jsonl")}
    assert list(edited) == run["order"]
    named = edited[run["order"][0]]
    fillers = [p for p in named["paragraphs"] if not p["is_supporting"] and p["title"] not in
               {"Selin River", "Refracting Telescope"}]
    assert fillers and all(p["title"].startswith("Album") for p in fillers)
    assert json.loads((tmp_path / "out/swap_report.json").read_text())["skipped"] == []


def test_a_question_without_enough_clean_fillers_is_skipped_and_listed(swap, tmp_path):
    run = make_run(tmp_path, [RAW_2HOP, NAMED], pool_paragraphs=TOPICAL)
    clean, bad = run["order"][1], run["order"][0]
    swap.main(args(run, tmp_path / "out"))
    assert [json.loads(l)["id"] for l in open(tmp_path / "out/accepted.jsonl")] == [clean]
    assert json.loads((tmp_path / "out/question_ids.json").read_text()) == [clean]
    report = json.loads((tmp_path / "out/swap_report.json").read_text())
    assert report["records"] == 1 and [c["question_id"] for c in report["per_record"]] == [clean]
    assert [s["question_id"] for s in report["skipped"]] == [bad]
    assert "Paragraph pool too small" in report["skipped"][0]["reason"]


def test_check_skips_a_filler_whose_title_alone_names_an_answer(swap, run):
    record = next(iter(run["records"].values()))
    distractors = {d["title"] for d in RAW_2HOP["distractors"]}
    slots = [i for i, p in enumerate(record["paragraphs"]) if not p["is_supporting"] and p["title"] not in distractors]
    titled = json.loads(json.dumps(record))
    titled["paragraphs"][slots[0]].update(title="Selin Archive", paragraph_text="A building with old maps.")
    with pytest.raises(swap.BadQuestion, match=r"mentions \['selin'\]"):
        swap.check(record, titled, slots, distractors)


def test_refuses_a_run_where_every_question_is_skipped(swap, tmp_path):
    run = make_run(tmp_path, [RAW_2HOP, RAW_2HOP], pool_paragraphs=TOPICAL[:5])
    with pytest.raises(SystemExit, match="every question was skipped.*Paragraph pool too small"):
        swap.main(args(run, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_dataset_flag_edits_every_record_when_no_ids_are_given(swap, run, tmp_path):
    swap.main(["--dataset", str(run["accepted"]), "--provenance", str(run["provenance"]),
               "--pool", str(run["pool"]), "--seed", "42", "--output", str(tmp_path / "out")])
    in_file = [json.loads(l)["id"] for l in open(run["accepted"])]
    assert [json.loads(l)["id"] for l in open(tmp_path / "out/accepted.jsonl")] == in_file
    assert json.loads((tmp_path / "out/question_ids.json").read_text()) == in_file
    assert len(json.loads((tmp_path / "out/swap_report.json").read_text())["inputs_sha256"]) == 3


def test_refuses_an_existing_output_so_the_dataset_is_never_overwritten(swap, run):
    before = run["accepted"].read_bytes()
    with pytest.raises(FileExistsError):
        swap.main(args(run, run["accepted"].parent))
    assert run["accepted"].read_bytes() == before


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
