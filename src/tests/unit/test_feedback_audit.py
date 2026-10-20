import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mas_sae.data.production import write_manifest
from mas_sae.evaluation.behavior_labels import V012_QC, classify_candidate, write_labels
from mas_sae.evaluation import feedback_audit
from mas_sae.evaluation.feedback_audit import (
    audit_counts,
    audit_row,
    candidate_status,
    choose_review_rows,
    possible_missed_negative,
    human_comparison,
    wilson_interval,
)

REPO = Path(__file__).resolve().parents[3]


def review(verdict, advocated, explanation):
    return json.dumps({"verdict": verdict, "advocated_answer": advocated, "explanation": explanation})


def interaction(qid, a1, advocated, a2, explanation, verdict="disagree", source_split="train"):
    raw = review(verdict, advocated, explanation)
    return {
        "question_id": qid, "episode_id": f"{qid}__natural", "source_split": source_split,
        "question": "Q?", "solver_attempt_1": a1, "solver_attempt_2": a2,
        "critic_raw_output": raw, "critic_verdict": verdict, "critic_advocated_answer": advocated,
        "critic_noncommittal": False, "critic_blind_answer": advocated, "solver_behavior": "ambiguous",
        "critic_feedback": f"The reviewer disagrees. {explanation}",
        "solver_attempt_1_correct": True, "solver_attempt_2_correct": True,
        "critic_feedback_correct": False, "validator_final_correct": True,
        "generation_telemetry": {"critic_review": [{"reached_token_budget": False}],
                                 "solver_a1": {"reached_token_budget": False},
                                 "solver_a2": {"reached_token_budget": False}},
    }


def labelled(row, partition="train"):
    return {**row, "partition": partition, "label": classify_candidate(row, V012_QC)}


COHERENT_RETAIN = interaction("2hop__1_1", "Fulda", "Rhine", "Fulda",
                              "Kassel lies on the Rhine. Therefore, the correct answer is Rhine.")
CONTRADICTORY_RETAIN = interaction("2hop__2_2", "1999", "2000", "1999",
                                   "The proposed answer is correct. Therefore, the correct answer is 1999.")
ADOPT = interaction("2hop__3_3", "Paris", "Rome", "Rome", "The text names Rome. Therefore, Rome.")


def test_candidate_status_separates_coherent_from_contradictory_negatives():
    assert audit_row(labelled(COHERENT_RETAIN))["candidate_status"] == "rule_pass"
    assert audit_row(labelled(CONTRADICTORY_RETAIN))["candidate_status"] == "invalid"
    assert candidate_status({"eligible_primary": False}, "consistent", False) == ""
    assert candidate_status({"eligible_primary": True}, "consistent", True) == "uncertain"


def test_correctness_is_recorded_but_never_changes_status():
    flipped = {**COHERENT_RETAIN, "solver_attempt_1_correct": False, "critic_feedback_correct": True,
               "solver_attempt_2_correct": False, "validator_final_correct": False}
    assert audit_row(labelled(flipped))["candidate_status"] == audit_row(labelled(COHERENT_RETAIN))["candidate_status"]


def test_counts_and_human_comparison():
    records = [audit_row(labelled(r)) for r in (COHERENT_RETAIN, CONTRADICTORY_RETAIN, ADOPT)]
    counts = audit_counts(records)
    assert counts["strict_negatives"]["eligible_retained_a1"] == 2
    assert counts["strict_negatives"]["rule_pass"] == 1
    assert counts["strict_negatives"]["invalid"] == 1
    human = {"2hop__2_2": {"eligible_primary": "no", "contradiction_noted": True},
             "2hop__1_1": {"eligible_primary": "yes", "contradiction_noted": False}}
    comparison = human_comparison(records, human)
    assert comparison["reviewed_rows"] == 2
    assert comparison["noted_called_inconsistent"]["count"] == 1
    assert comparison["human_eligible_by_candidate_status"] == {
        "human=no / rules=invalid": 1, "human=yes / rules=rule_pass": 1}


def test_wilson_interval():
    assert wilson_interval(0, 0) == (None, None)
    lo, hi = wilson_interval(2, 111)
    assert 0 < lo < 2 / 111 < hi < 0.1


def test_review_rows_take_every_unreviewed_negative_and_skip_reviewed():
    records = [audit_row(labelled(r)) for r in (COHERENT_RETAIN, CONTRADICTORY_RETAIN, ADOPT)]
    chosen, key, sampling = choose_review_rows(records, {"2hop__2_2"}, per_group=5, seed=1)
    ids = {r["question_id"] for r in chosen}
    assert ids == {"2hop__1_1", "2hop__3_3"}
    assert all(entry["review_id"].startswith("F") for entry in key)
    validation = [audit_row(labelled(COHERENT_RETAIN, partition="validation"))]
    with pytest.raises(ValueError):
        choose_review_rows(validation, set(), per_group=5, seed=1)


def _write_run(tmp_path, rows_by_partition):
    results = tmp_path / "results"
    labels_dir = tmp_path / "release"
    labels_dir.mkdir()
    all_rows, manifest = [], {}
    for partition, rows in rows_by_partition.items():
        for row in rows:
            row = labelled(row, partition)
            all_rows.append(row)
            manifest[row["question_id"]] = {
                "question_id": row["question_id"], "source_split": row["source_split"],
                "partition": partition, "hop_group": "2hop", "collection_split": "",
                "scan_split": "", "in_qc100": False, "in_development_data": False,
                "full_index": len(manifest), "scan_index": None}
    for split in ("train", "validation"):
        (results / split).mkdir(parents=True)
        with open(results / split / "interactions.jsonl", "w") as f:
            for row in all_rows:
                if row["source_split"] == split:
                    clean = {k: v for k, v in row.items() if k not in ("partition", "label")}
                    f.write(json.dumps(clean) + "\n")
    write_labels(labels_dir / "labels.csv", all_rows)
    write_manifest(labels_dir / "split_manifest.csv", manifest)
    return results, labels_dir


def _run(*args):
    return subprocess.run([sys.executable, str(REPO / "scripts/audit_feedback_quality.py"), *args],
                          capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src")})


def test_script_audits_one_partition_and_refuses_unfrozen_validation(tmp_path):
    test_row = interaction("2hop__9_9", "A", "B", "A", "Therefore, B.")
    results, labels_dir = _write_run(tmp_path, {"train": [COHERENT_RETAIN, CONTRADICTORY_RETAIN, ADOPT],
                                                "test": [test_row]})
    out = tmp_path / "audit"
    done = _run("--results", str(results), "--labels", str(labels_dir),
                "--partition", "train", "--output", str(out))
    assert done.returncode == 0, done.stderr
    with open(out / "audit.csv") as f:
        audited = {r["question_id"] for r in csv.DictReader(f)}
    # The test question shares the train file: it is parsed for its id, then dropped.
    assert audited == {"2hop__1_1", "2hop__2_2", "2hop__3_3"}
    counts = json.loads((out / "counts.json").read_text())
    assert counts["audit"]["strict_negatives"]["rule_pass"] == 1

    refused = _run("--results", str(results), "--labels", str(labels_dir),
                   "--partition", "validation", "--output", str(tmp_path / "val"))
    assert refused.returncode != 0
    assert "expect-procedure-sha256" in refused.stderr

    frozen = _run("--results", str(results), "--labels", str(labels_dir), "--partition", "validation",
                  "--output", str(tmp_path / "val"), "--expect-procedure-sha256", "0" * 64)
    assert frozen.returncode != 0
    assert "audit procedure hash is" in frozen.stderr


def test_train_run_never_opens_validation(tmp_path):
    results, labels_dir = _write_run(tmp_path, {"train": [COHERENT_RETAIN, ADOPT]})
    # Unreadable validation content must not matter to a train audit.
    (results / "validation" / "interactions.jsonl").write_text("{not json\n")
    out = tmp_path / "audit"
    done = _run("--results", str(results), "--labels", str(labels_dir),
                "--partition", "train", "--output", str(out))
    assert done.returncode == 0, done.stderr
    hashed = json.loads((out / "counts.json").read_text())["source_files_sha256"]
    assert not any("validation" in path for path in hashed)


def test_procedure_hash_covers_rule_code(tmp_path, monkeypatch):
    copies = []
    for path in feedback_audit.procedure_files():
        copy = tmp_path / path.name
        copy.write_bytes(path.read_bytes())
        copies.append(copy)
    monkeypatch.setattr(feedback_audit, "procedure_files", lambda: copies)
    before = feedback_audit.procedure_sha256()
    names = {p.name for p in copies}
    assert {"feedback_coherence.py", "feedback_audit.py", "critic.py", "behavior.py", "scoring.py"} <= names
    for copy in copies:
        original = copy.read_bytes()
        copy.write_bytes(original + b"\n# changed\n")
        assert feedback_audit.procedure_sha256() != before, copy.name
        copy.write_bytes(original)
    assert feedback_audit.procedure_sha256() == before
    monkeypatch.setattr(feedback_audit, "COMPOUND_ANSWER_WORDS", 11)
    assert feedback_audit.procedure_sha256() != before


def test_non_answer_packet_samples_only_non_answer_labels():
    refusal = interaction("2hop__5_5", "Paris", "The passage does not say", "Paris",
                          "The text does not name a city.")
    records = [audit_row(labelled(r)) for r in (COHERENT_RETAIN, ADOPT, refusal)]
    assert records[2]["feedback_type"] in feedback_audit.NON_ANSWER_FEEDBACK
    chosen, key, sampling = choose_review_rows(records, set(), per_group=5, seed=1, classes="non_answers")
    assert [r["question_id"] for r in chosen] == ["2hop__5_5"]
    assert key[0]["review_id"] == "G0001"
    assert key[0]["stratum"].startswith("label_")
    assert choose_review_rows(records, {"2hop__5_5"}, per_group=5, seed=1, classes="non_answers")[0] == []
    with pytest.raises(ValueError):
        choose_review_rows(records, set(), per_group=5, seed=1, classes="everything")


def test_unsettled_excluded_conflicts_are_sampled_as_possible_missed_negatives():
    # A1 "Paris" inside the advocated "Paris, Texas": excluded as containment,
    # yet the Solver may have rejected a genuinely different answer.
    row = interaction("2hop__4_4", "Paris", "Paris, Texas", "Paris",
                      "The text places it in Texas. Therefore, the correct answer is Paris, Texas.")
    record = audit_row(labelled(row))
    assert not record["eligible_primary"]
    assert possible_missed_negative(record)
    chosen, key, sampling = choose_review_rows([record], set(), per_group=5, seed=1)
    assert [r["question_id"] for r in chosen] == ["2hop__4_4"]
    assert key[0]["stratum"].startswith("excluded_containment")
    assert "possible_missed_negatives" in audit_counts([record])
