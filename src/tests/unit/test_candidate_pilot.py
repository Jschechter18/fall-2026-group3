"""Tests for the 250-episode candidates pilot selection (scripts/sample_candidate_pilot.py)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mas_sae.experiments.records import write_jsonl


def _load():
    path = Path(__file__).resolve().parents[3] / "scripts" / "sample_candidate_pilot.py"
    spec = importlib.util.spec_from_file_location("sample_candidate_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PILOT = _load()
FLAGS = ("letter_reply", "citation_only", "critic_mentions_a1", "potential_hybrid")


def _labels():
    rows, n = [], 0
    for label in ("adoption", "retention", "agreement", "out"):
        for _ in range(80):
            flags = [FLAGS[n % 4]] if n % 9 == 0 else []
            rows.append({"question_id": f"2hop__{n}", "partition": "train", "collected": True,
                         "label": label, "flags": flags, "team_label": None}); n += 1
    for i in range(46):  # team-reviewed references
        rows.append({"question_id": f"3hop1__team{i}", "partition": "train", "collected": True,
                     "label": "adoption", "flags": ["team_reviewed"], "team_label": "adoption"})
    rows += [{"question_id": "2hop__val", "partition": "validation", "collected": True, "label": "out", "flags": [], "team_label": None},
             {"question_id": "2hop__fail", "partition": "train", "collected": False, "label": "out", "flags": [], "team_label": None}]
    return rows


def test_selection_sizes_and_uniqueness():
    picked = PILOT.select_pilot(_labels(), seed=20261029)
    ev, ch = picked["evaluation"], picked["challenge"]
    ids = [r["question_id"] for r in ev + ch]
    assert len(ev) == 200 and len(ch) == 50 and len(set(ids)) == 250
    assert {l: sum(r["label"] == l for r in ev) for l in ("adoption", "retention", "agreement", "out")} == \
        {"adoption": 50, "retention": 50, "agreement": 50, "out": 50}
    assert not any("team_reviewed" in r["flags"] for r in ev)
    kinds = sorted(picked["challenge_kinds"].values())
    assert kinds.count("team_reviewed") == 46 and sorted(k for k in kinds if k != "team_reviewed") == sorted(f"difficult:{f}" for f in FLAGS)
    assert all(r["partition"] == "train" and r["collected"] for r in ev + ch)


def test_selection_is_deterministic_and_order_independent():
    a = PILOT.select_pilot(_labels(), seed=7)
    b = PILOT.select_pilot(list(reversed(_labels())), seed=7)
    assert [r["question_id"] for r in a["evaluation"] + a["challenge"]] == [r["question_id"] for r in b["evaluation"] + b["challenge"]]


def test_insufficient_class_raises():
    rows = [r for r in _labels() if r["label"] != "out" or r["question_id"].endswith(("1", "2"))]
    with pytest.raises(ValueError, match="'out'"):
        PILOT.select_pilot(rows, seed=1)


def test_refuses_unapproved_labels(tmp_path):
    labels = tmp_path / "final_labels.jsonl"
    write_jsonl(labels, _labels())
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps({"labels_sha256": "0" * 64, "policy_version": "x"}))
    with pytest.raises(SystemExit, match="not the approved"):
        PILOT.main(["--run", str(tmp_path), "--partitions", str(tmp_path / "p.csv"), "--labels", str(labels),
                    "--counts", str(counts), "--output", str(tmp_path / "out")])
    assert hashlib.sha256(labels.read_bytes()).hexdigest() != "0" * 64


# ---------------------------------------------------------------------------
# review fixes: frozen partitions are enforced, comparison is tied to the sampled file
# ---------------------------------------------------------------------------

def _write_partitions(path, rows):
    with path.open("w", newline="") as fh:
        fh.write("question_id,source_split,partition,hop_group\n")
        for qid, partition in rows:
            fh.write(f"{qid},train,{partition},2hop\n")


def _fixture(tmp_path, *, frozen_override=None):
    labels = _labels()
    frozen = [(r["question_id"], r["partition"]) for r in labels if r["partition"] is not None]
    if frozen_override:
        frozen = [(q, frozen_override.get(q, p)) for q, p in frozen]
    _write_partitions(tmp_path / "partitions.csv", frozen)
    labels_path = tmp_path / "final_labels.jsonl"
    write_jsonl(labels_path, labels)
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps({"labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                                  "policy_version": "fixture"}))
    run = tmp_path / "run"
    write_jsonl(run / "train" / "interactions.jsonl",
                [{"question_id": r["question_id"], "episode_id": r["question_id"] + "__natural__candidates",
                  "candidate_order": "a1_first"} for r in labels if r["collected"]])
    write_jsonl(run / "validation" / "interactions.jsonl", [])
    return ["--run", str(run), "--partitions", str(tmp_path / "partitions.csv"), "--labels", str(labels_path),
            "--counts", str(counts), "--output", str(tmp_path / "out")]


def test_frozen_partitions_file_is_validated(tmp_path):
    _write_partitions(tmp_path / "dup.csv", [("q1", "train"), ("q1", "test")])
    with pytest.raises(ValueError, match="more than once"):
        PILOT.read_frozen_partitions(tmp_path / "dup.csv")
    _write_partitions(tmp_path / "bad.csv", [("q1", "holdout")])
    with pytest.raises(ValueError, match="unknown partition"):
        PILOT.read_frozen_partitions(tmp_path / "bad.csv")


def test_labels_disagreeing_with_frozen_partitions_are_refused(tmp_path):
    argv = _fixture(tmp_path, frozen_override={"2hop__3": "test", "2hop__90": "intervention"})
    with pytest.raises(ValueError, match="disagree with the frozen partitions"):
        PILOT.main(argv)
    assert not (tmp_path / "out").exists()


def test_end_to_end_selection_is_frozen_train_and_hashed(tmp_path):
    argv = _fixture(tmp_path)
    PILOT.main(argv)
    out = tmp_path / "out"
    selection = json.loads((out / "selection.json").read_text())
    frozen = PILOT.read_frozen_partitions(tmp_path / "partitions.csv")
    assert len(selection["episodes"]) == 250
    assert all(frozen[e["question_id"]] == "train" for e in selection["episodes"])
    assert selection["sampled_episodes_sha256"] == hashlib.sha256((out / "sampled_episodes.jsonl").read_bytes()).hexdigest()


def _load_compare():
    path = Path(__file__).resolve().parents[3] / "scripts" / "compare_claude_pr153.py"
    spec = importlib.util.spec_from_file_location("compare_claude_pr153", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_refuses_runs_not_over_the_sampled_file(tmp_path):
    compare = _load_compare()
    sampled = tmp_path / "sampled_episodes.jsonl"
    sampled.write_text('{"episode_id": "e1"}\n')
    sha = hashlib.sha256(sampled.read_bytes()).hexdigest()
    selection = {"sampled_episodes_sha256": sha}
    compare.check_run_matches_selection({"episodes_sha256": sha, "episodes_path": str(sampled)}, selection)

    with pytest.raises(SystemExit, match="differs from the sampled"):
        compare.check_run_matches_selection({"episodes_sha256": "0" * 64, "episodes_path": str(sampled)}, selection)
    with pytest.raises(SystemExit, match="no sampled_episodes_sha256"):
        compare.check_run_matches_selection({"episodes_sha256": sha, "episodes_path": str(sampled)}, {})
    sampled.write_text('{"episode_id": "e1", "solver_attempt_2": "edited"}\n')
    with pytest.raises(SystemExit, match="has changed since it was sampled"):
        compare.check_run_matches_selection({"episodes_sha256": sha, "episodes_path": str(sampled)}, selection)
