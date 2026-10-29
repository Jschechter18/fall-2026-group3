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
