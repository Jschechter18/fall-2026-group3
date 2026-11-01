"""Tests for selecting a PR #153 label population for a full Claude pass."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mas_sae.experiments.records import write_jsonl


def _load():
    path = Path(__file__).resolve().parents[3] / "scripts" / "select_pr153_population.py"
    spec = importlib.util.spec_from_file_location("select_pr153_population", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POP = _load()


def _fixture(tmp_path):
    labels = ([{"question_id": f"2hop__{i}", "partition": "train", "collected": True, "label": "out",
                "reason": "E5", "flags": []} for i in range(6)]
              + [{"question_id": "2hop__a", "partition": "train", "collected": True, "label": "adoption", "flags": []},
                 {"question_id": "2hop__v", "partition": "validation", "collected": True, "label": "out", "flags": []},
                 {"question_id": "2hop__f", "partition": "train", "collected": False, "label": "out", "flags": []}])
    (tmp_path / "partitions.csv").write_text("question_id,source_split,partition,hop_group\n" + "".join(
        f"{r['question_id']},train,{r['partition']},2hop\n" for r in labels))
    write_jsonl(tmp_path / "final_labels.jsonl", labels)
    (tmp_path / "counts.json").write_text(json.dumps({
        "labels_sha256": hashlib.sha256((tmp_path / "final_labels.jsonl").read_bytes()).hexdigest(), "policy_version": "x"}))
    run = tmp_path / "run"
    write_jsonl(run / "train" / "interactions.jsonl",
                [{"question_id": r["question_id"], "episode_id": r["question_id"] + "__e"} for r in labels if r["collected"]])
    write_jsonl(run / "validation" / "interactions.jsonl", [])
    return ["--run", str(run), "--partitions", str(tmp_path / "partitions.csv"), "--labels", str(tmp_path / "final_labels.jsonl"),
            "--counts", str(tmp_path / "counts.json"), "--label", "out", "--partition", "train"]


def _prior_run(tmp_path, name, rows, version="claude_binary_v3_candidates"):
    run = tmp_path / name
    write_jsonl(run / "raw_results.jsonl", rows)
    (run / "summary.json").write_text(json.dumps({"prompt_version": version, "models_returned": ["m"], "episodes_sha256": "x"}))
    return run


def test_population_is_collected_frozen_partition_label_only(tmp_path):
    argv = _fixture(tmp_path)
    POP.main(argv + ["--output", str(tmp_path / "out")])
    sel = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert sel["population"]["size"] == 6 and sel["to_label"] == 6
    assert {e["question_id"] for e in sel["episodes"]} == {f"2hop__{i}" for i in range(6)}
    episodes = (tmp_path / "out" / "sampled_episodes.jsonl").read_bytes()
    assert sel["sampled_episodes_sha256"] == hashlib.sha256(episodes).hexdigest()
    assert b"label" not in episodes  # Claude's input carries no PR #153 labels


def test_already_labeled_episodes_are_skipped_but_errors_are_relabeled(tmp_path):
    argv = _fixture(tmp_path)
    prior = _prior_run(tmp_path, "prior", [{"episode_id": "2hop__0__e", "behavior_label": "adoption"},
                                           {"episode_id": "2hop__1__e", "behavior_label": None}])
    POP.main(argv + ["--already-labeled", str(prior), "--output", str(tmp_path / "out")])
    sel = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert sel["already_labeled_in_population"] == 1 and sel["to_label"] == 5
    written = [json.loads(l)["episode_id"] for l in (tmp_path / "out" / "sampled_episodes.jsonl").read_text().splitlines()]
    assert "2hop__0__e" not in written and "2hop__1__e" in written


def test_mixed_prompt_versions_in_prior_runs_refused(tmp_path):
    argv = _fixture(tmp_path)
    a = _prior_run(tmp_path, "a", [], version="claude_binary_v3_candidates")
    b = _prior_run(tmp_path, "b", [], version="claude_binary_v3")
    with pytest.raises(SystemExit, match="mix prompt versions"):
        POP.main(argv + ["--already-labeled", str(a), str(b), "--output", str(tmp_path / "out")])
