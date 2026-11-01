"""PR #153 labels exported as a probe package: read by the real probe loader, never relabeled."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mas_sae.experiments.records import write_jsonl
from mas_sae.probe import behavior_data as bd

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export():
    return _load("export_final_labels_for_probes")


def _build_files(tmp_path, records, policy="candidate-final-2026-10-27"):
    labels = tmp_path / "build/final_labels.jsonl"
    write_jsonl(labels, records)
    counts = tmp_path / "build/counts.json"
    counts.write_text(json.dumps({"policy_version": policy,
                                  "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest()}))
    return labels, counts


def _record(qid, partition, label):
    return {"question_id": qid, "partition": partition, "collected": True, "label": label, "reason": "r",
            "step": "proposal", "previous_label": None, "flags": []}


@pytest.fixture
def legacy(tmp_path):
    """A two-folder candidates-style run, its frozen partitions and a matching PR #153 build."""
    run = tmp_path / "run"
    ids = {"train": ["2hop__1_2", "2hop__3_4", "2hop__5_6"], "validation": ["2hop__7_8"]}
    for split, qids in ids.items():
        write_jsonl(run / split / "interactions.jsonl", [
            {"question_id": q, "episode_id": f"{q}__natural__candidates", "source_split": split,
             "attempt1_activation_index": i, "attempt2_activation_index": i, "solver_attempt_1": "a",
             "solver_behavior": "adopted_critic"} for i, q in enumerate(qids)])
    partitions = tmp_path / "partitions.csv"  # 2hop__5_6 is collected but has no frozen partition
    partitions.write_text("question_id,source_split,partition,hop_group\n2hop__1_2,train,train,2hop\n"
                          "2hop__3_4,train,train,2hop\n2hop__7_8,validation,validation,2hop\n")
    records = [_record("2hop__1_2", "train", "adoption"), _record("2hop__3_4", "train", "retention"),
               _record("2hop__5_6", None, "out"), _record("2hop__7_8", "validation", "agreement")]
    return run, partitions, records


def _args(labels, counts, run, partitions, output):
    return ["--labels", str(labels), "--counts", str(counts), "--full-results", str(run),
            "--frozen-partitions", str(partitions), "--output", str(output)]


def test_legacy_package_keeps_labels_and_excludes_unpartitioned(export, legacy, tmp_path):
    run, partitions, records = legacy
    labels, counts = _build_files(tmp_path, records)
    export.main(_args(labels, counts, run, partitions, tmp_path / "pkg"))
    table = pd.read_csv(tmp_path / "pkg/labels.csv").set_index("question_id")
    assert table.loc["2hop__1_2", "primary_target"] == 1 and table.loc["2hop__3_4", "primary_target"] == 0
    assert not table.loc["2hop__7_8", "eligible_primary"] and table.loc["2hop__7_8", "behavior_label"] == "agreement"
    excluded = json.loads((tmp_path / "pkg/excluded_episodes.json").read_text())
    assert list(excluded) == ["2hop__5_6"] and "2hop__5_6" not in table.index

    episodes = bd.build_episode_table(run, ["train", "validation"], ["solver_attempt_1"])
    manifest = pd.read_csv(tmp_path / "pkg/split_manifest.csv")
    aligned, diag = bd.align_episodes(episodes, pd.read_csv(tmp_path / "pkg/labels.csv"), manifest,
                                      bd.Columns(), "full_index", excluded=excluded)
    assert diag["index_verified_rows"] == 3
    selected, _ = bd.select_partitions(aligned, bd.Columns(), ["train"])
    assert sorted(selected["target_name"]) == ["adopted_critic", "not_adopted"]


def test_refuses_a_changed_build_or_a_partition_mismatch(export, legacy, tmp_path):
    run, partitions, records = legacy
    labels, counts = _build_files(tmp_path, records)
    with labels.open("a") as fh:
        fh.write("\n")
    with pytest.raises(SystemExit, match="does not match labels_sha256"):
        export.main(_args(labels, counts, run, partitions, tmp_path / "a"))
    moved = [dict(r, partition="test") if r["question_id"] == "2hop__1_2" else r for r in records]
    labels, counts = _build_files(tmp_path, moved)
    with pytest.raises(SystemExit, match="partition"):
        export.main(_args(labels, counts, run, partitions, tmp_path / "b"))


@pytest.mark.parametrize("change", [{"label": "adoption"}, {"collected": False}, {"partition": "train"}, None])
def test_excludes_an_unpartitioned_episode_only_as_a_build_out(export, legacy, tmp_path, change):
    """2hop__5_6 has no frozen partition: a build that labels it otherwise, or not at all, is refused."""
    run, partitions, records = legacy
    kept = [r for r in records if r["question_id"] != "2hop__5_6"]
    if change is not None:
        kept.append(dict(next(r for r in records if r["question_id"] == "2hop__5_6"), **change))
    labels, counts = _build_files(tmp_path, kept)
    with pytest.raises(SystemExit, match="2hop__5_6: has no frozen partition"):
        export.main(_args(labels, counts, run, partitions, tmp_path / "pkg"))
    assert not (tmp_path / "pkg").exists()
