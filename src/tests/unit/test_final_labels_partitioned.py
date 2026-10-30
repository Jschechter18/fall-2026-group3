"""The PR #153 rules on a partitioned (distributed) export: unchanged rules, complete accounting."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")  # the distributed-collection fixtures build activation tensors

from mas_sae.experiments.distributed_collection import core  # noqa: E402
from mas_sae.experiments.records import read_jsonl, write_jsonl  # noqa: E402
from test_distributed_collection import collect, fixture_collector, make_config, prepare  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/build_final_labels_partitioned.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_final_labels_partitioned", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build():
    return _load()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """A real prepare → collect → merge → split export, with the fields real collection rows carry."""
    config = make_config(tmp_path)
    path, manifest = prepare(tmp_path / "run", config, 3)
    collector = fixture_collector(manifest, excluded=[manifest["questions"][i]["question_id"] for i in (3, 43)])

    def collect_examples(**kw):
        result = collector(**kw)
        for record in result["records"]:
            record["critic_verdict"] = "disagree"
        for exclusion in result["exclusions"]:
            exclusion["stage"] = "critic_recommendation"
        return result

    monkeypatch.setattr(core, "collect_examples", collect_examples)
    collect(path, manifest, config)
    core.merge_run(path)
    return core.split_run(path, config)


def _args(prepared, output, runs=("fixture", "fixture_intervention")):
    return ["--partitions", str(prepared / "partitions.csv"),
            "--runs", *(str(prepared / "results" / run) for run in runs), "--output", str(output)]


def test_labels_every_question_with_the_unchanged_rules(build, prepared, tmp_path):
    build.main(_args(prepared, tmp_path / "labels"))
    rules = build.load_rules()
    labels = {r["question_id"]: r for r in read_jsonl(tmp_path / "labels/final_labels.jsonl")}
    frozen = {line.split(",")[0] for line in (prepared / "partitions.csv").read_text().splitlines()[1:]}
    assert set(labels) == frozen
    for run in ("fixture", "fixture_intervention"):
        for partition in build.run_partitions(prepared / "results" / run):
            folder = prepared / "results" / run / partition
            for row in read_jsonl(folder / "interactions.jsonl"):
                assert labels[row["question_id"]] == dict(rules.label_episode(row, partition), source_split=partition)
            for failure in read_jsonl(folder / "exclusions.jsonl"):
                assert labels[failure["question_id"]]["reason"] == "collection_failure"
    counts = json.loads((tmp_path / "labels/counts.json").read_text())
    assert sum(counts["labels"].values()) == len(frozen) and counts["policy_version"] == rules.POLICY_VERSION
    assert any(r["source_split"] == "intervention" for r in labels.values())


def test_refuses_incomplete_partitions_and_unknown_failures(build, prepared, tmp_path):
    with pytest.raises(SystemExit, match="no run folder"):
        build.main(_args(prepared, tmp_path / "a", runs=("fixture",)))
    ordinary = prepared / "results/fixture"
    failed = next(ordinary / p for p in build.run_partitions(ordinary)
                  if read_jsonl(ordinary / p / "exclusions.jsonl"))
    exclusions = read_jsonl(failed / "exclusions.jsonl")
    write_jsonl(failed / "exclusions.jsonl", [dict(exclusions[0], stage="worker_crash"), *exclusions[1:]])
    with pytest.raises(SystemExit, match="unknown collection failure stage"):
        build.main(_args(prepared, tmp_path / "b"))
    write_jsonl(failed / "exclusions.jsonl", exclusions[1:])
    with pytest.raises(SystemExit, match="do not account"):
        build.main(_args(prepared, tmp_path / "c"))


def test_sources_are_hashed_before_the_build_and_rechecked(build, prepared, tmp_path, monkeypatch):
    runs = [prepared / "results" / run for run in ("fixture", "fixture_intervention")]
    before = build.input_hashes(prepared / "partitions.csv", runs)
    assert {"fixture/run_layout.json", "fixture_intervention/run_layout.json"} <= set(before)
    build.main(_args(prepared, tmp_path / "labels"))
    assert json.loads((tmp_path / "labels/counts.json").read_text())["sources"] == before

    label = build.build
    interactions = runs[0] / build.run_partitions(runs[0])[0] / "interactions.jsonl"

    def label_then_edit(*args):  # a run file rewritten after it was read
        records = label(*args)
        interactions.write_text(interactions.read_text() + "\n")
        return records

    monkeypatch.setattr(build, "build", label_then_edit)
    with pytest.raises(RuntimeError, match="an input file changed during the build"):
        build.main(_args(prepared, tmp_path / "edited"))
    assert not (tmp_path / "edited").exists()


def test_refuses_changed_rule_files(build, tmp_path):
    for name in ("counts.json", *build.RULE_FILES):
        shutil.copy(build.POLICY_DIR / name, tmp_path / name)
    with (tmp_path / "final_labels.py").open("a") as fh:
        fh.write("\n# edited\n")
    with pytest.raises(SystemExit, match="not the approved PR #153 rule file"):
        build.load_rules(tmp_path)
