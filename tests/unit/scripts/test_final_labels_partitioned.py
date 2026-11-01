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


def test_probe_package_from_the_partitioned_labels(build, prepared, tmp_path):
    """Distributed export -> PR #153 labels -> probe package -> align_episodes with index checks."""
    pd = pytest.importorskip("pandas")
    from mas_sae.probe import behavior_data as bd

    build.main(_args(prepared, tmp_path / "labels"))
    ordinary = prepared / "results/fixture"
    spec = importlib.util.spec_from_file_location("export_final_labels_for_probes",
                                                  SCRIPT.parent / "export_final_labels_for_probes.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    export.main(["--labels", str(tmp_path / "labels/final_labels.jsonl"), "--counts", str(tmp_path / "labels/counts.json"),
                 "--full-results", str(ordinary), "--layout", "partitioned", "--output", str(tmp_path / "pkg")])

    episodes = bd.build_episode_table(ordinary, ["train", "validation", "test"], ["solver_attempt_1"])
    table = pd.read_csv(tmp_path / "pkg/labels.csv")
    _, diag = bd.align_episodes(episodes, table, pd.read_csv(tmp_path / "pkg/split_manifest.csv"),
                                bd.Columns(), "full_index")
    assert diag["index_verified_rows"] == len(episodes)
    final = {r["question_id"]: r["label"] for r in read_jsonl(tmp_path / "labels/final_labels.jsonl")}
    assert all(final[q] == label for q, label in zip(table.question_id, table.behavior_label))


# ---- Claude labels as the training targets (#161)

CLASSES = ("adoption", "retention", "agreement", "out")


def _exporter():
    spec = importlib.util.spec_from_file_location("export_final_labels_for_probes",
                                                  SCRIPT.parent / "export_final_labels_for_probes.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    return export


def _claude_rows(ordinary, partition, technical=()):
    """One Claude result per collected episode, cycling the four classes; ``technical`` ids fail."""
    rows = []
    for i, episode in enumerate(read_jsonl(ordinary / partition / "interactions.jsonl")):
        label = None if episode["question_id"] in technical else CLASSES[i % 4]
        rows.append({**{k: episode[k] for k in ("episode_id", "question_id", "source_split",
                                                 "attempt1_activation_index", "attempt2_activation_index")},
                     "partition": partition, "behavior_label": label, "claude_label": label,
                     "claude_reason": "reason", "label_source": "claude", "claude_model": "claude-opus-5",
                     "prompt_version": "claude_binary_v3_candidates",
                     "eligible_primary": label in ("adoption", "retention"),
                     "primary_target": {"adoption": 1, "retention": 0}.get(label),
                     "exclusion_reason": "parse_error" if label is None else None})
    return rows


def _write_claude_run(folder, rows):
    from mas_sae.evaluation.claude_validator import export_binary_labels, export_exclusions
    export_binary_labels(rows, folder / "binary_labels.csv")
    export_exclusions(rows, folder / "exclusions.jsonl")
    return folder


def _claude_args(runs, ordinary, output):
    return ["--claude-runs", *map(str, runs), "--full-results", str(ordinary), "--layout", "partitioned",
            "--output", str(output)]


def test_claude_labels_package_aligns_and_leaves_held_out_partitions_unlabeled(prepared, tmp_path):
    pd = pytest.importorskip("pandas")
    from mas_sae.probe import behavior_data as bd

    ordinary = prepared / "results/fixture"
    failed = read_jsonl(ordinary / "train/interactions.jsonl")[0]["question_id"]
    rows = {p: _claude_rows(ordinary, p, technical={failed}) for p in ("train", "validation")}
    runs = [_write_claude_run(tmp_path / f"claude/{p}", r) for p, r in rows.items()]
    _exporter().main(_claude_args(runs, ordinary, tmp_path / "pkg"))

    table = pd.read_csv(tmp_path / "pkg/labels.csv").set_index("question_id")
    expected = {r["question_id"]: r["behavior_label"] for p in rows for r in rows[p] if r["behavior_label"]}
    assert table["behavior_label"].to_dict() == expected                         # all four classes kept
    assert set(table[table.eligible_primary].behavior_label) == {"adoption", "retention"}
    assert set(table.prompt_version) == {"claude_binary_v3_candidates"}
    excluded = json.loads((tmp_path / "pkg/excluded_episodes.json").read_text())
    assert excluded == {failed: "Claude technical failure: parse_error"}
    counts = json.loads((tmp_path / "pkg/counts.json").read_text())
    assert counts["unlabeled_partitions"] == ["test"] and counts["label_policy"]["name"] == "Claude labels"

    episodes = bd.build_episode_table(ordinary, ["train", "validation"], ["solver_attempt_1"])
    aligned, diag = bd.align_episodes(episodes, table.reset_index(), pd.read_csv(tmp_path / "pkg/split_manifest.csv"),
                                      bd.Columns(), "full_index", excluded=excluded)
    assert diag["index_verified_rows"] == len(episodes) - 1
    selected, _ = bd.select_partitions(aligned, bd.Columns(), ["train"])
    assert set(selected["target_name"]) == {"adopted_critic", "not_adopted"}


def _first(rows):
    return rows["train"][0]


@pytest.mark.parametrize("change, message", [
    (lambda rows: _first(rows).update(prompt_version="claude_binary_v2"), "share one prompt_version"),
    (lambda rows: rows["validation"].append(dict(_first(rows))), "more than one Claude run"),
    (lambda rows: rows["train"].pop(0), "but not this episode"),
    (lambda rows: _first(rows).update(partition="validation"), "Claude row has partition"),
    (lambda rows: _first(rows).update(attempt2_activation_index=_first(rows)["attempt2_activation_index"] + 1000),
     "activation index"),
    (lambda rows: rows["train"].append(dict(_first(rows), question_id="musique:not_collected")), "did not collect"),
    (lambda rows: rows["validation"][0].update(behavior_label="adopted", claude_label="adopted"),
     "unknown Claude label 'adopted'"),                                  # binary_labels.csv row
    (lambda rows: rows["validation"][2].update(behavior_label="agree", claude_label="agree"),
     "unknown Claude label 'agree'"),                                    # exclusions.jsonl row
])
def test_claude_labels_refuse_inconsistent_runs(prepared, tmp_path, change, message):
    ordinary = prepared / "results/fixture"
    rows = {p: _claude_rows(ordinary, p) for p in ("train", "validation")}
    change(rows)
    runs = [_write_claude_run(tmp_path / f"claude/{p}", r) for p, r in rows.items()]
    with pytest.raises(SystemExit, match=message):
        _exporter().main(_claude_args(runs, ordinary, tmp_path / "pkg"))
    assert not (tmp_path / "pkg").exists()


def test_label_sources_are_exclusive(prepared, tmp_path):
    ordinary = prepared / "results/fixture"
    run = _write_claude_run(tmp_path / "claude/train", _claude_rows(ordinary, "train"))
    for argv in (["--labels", "x.jsonl"], _claude_args([run], ordinary, tmp_path / "pkg")[:2] + ["--counts", "c.json"]):
        with pytest.raises(SystemExit):
            _exporter().main(argv + ["--full-results", str(ordinary), "--layout", "partitioned",
                                     "--output", str(tmp_path / "pkg")])
    assert not (tmp_path / "pkg").exists()
