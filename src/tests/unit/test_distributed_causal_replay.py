"""Map-reduce causal replay manifests, workers, resume, and strict reduction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from mas_sae.causal import distributed, replay
from mas_sae.causal.smoke import smoke_components, tiny_features, tiny_saes
from mas_sae.experiments.artifacts import file_facts


def _preflight(tmp_path, n=5):
    root = tmp_path / "preflight"
    root.mkdir()
    features = {"Fixture SAE": {"targeted": [1], "random": [[2]]}}
    conditions = replay.conditions(features, error_preserving=True)
    config = {
        "capstone_data_root": ".",
        "interactions_jsonl": "interactions.jsonl",
        "labels_csv": "labels.csv",
        "feature_selection_summary": "features.json",
        "layer": 1,
        "n_random_draws": 1,
        "error_preserving": True,
        "output_dir": "unused",
    }
    provenance = {key: f"fixture-{key}" for key in replay.IDENTITY_KEYS}
    (root / "replay_config.json").write_text(json.dumps({
        "config": config, "conditions": conditions, "features": features, "provenance": provenance,
    }))
    question_ids = [f"q{i}" for i in range(n)]
    pd.DataFrame({"question_id": question_ids, "stored_solver_response": [replay.ACCEPT] * n}).to_csv(
        root / "episodes.csv", index=False,
    )
    baseline = [{"question_id": qid, "experiment": replay.BASELINE, "condition": "no_sae", "draw": None,
                 "mode": None, "strength": None, "fraction": None, "edit_norm": None,
                 "first_token_id": 10 + i, "input_ids_sha256": f"prompt-{qid}",
                 "solver_attempt_2": "Lyon", "stored_solver_attempt_2": "Lyon",
                 "stored_solver_response": replay.ACCEPT, "solver_response": replay.ACCEPT,
                 "adopted": 1, "prompt_matches_stored": True} for i, qid in enumerate(question_ids)]
    (root / "per_row.jsonl").write_text("".join(json.dumps(row) + "\n" for row in baseline))
    (root / "replay_check.json").write_text(json.dumps({
        "passed": True, "failures": [], "n_episodes": n, "n_planned": n,
    }))
    return root, question_ids, features, conditions, provenance


def _fake_workers(monkeypatch, question_ids, features, provenance):
    episodes = pd.DataFrame({"question_id": question_ids, "question": [f"Question {q}" for q in question_ids]})
    monkeypatch.setattr(replay, "load_episodes", lambda cfg: episodes.copy())
    monkeypatch.setattr(replay, "load_components", lambda cfg, eps: SimpleNamespace(features=features, info={}))
    monkeypatch.setattr(replay, "provenance", lambda cfg, config_path, comp: provenance)

    def generate(todo, comp, output, first, what, max_hours=None):
        with output.open("a") as file:
            for episode, condition in todo:
                control = replay.control_key(condition)
                assert control is None or (episode["question_id"], *control) in first
                row = {
                    "question_id": episode["question_id"],
                    "experiment": condition["experiment"],
                    "condition": condition["condition"],
                    "draw": condition["draw"],
                    "mode": condition.get("mode"),
                    "strength": condition.get("strength"),
                    "fraction": condition.get("fraction"),
                    "edit_norm": condition.get("edit_norm"),
                    "first_token_id": (10 + int(episode["question_id"][1:])
                                       if condition["experiment"] == replay.BASELINE else 100 + len(first)),
                    "input_ids_sha256": f"prompt-{episode['question_id']}",
                    "solver_attempt_2": "Lyon",
                    "stored_solver_attempt_2": "Lyon",
                    "stored_solver_response": replay.ACCEPT,
                    "solver_response": replay.ACCEPT,
                    "adopted": 1,
                    "prompt_matches_stored": True,
                }
                first[replay.key(row)] = row["first_token_id"]
                file.write(json.dumps(row) + "\n")

    monkeypatch.setattr(replay, "generate_all", generate)


def test_prepare_map_reduce_has_exact_balanced_coverage(tmp_path, monkeypatch):
    preflight, question_ids, features, conditions, provenance = _preflight(tmp_path)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=2)
    manifest = distributed.read_manifest(manifest_path)
    assert [len(shard["question_ids"]) for shard in manifest["shards"]] == [3, 2]
    assert [q for shard in manifest["shards"] for q in shard["question_ids"]] == question_ids

    _fake_workers(monkeypatch, question_ids, features, provenance)
    for shard_id in range(2):
        root = distributed.run_shard(manifest_path, shard_id)
        assert (root / "complete.json").is_file()
        assert distributed.run_shard(manifest_path, shard_id, resume=True) == root

    def analyse(run_dir):
        rows = replay.read_rows(run_dir / "per_row.jsonl")
        (run_dir / "analysis.json").write_text(json.dumps({"rows": len(rows)}))

    monkeypatch.setattr(replay, "analyse", analyse)
    merged = distributed.reduce_shards(manifest_path)
    rows = replay.read_rows(merged / "per_row.jsonl")
    assert len(rows) == len(question_ids) * len(conditions)
    assert len({replay.key(row) for row in rows}) == len(rows)
    assert [row["question_id"] for row in rows[::len(conditions)]] == question_ids
    assert json.loads((merged / "analysis.json").read_text())["rows"] == len(rows)


def test_worker_resume_repairs_tail_and_does_not_repeat_rows(tmp_path, monkeypatch):
    preflight, question_ids, features, conditions, provenance = _preflight(tmp_path, n=2)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=1)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    root = distributed.shard_directory(manifest_path, 0)

    original = replay.generate_all
    calls = 0

    def interrupted(todo, comp, output, first, what, max_hours=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            original(todo[:2], comp, output, first, what, max_hours)
            with output.open("a") as file:
                file.write('{"unfinished"')
            raise RuntimeError("interrupted")
        original(todo, comp, output, first, what, max_hours)

    monkeypatch.setattr(replay, "generate_all", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        distributed.run_shard(manifest_path, 0)
    committed = replay.read_rows(root / "per_row.jsonl")
    assert len(committed) == 2

    monkeypatch.setattr(replay, "generate_all", original)
    distributed.run_shard(manifest_path, 0, resume=True)
    rows = replay.read_rows(root / "per_row.jsonl")
    assert len(rows) == len(question_ids) * len(conditions)
    assert len({replay.key(row) for row in rows}) == len(rows)


def test_worker_stops_before_interventions_when_its_baseline_differs(tmp_path, monkeypatch):
    preflight, question_ids, features, conditions, provenance = _preflight(tmp_path, n=2)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=1)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    generate = replay.generate_all

    def mismatched_baseline(todo, comp, output, first, what, max_hours=None):
        generate(todo, comp, output, first, what, max_hours)
        if what.endswith("baseline"):
            rows = replay.read_rows(output)
            rows[0]["first_token_id"] += 1
            output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(replay, "generate_all", mismatched_baseline)
    with pytest.raises(ValueError, match="baseline differs"):
        distributed.run_shard(manifest_path, 0)
    rows = replay.read_rows(distributed.shard_directory(manifest_path, 0) / "per_row.jsonl")
    assert len(rows) == len(question_ids)
    assert {row["experiment"] for row in rows} == {replay.BASELINE}
    assert len(conditions) > 1


def test_worker_baseline_parity_includes_the_full_answer(tmp_path, monkeypatch):
    preflight, question_ids, features, _, provenance = _preflight(tmp_path, n=2)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=1)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    generate = replay.generate_all

    def changed_answer(todo, comp, output, first, what, max_hours=None):
        generate(todo, comp, output, first, what, max_hours)
        if what.endswith("baseline"):
            rows = replay.read_rows(output)
            rows[0]["solver_attempt_2"] = "Lyon, France"
            output.write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(replay, "generate_all", changed_answer)
    with pytest.raises(ValueError, match="solver_attempt_2"):
        distributed.run_shard(manifest_path, 0)
    rows = replay.read_rows(distributed.shard_directory(manifest_path, 0) / "per_row.jsonl")
    assert len(rows) == len(question_ids)
    assert {row["experiment"] for row in rows} == {replay.BASELINE}


def test_worker_resume_initializes_an_empty_shard_directory(tmp_path, monkeypatch):
    preflight, question_ids, features, conditions, provenance = _preflight(tmp_path, n=2)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=1)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    root = distributed.shard_directory(manifest_path, 0)
    root.mkdir(parents=True)

    assert distributed.run_shard(manifest_path, 0, resume=True) == root
    assert (root / "worker.json").is_file()
    assert len(replay.read_rows(root / "per_row.jsonl")) == len(question_ids) * len(conditions)


def test_worker_resume_rejects_a_nonempty_shard_without_identity(tmp_path, monkeypatch):
    preflight, question_ids, features, _, provenance = _preflight(tmp_path, n=2)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=1)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    root = distributed.shard_directory(manifest_path, 0)
    root.mkdir(parents=True)
    (root / "per_row.jsonl").write_text("")

    with pytest.raises(ValueError, match="worker.json is missing from a nonempty"):
        distributed.run_shard(manifest_path, 0, resume=True)


def test_tiny_gemma_serial_and_two_shard_runs_are_equivalent(tmp_path, monkeypatch):
    for name in ("interactions.jsonl", "labels.csv", "features.json"):
        (tmp_path / name).write_text("fixture\n")
    cfg = replay.ReplayConfig(
        capstone_data_root=tmp_path,
        interactions_jsonl="interactions.jsonl",
        labels_csv="labels.csv",
        feature_selection_summary="features.json",
        layer=1,
        partition="validation",
        n_random_draws=1,
        n_bootstrap=5,
        n_permutations=10,
        min_replay_label_match=1,
        output_dir=tmp_path / "unused",
    )
    episodes = pd.DataFrame([
        {
            "question_id": f"q{i}",
            "question": f"Question {i}?",
            "solver_attempt_1": "Paris",
            "critic_advocated_answer": "Lyon",
            "critic_feedback": "The reviewer recommends Lyon.",
            "stored_solver_attempt_2": "",
            "stored_solver_response": "unresolved",
            "stored_adopted": 0,
            "a2_presentation": "feedback",
        }
        for i in range(3)
    ])

    def components(frame):
        saes = {"TopK SAE": tiny_saes()["TopK SAE"]}
        features = tiny_features(saes, n_random_draws=1)
        paragraphs = {
            question_id: [{"idx": 0, "title": "T", "paragraph_text": "text"}]
            for question_id in frame.question_id
        }
        return smoke_components(paragraphs, cfg.layer, saes, features, max_new_tokens=2)

    serial = replay.replay(cfg, episodes, components(episodes), tmp_path / "serial")
    replay.analyse(serial)
    preflight = replay.replay(
        cfg, episodes, components(episodes), tmp_path / "preflight", preflight_only=True,
    )
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=2)
    monkeypatch.setattr(replay, "load_episodes", lambda _: episodes.copy())
    monkeypatch.setattr(replay, "load_components", lambda _, frame: components(frame))
    for shard_id in range(2):
        distributed.run_shard(manifest_path, shard_id)
    merged = distributed.reduce_shards(manifest_path)

    def stable_rows(path):
        rows = [{key: value for key, value in row.items() if key != "seconds"}
                for row in replay.read_rows(path / "per_row.jsonl")]
        return sorted(rows, key=lambda row: tuple(str(value) for value in replay.key(row)))

    assert stable_rows(serial) == stable_rows(merged)
    for name in ("causal_table.csv", "targeted_vs_random.csv"):
        assert (serial / name).read_bytes() == (merged / name).read_bytes()


def test_prepare_and_reduce_fail_closed(tmp_path, monkeypatch):
    preflight, question_ids, features, _, provenance = _preflight(tmp_path)
    check_path = preflight / "replay_check.json"
    check = json.loads(check_path.read_text())
    check["passed"] = False
    check["failures"] = ["fixture"]
    check_path.write_text(json.dumps(check))
    with pytest.raises(ValueError, match="did not pass"):
        distributed.prepare_distributed_run(preflight, num_shards=2)

    check["passed"] = True
    check["failures"] = []
    check_path.write_text(json.dumps(check))
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=2)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    distributed.run_shard(manifest_path, 0)
    with pytest.raises(ValueError, match="Shard set mismatch"):
        distributed.reduce_shards(manifest_path)


def test_reduce_rejects_changed_completed_worker(tmp_path, monkeypatch):
    preflight, question_ids, features, _, provenance = _preflight(tmp_path, n=2)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=1)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    root = distributed.run_shard(manifest_path, 0)
    with (root / "per_row.jsonl").open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="files changed"):
        distributed.reduce_shards(manifest_path)


def test_reduce_rejects_condition_metadata_even_with_updated_checksums(tmp_path, monkeypatch):
    preflight, question_ids, features, _, provenance = _preflight(tmp_path, n=2)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=1)
    _fake_workers(monkeypatch, question_ids, features, provenance)
    root = distributed.run_shard(manifest_path, 0)
    rows = replay.read_rows(root / "per_row.jsonl")
    rows[0]["mode"] = "wrong"
    (root / "per_row.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    complete_path = root / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["files"] = file_facts(root, exclude_names={"complete.json"})
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="metadata differs"):
        distributed.reduce_shards(manifest_path)


def test_manifest_tampering_is_rejected(tmp_path):
    preflight, *_ = _preflight(tmp_path)
    manifest_path = distributed.prepare_distributed_run(preflight, num_shards=2)
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["question_ids"].reverse()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        distributed.read_manifest(manifest_path)
