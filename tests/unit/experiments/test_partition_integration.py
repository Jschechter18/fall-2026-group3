"""Exercise the actual consumers of exported rows, tensors, labels, and contexts."""
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from mas_sae.causal import replay
from mas_sae.data import production
from mas_sae.experiments.distributed_collection import core
from mas_sae.experiments.records import read_jsonl
from mas_sae.probe import behavior_data as bd
from mas_sae.sae.dataloader import create_sae_dataloader
from test_distributed_collection import make_config, prepare, collect, fixture_collector


@pytest.fixture
def exported(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    path, manifest = prepare(tmp_path/"run", config, 7)
    excluded = [manifest["questions"][i]["question_id"] for i in (3, 43, 79)]
    monkeypatch.setattr(core, "collect_examples", fixture_collector(manifest, excluded=excluded))
    collect(path, manifest, config)
    core.merge_run(path)
    prepared = core.split_run(path, config)
    return config, path, manifest, prepared, excluded


def label_run(results, output, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prepare_behavior_v01.py", "--full-results", str(results),
        "--layout", "partitioned", "--label-policy", "v0.2.0-candidate", "--output", str(output)])
    runpy.run_path(str(Path(__file__).resolve().parents[3]/"scripts/evaluation/prepare_behavior_v01.py"), run_name="__main__")


def test_exported_tensors_and_all_index_fields_match_actual_questions(exported):
    config, path, master, prepared, excluded = exported
    numbers = {r["question_id"]: r["master_index"]+1 for r in master["questions"]}
    raw = read_jsonl(path.parent/"merged/results/raw/all/interactions.jsonl")
    raw_indices = {r["question_id"]: r["attempt2_activation_index"] for r in raw}
    any_reindex = False
    all_seen = set()
    for name, partitions in (("fixture", ["train", "validation", "test"]), ("fixture_intervention", ["intervention"])):
        results, acts = prepared/"results"/name, prepared/"activations"/name
        manifest = production.read_manifest(results/"split_manifest.csv")
        fixed = production.read_partitions(results/"partitions.csv", layout="partitioned")
        rows = production.read_run_interactions(results, layout="partitioned")
        for split in partitions:
            group = [r for r in rows if r["partition"] == split]
            assert group
            purpose = production.PARTITION_PURPOSE[split]
            for i, row in enumerate(group):
                qid = row["question_id"]
                assert row["source_split"] == split
                assert row["original_source_split"] == "train"
                assert manifest[qid]["full_index"] == row["attempt1_activation_index"] == row["attempt2_activation_index"] == i
                assert row["sae_attempt1_index"] == i
                assert row["sae_attempt2_index"] == len(group)+i
                assert row["collection_activation_index"] == raw_indices[qid]
                assert "source_activation_index" not in row
                any_reindex |= i != raw_indices[qid]
            for layer, site_offset in ((8, 0), (17, 1000)):
                for attempt, offset in ((1, 0), (2, 100)):
                    tensor = production.load_production_activations(group, manifest, acts, purpose=purpose, layer=layer, attempt=attempt)
                    assert tensor[:, 0].tolist() == [numbers[r["question_id"]]+site_offset+offset for r in group]
                combined = next(iter(create_sae_dataloader(200, split, 0, location=acts/f"layer_{layer:02d}")))
                values = [numbers[r["question_id"]]+site_offset for r in group]
                assert combined[:, 0].tolist() == values+[v+100 for v in values]
            all_seen.update(r["question_id"] for r in group)
        assert set(fixed) == set(manifest) | (set(excluded) & set(fixed))
        assert not set(excluded) & set(manifest)
        reserved = json.loads((results/"intervention_question_ids.json").read_text())
        assert set(reserved) == {r["question_id"] for r in master["questions"] if r["partition"] == "intervention"}
    assert any_reindex
    assert all_seen | set(excluded) == set(numbers)
    assert not list((prepared/"activations/fixture").glob("*/intervention*.pt"))
    assert not list((prepared/"activations/fixture_intervention").glob("*/train*.pt"))


def test_real_behavior_packages_probe_alignment_and_causal_inputs(exported, monkeypatch):
    _, path, master, prepared, _ = exported
    ordinary = prepared/"results/fixture"
    package = path.parent/"labels"
    label_run(ordinary, package, monkeypatch)
    rows, manifest = production.load_labeled_rows(ordinary, package, layout="partitioned")
    episodes = bd.build_episode_table(ordinary, ["train", "validation", "test"], ["solver_attempt_1"])
    labels, table = pd.read_csv(package/"labels.csv"), pd.read_csv(package/"split_manifest.csv")
    aligned, diag = bd.align_episodes(episodes, labels, table, bd.Columns(target="strict_target", eligible="eligible_strict"), "full_index")
    assert diag["index_verified_rows"] == len(rows)
    selected, _ = bd.select_partitions(aligned, bd.Columns(target="strict_target", eligible="eligible_strict"), ["train", "validation"])
    assert set(selected.partition) == {"train", "validation"}
    with pytest.raises(PermissionError):
        bd.select_partitions(aligned, bd.Columns(), ["test"])
    for partition in ("test", "validation"):
        with pytest.raises(ValueError, match="requires train"):
            production.authorize_rows([r for r in rows if r["partition"] == partition], manifest, purpose="fit")

    intervention = prepared/"results/fixture_intervention"
    intervention_package = path.parent/"intervention-labels"
    label_run(intervention, intervention_package, monkeypatch)
    cfg = replay.ReplayConfig(capstone_data_root=path.parent, interactions_jsonl=str(intervention/"intervention/interactions.jsonl"),
        labels_csv=str(intervention_package/"labels.csv"), contexts_jsonl=str(intervention/"contexts.jsonl"),
        label_policy="v0.2.0-candidate", feature_selection_summary="unused.json", layer=8)
    episodes = replay.load_episodes(cfg)
    paragraphs = replay.load_paragraphs(episodes, None, Path(cfg.contexts_jsonl))
    by_id = {r["question_id"]: r for r in master["questions"]}
    assert set(episodes.question_id) == set(paragraphs)
    assert any(q.startswith("synthetic:") for q in paragraphs)
    for qid, context in paragraphs.items():
        assert context == by_id[qid]["payload"]["paragraphs"]
    intervention_manifest = production.read_manifest(intervention/"split_manifest.csv")
    intervention_rows = production.read_run_interactions(intervention, layout="partitioned")
    with pytest.raises(ValueError, match="requires train"):
        production.authorize_rows(intervention_rows, intervention_manifest, purpose="fit")


def test_export_failure_never_publishes_partial_bundle(exported, monkeypatch):
    config, path, _, prepared, _ = exported
    import shutil
    shutil.rmtree(prepared)
    original = core.export_partition_activations
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("export interrupted")
        return original(*args, **kwargs)
    monkeypatch.setattr(core, "export_partition_activations", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        core.split_run(path, config)
    assert not prepared.exists()
    monkeypatch.setattr(core, "export_partition_activations", original)
    assert core.split_run(path, config) == prepared


def test_all_excluded_checkpoint_and_partition_are_reported(tmp_path, monkeypatch):
    config = make_config(tmp_path, count=8)
    path, master = prepare(tmp_path/"run", config, 4)
    ids = [r["question_id"] for r in master["questions"]]
    monkeypatch.setattr(core, "collect_examples", fixture_collector(master, excluded=ids))
    collect(path, master, config)
    core.merge_run(path)
    prepared = core.split_run(path, config)
    ordinary = prepared/"results/fixture"
    status = json.loads((ordinary/"collection_status.json").read_text())
    assert set(status["empty_partitions"]) == {"train", "validation", "test"}
    assert not status["collected"]
    assert not production.read_manifest(ordinary/"split_manifest.csv")
    with pytest.raises(SystemExit):
        label_run(ordinary, path.parent/"empty-labels", monkeypatch)


def test_incorrect_final_manifest_index_is_rejected_by_real_probe_loader(exported, monkeypatch):
    _, path, _, prepared, _ = exported
    results = prepared/"results/fixture"
    package = path.parent/"labels"
    label_run(results, package, monkeypatch)
    episodes = bd.build_episode_table(results, ["train", "validation"], [])
    manifest = pd.read_csv(package/"split_manifest.csv")
    manifest.loc[manifest.partition == "train", "full_index"] += 1
    with pytest.raises(bd.AlignmentError, match="disagrees"):
        bd.align_episodes(episodes, pd.read_csv(package/"labels.csv"), manifest, bd.Columns(), "full_index")
