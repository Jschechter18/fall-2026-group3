"""Scientific identity, checkpoints, and serial/distributed equivalence without model downloads."""
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from mas_sae.evaluation.behavior import classify_behavior
from mas_sae.experiments import collection_progress
from mas_sae.experiments.artifacts import write_json_atomic
from mas_sae.experiments.collection_config import load_collection_config
from mas_sae.experiments.distributed_collection import core, dataset
from mas_sae.models.roles import resolve_roles

SITES = ["model.layers.8", "model.layers.17"]


def make_config(tmp_path, *, presentation="candidates", count=40):
    specs = []
    for source in ("musique", "synthetic"):
        rows = [{"id": f"2hop__{i}" if source == "musique" else f"q{i}",
                 "question": f"Question {source} {i}?", "answer": "Paris", "answer_aliases": [],
                 "paragraphs": [{"title": "Source", "paragraph_text": f"Exact context {source} {i}", "is_supporting": True, "idx": 0}],
                 "hop": "2hop"} for i in range(count)]
        # An explicit synthetic variant shares its original question's family.
        if source == "synthetic":
            rows[0]["split_group_id"] = "musique:2hop__0"
        path = tmp_path/f"{source}.json"
        path.write_text(json.dumps(rows))
        specs.append({"source": source, "version": "fixture", "source_split": "train",
                      "path": str(path), "hop_field": "hop"})
    return {"roles": {role: {"id": f"fixture/{role}", "loader": "qwen3", "dtype": "float32", "device": "cpu",
                  "generation": {"max_new_tokens": 8, "do_sample": False}} for role in ("solver", "critic", "validator")},
            "dataset": {"inputs": specs, "intervention": {"fraction": .2, "seed": 42},
                        "split": {"proportions": {"train": .6, "validation": .2, "test": .2}, "seed": 43}},
            "collection": {"protocol_version": "v2", "seed": 42, "layers": [8, 17],
                           "active_conditions": ["natural"], "a2_presentations": [presentation]},
            "output": {"run_name": "fixture", "chunk_size": 3}}


@pytest.mark.parametrize("question_id,metadata,expected", [
    ("2hop__synthetic_1", {}, "2hop"),
    ("3hop1__synthetic_1", {}, "3hop"),
    ("2hop__synthetic_1", {"hop_field": "hop"}, "4hop"),
    ("2hop__synthetic_1", {"hop": "3hop"}, "3hop"),
    ("q1", {}, None),
])
def test_synthetic_hop_metadata_falls_back_to_id(tmp_path, question_id, metadata, expected):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps([{"id": question_id, "question": "Where?",
                                 "answer": "Paris", "paragraphs": [], "hop": "4hop"}]))
    inputs = [{"source": "synthetic", "version": "fixture", "path": str(path), **metadata}]
    if expected is None:
        with pytest.raises(ValueError, match="explicit hop metadata is required"):
            dataset.build_master_dataset(inputs)
    else:
        assert dataset.build_master_dataset(inputs)["questions"][0]["hop_group"] == expected


def make_resolved(config, *, gpu="fixture"):
    return {**copy.deepcopy(config), "provenance": {
        "roles": {role: {"model_id": spec["id"], "requested_revision": None, "resolved_revision": None,
                  "loader": spec["loader"], "dtype": "torch.float32", "generation": spec["generation"],
                  "quantization": None, "chat_template_kwargs": {}} for role, spec in resolve_roles(config).items()},
        "protocol_version": "v2", "capabilities": {}, "active_conditions": ["natural"],
        "a2_presentations": config["collection"]["a2_presentations"], "prompt_sha256": {"solve": "a"*64},
        "environment": {"gpus": [gpu]}}}


def fixture_collector(manifest, *, excluded=()):
    numbers = {r["question_id"]: r["master_index"]+1 for r in manifest["questions"]}
    def collect(**kw):
        result = {"records": [], "exclusions": [],
                  "attempt1_by_site": {s: [] for s in SITES}, "attempt2_by_site": {s: [] for s in SITES}}
        for offset, example in enumerate(kw["examples"]):
            qid = example["id"]
            if qid in excluded:
                result["exclusions"].append({"question_id": qid, "seed": kw["question_seeds"][qid], "reason": "fixture failure"})
                continue
            number = numbers[qid]
            a2 = "Paris" if number % 2 else "London"
            presentation = kw["a2_presentations"][0]
            result["records"].append({"question_id": qid, "episode_id": qid+"__natural__"+presentation,
                "source_split": "all", "question": example["question"], "ground_truth": "Paris", "answer_aliases": [],
                "seed": kw["question_seeds"][qid], "critic_condition": "natural", "a2_presentation": presentation,
                "candidate_order": kw["candidate_orders"].get(qid), "solver_attempt_1": "London", "critic_advocated_answer": "Paris",
                "critic_feedback": "The reviewer disagrees. Paris", "solver_attempt_2": a2, "critic_noncommittal": False,
                "solver_accepted_feedback": a2 == "Paris", "solver_attempt_1_correct": False, "critic_feedback_correct": True,
                "candidate_first_token_ids": {"a1": 1, "critic": 2}, "decision_token_distinct": True, "solver_a2_first_token_id": 2,
                **classify_behavior("London", "Paris", a2, usable=True)})
            for i, site in enumerate(SITES):
                result["attempt1_by_site"][site].append(torch.tensor([[float(number+1000*i), 1.]]))
                result["attempt2_by_site"][site].append(torch.tensor([[float(number+1000*i+100), 2.]]))
        return result
    return collect


def prepare(tmp_path, config, count):
    manifest = dataset.prepare_manifest(config, num_chunks=count)
    path = tmp_path/"manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, manifest)
    return path, manifest


def collect(path, manifest, config, chunks=None, *, resume=False, gpu="fixture"):
    core.collect_manifest_chunks(path, manifest, dataset.select_chunks(manifest, chunks), config=config,
        resolved_config=make_resolved(config, gpu=gpu), model=None, solver=None, critic=None, validator=None,
        candidate_sites=SITES, type_checked_target=True, resume=resume)


def test_prepare_is_chunk_independent_and_does_not_assign_research_splits(tmp_path):
    config = make_config(tmp_path)
    a = dataset.prepare_manifest(config, num_chunks=1)
    b = dataset.prepare_manifest(config, num_chunks=7)
    assert a["questions"] == b["questions"]
    rows = {r["question_id"]: r for r in a["questions"]}
    assert {r["partition"] for r in rows.values()} == {None, "intervention"}
    assert rows["musique:2hop__0"]["partition"] == rows["synthetic:q0"]["partition"]
    assert rows["musique:2hop__0"]["payload"]["id"] == "2hop__0"
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_collection_config(path) == config
    config["dataset"]["source_split"] = "train"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="cannot be combined"):
        load_collection_config(path)


@pytest.mark.parametrize("changed", ["seed", "proportions"])
def test_export_rejects_split_changes_after_collection(tmp_path, monkeypatch, changed):
    config = make_config(tmp_path, count=10)
    path, manifest = prepare(tmp_path/"run", config, 2)
    monkeypatch.setattr(core, "collect_examples", fixture_collector(manifest))
    collect(path, manifest, config)
    core.merge_run(path)
    altered = copy.deepcopy(config)
    altered["dataset"]["split"][changed] = (99 if changed == "seed" else
        {"train": .8, "validation": .1, "test": .1})
    with pytest.raises(ValueError, match="Split settings differ"):
        core.split_run(path, altered)
    assert not (path.parent/"prepared").exists()
    core.split_run(path, config)
    assert json.loads((path.parent/"prepared/split.json").read_text())["split"] == manifest["split"]


def test_preparation_freezes_effective_split_settings(tmp_path):
    config = make_config(tmp_path)
    del config["dataset"]["split"]["seed"]
    manifest = dataset.prepare_manifest(config, num_chunks=2)
    assert manifest["split"]["seed"] == config["collection"]["seed"]
    # An explicit default has the same meaning as an omitted seed.
    config["dataset"]["split"]["seed"] = config["collection"]["seed"]
    core.check_config(manifest, config)
    config["dataset"]["split"]["proportions"]["train"] = .5
    assert manifest["split"]["proportions"]["train"] == .6
    assert dataset.prepare_manifest(config, num_chunks=2)["manifest_sha256"] != manifest["manifest_sha256"]
    del manifest["split"]
    with pytest.raises(ValueError, match="no frozen split settings; prepare a new run"):
        core.check_config(manifest, config)


def test_explicit_intervention_list_reserves_related_family(tmp_path):
    config = make_config(tmp_path)
    path = tmp_path/"ids.json"
    path.write_text(json.dumps(["synthetic:q0"]))
    config["dataset"]["intervention"] = {"question_ids_file": str(path)}
    manifest = dataset.prepare_manifest(config, num_chunks=2)
    assert {r["question_id"] for r in manifest["questions"] if r["partition"] == "intervention"} == {"synthetic:q0", "musique:2hop__0"}


def test_serial_and_distributed_merge_are_exactly_equal(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    outputs = []
    for name, count in (("serial", 1), ("distributed", 7)):
        path, manifest = prepare(tmp_path/name, config, count)
        excluded = [manifest["questions"][i]["question_id"] for i in (0, 3, 40, 79)]
        monkeypatch.setattr(core, "collect_examples", fixture_collector(manifest, excluded=excluded))
        # Reverse worker arrival order, and vary operational GPU metadata.
        for chunk in reversed(manifest["chunks"]):
            collect(path, manifest, config, [chunk["chunk_id"]], gpu=str(chunk["chunk_id"]))
        merged = core.merge_run(path)
        outputs.append(merged)
        core.split_run(path, config)
        info = json.loads((merged/"collection.json").read_text())
        assert len(info["collected"])+len(info["excluded"]) == len(manifest["questions"])
        assert any(r["partition"] == "intervention" and r["question_id"] in info["collected"] for r in manifest["questions"])
    for site in ("layer_08", "layer_17"):
        for suffix in ("all", "all_attempt1", "all_attempt2"):
            tensors = [torch.load(p/"activations/raw"/site/f"{suffix}.pt", weights_only=True) for p in outputs]
            assert torch.equal(*tensors)
    assert (outputs[0]/"results/raw/all/interactions.jsonl").read_text() == (outputs[1]/"results/raw/all/interactions.jsonl").read_text()
    assert (outputs[0].parent/"prepared/partitions.csv").read_text() == (outputs[1].parent/"prepared/partitions.csv").read_text()


def test_resume_preserves_committed_batches_and_matches_complete_run(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    path, manifest = prepare(tmp_path/"run", config, 1)
    original = fixture_collector(manifest)
    calls = []
    def interrupted(**kw):
        calls.append([r["id"] for r in kw["examples"]])
        if len(calls) == 2:
            raise RuntimeError("instance interrupted")
        return original(**kw)
    monkeypatch.setattr(core, "collect_examples", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        collect(path, manifest, config)
    root = core.chunk_directory(path, manifest["chunks"][0])
    saved = (root/"chunk_00000000/activations.pt").read_bytes()
    assert collection_progress.resume_position(root, [r["question_id"] for r in manifest["questions"]]) == 3
    resumed = []
    def recording(**kw):
        resumed.extend(r["id"] for r in kw["examples"])
        return original(**kw)
    monkeypatch.setattr(core, "collect_examples", recording)
    collect(path, manifest, config, resume=True, gpu="different GPU")
    assert resumed[0] == manifest["questions"][3]["question_id"]
    assert saved == (root/"chunk_00000000/activations.pt").read_bytes()
    collect(path, manifest, config, resume=True)
    assert len(resumed) == len(manifest["questions"])-3
    core.merge_run(path)


@pytest.mark.parametrize("damage", ["missing", "extra", "changed-file", "settings", "seed", "order", "membership"])
def test_merge_rejects_bad_inputs(tmp_path, monkeypatch, damage):
    config = make_config(tmp_path, count=8)
    path, manifest = prepare(tmp_path/"run", config, 2)
    monkeypatch.setattr(core, "collect_examples", fixture_collector(manifest))
    collect(path, manifest, config)
    root = core.chunk_directory(path, manifest["chunks"][0])
    checkpoint = root/"chunk_00000000"
    if damage == "missing":
        (root/"complete.json").unlink()
    elif damage == "extra":
        (root.parent/"999999").mkdir()
    elif damage == "changed-file":
        with (checkpoint/"records.jsonl").open("a") as f:
            f.write("\n")
    else:
        if damage == "settings":
            p = root/"provenance.json"
            data = json.loads(p.read_text())
            data["resolved_config"]["provenance"]["prompt_sha256"]["solve"] = "b"*64
            p.write_text(json.dumps(data))
        elif damage == "membership":
            p = checkpoint/"metadata.json"
            data = json.loads(p.read_text())
            data["question_ids"].reverse()
            p.write_text(json.dumps(data))
        else:
            from mas_sae.experiments.records import read_jsonl, write_jsonl
            records = read_jsonl(checkpoint/"records.jsonl")
            if damage == "seed":
                records[0]["seed"] += 1
            else:
                records[0]["candidate_order"] = "invalid"
            write_jsonl(checkpoint/"records.jsonl", records)
        # Even coherent transport checksums cannot bypass semantic checks.
        complete = json.loads((root/"complete.json").read_text())
        complete["files"] = core._file_facts(root)
        (root/"complete.json").write_text(json.dumps(complete))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        core.merge_run(path)
    assert not (path.parent/"merged").exists()


def test_manifest_tampering_and_multiple_presentations_fail_before_collection(tmp_path):
    config = make_config(tmp_path)
    path, manifest = prepare(tmp_path/"run", config, 2)
    manifest["questions"][0]["payload"]["answer"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        dataset.read_manifest(path)
    config["collection"]["a2_presentations"] = ["feedback", "candidates"]
    with pytest.raises(ValueError, match="one condition"):
        dataset.prepare_manifest(config, num_chunks=2)


def test_collection_entrypoint_loads_models_once_for_all_assigned_chunks(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("collection_entry", Path(__file__).resolve().parents[3]/"scripts/collect_activations.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    config = make_config(tmp_path)
    path, manifest = prepare(tmp_path/"run", config, 3)
    monkeypatch.setattr(entry, "parse_args", lambda: SimpleNamespace(config="unused", manifest=path, chunk=None, resume=False))
    monkeypatch.setattr(entry, "load_collection_config", lambda _: config)
    loads = []
    def load(specs):
        loads.append(specs)
        return {role: (None, None) for role in specs}
    monkeypatch.setattr(entry, "load_role_models", load)
    monkeypatch.setattr(entry, "resolve_solver_sites", lambda *a: SITES)
    for role in ("Solver", "Critic", "Validator"):
        monkeypatch.setattr(entry, role, lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(entry, "configure_agent", lambda *a: None)
    monkeypatch.setattr(entry, "build_resolved_config", lambda *a, **kw: make_resolved(config))
    monkeypatch.setattr(core, "collect_examples", fixture_collector(manifest))
    entry.main()
    assert len(loads) == 1
    assert all((core.chunk_directory(path, c)/"complete.json").is_file() for c in manifest["chunks"])


def test_zero_intervention_fraction_keeps_every_question_in_ordinary_pool(tmp_path):
    config = make_config(tmp_path)
    config["dataset"]["intervention"]["fraction"] = 0
    manifest = dataset.prepare_manifest(config, num_chunks=2)
    assert all(r["partition"] is None for r in manifest["questions"])
