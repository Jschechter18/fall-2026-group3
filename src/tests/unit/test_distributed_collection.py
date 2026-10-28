import io
import json
from pathlib import Path

import pytest
import torch

from mas_sae.experiments.distributed_collection import (
    S3Transport, build_master_dataset, freeze_partitions, plan_chunks,
    write_work_packages,
    finalize_local_chunks,
)
from mas_sae.experiments import collection_progress
from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.records import read_jsonl
from mas_sae.experiments.artifacts import sha256_json


PROPORTIONS = {"train": .5, "validation": .2, "test": .2, "intervention": .1}


def inputs(tmp_path):
    musique = tmp_path/"m.jsonl"; synthetic = tmp_path/"s.jsonl"
    musique.write_text('\n'.join(json.dumps({"id": f"2hop__{i}_{i}", "question": "q", "paragraphs": [], "answer": "a"}) for i in range(4)))
    synthetic.write_text('\n'.join(json.dumps({"id": f"s{i}", "question": "q", "paragraphs": [], "answer": "a", "hop_group": "2hop",
                                                   "generation_provenance": {"generator": "fixture"}}) for i in range(4)))
    return [{"source": "musique", "version": "rev", "path": str(musique), "source_split": "train"},
            {"source": "synthetic", "version": "v1", "path": str(synthetic), "hop_field": "hop_group"}]


def test_master_namespaces_ids_preserves_payload_and_plans_no_intervention(tmp_path):
    master = build_master_dataset(inputs(tmp_path))
    assert master["questions"][0]["question_id"].startswith("musique:")
    assert master["questions"][-1]["generation_provenance"] == {"generator": "fixture"}
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=7)
    plan = plan_chunks(master, frozen, run_id="r", chunk_size=2, compatibility={"collection": {"protocol_version": "v2"}})
    assert all(c["partition"] != "intervention" for c in plan["chunks"])
    packages = tmp_path/"packages"; write_work_packages(packages, master, plan)
    package = json.loads(next(packages.iterdir()).read_text())
    assert package["examples"][0]["id"].count(":") == 1
    assert package["chunk"]["membership_sha256"]


def test_frozen_extension_never_reshuffles_and_seeds_ignore_chunk_size(tmp_path):
    specs = inputs(tmp_path); master = build_master_dataset(specs)
    first = freeze_partitions(master, proportions=PROPORTIONS, seed=9)
    old = {r["question_id"]: r for r in first["questions"]}
    with Path(specs[1]["path"]).open("a") as f:
        f.write('\n'+json.dumps({"id": "new", "question": "q", "paragraphs": [], "answer": "a", "hop_group": "3hop"}))
    extended = build_master_dataset(specs)
    second = freeze_partitions(extended, proportions=PROPORTIONS, seed=9, existing=old)
    now = {r["question_id"]: r for r in second["questions"]}
    assert all(now[q]["partition"] == r["partition"] and now[q]["collection_seed"] == r["collection_seed"] for q, r in old.items())
    p1 = plan_chunks(extended, second, run_id="r", chunk_size=1, compatibility={})
    p3 = plan_chunks(extended, second, run_id="r", chunk_size=3, compatibility={})
    seeds = lambda p: {q["question_id"]: q["collection_seed"] for c in p["chunks"] for q in c["questions"]}
    assert seeds(p1) == seeds(p3)


def test_musique_validation_policy_defaults_to_historical_behavior(tmp_path):
    path = tmp_path/"v.json"; path.write_text(json.dumps([{"id": "2hop__1_1", "question": "q", "paragraphs": [], "answer": "a"}]))
    master = build_master_dataset([{"source": "musique", "version": "r", "path": str(path), "source_split": "validation"}])
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=1)
    assert frozen["questions"][0]["partition"] == "validation"


class FakeS3:
    def __init__(self): self.objects = {}
    def upload_file(self, path, bucket, key): self.objects[(bucket,key)] = Path(path).read_bytes()
    def download_file(self, bucket, key, path): Path(path).write_bytes(self.objects[(bucket,key)])
    def head_object(self, Bucket, Key): return {"ContentLength": len(self.objects[(Bucket,Key)])}
    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **kwargs):
        if IfNoneMatch == "*" and (Bucket,Key) in self.objects: raise RuntimeError("precondition")
        self.objects[(Bucket,Key)] = Body
    def get_object(self, Bucket, Key): return {"Body": io.BytesIO(self.objects[(Bucket,Key)])}


def test_transport_completion_retry_and_conflict(tmp_path):
    chunk = tmp_path/"chunk"; chunk.mkdir()
    for name in ("metadata.json", "records.jsonl", "exclusions.jsonl", "activations.pt"): (chunk/name).write_bytes(name.encode())
    s3 = FakeS3(); transport = S3Transport("bucket", "prefix", s3)
    one = transport.upload_attempt("run", "c", chunk, "a1")
    assert transport.accept_attempt(one)["attempt_id"] == "a1"
    two = transport.upload_attempt("run", "c", chunk, "a2")
    assert transport.accept_attempt(two)["package_sha256"] == one["package_sha256"]
    (chunk/"records.jsonl").write_text("different")
    bad = transport.upload_attempt("run", "c", chunk, "a3")
    with pytest.raises(RuntimeError, match="conflicting"): transport.accept_attempt(bad)


def _finalize_fixture(tmp_path, name, groups):
    site = "model.layers.8"; chunks = []; locations = {}
    for number, ids in enumerate(groups):
        start = sum(len(g) for g in groups[:number]); chunk_id = f"train-{number:06d}"
        membership = [{"question_id": q, "collection_seed": i+10, "payload_sha256": q}
                      for i, q in enumerate(ids, start=start)]
        chunks.append({"chunk_id": chunk_id, "partition": "train", "partition_start": start,
                       "question_ids": ids, "membership_sha256": sha256_json(membership),
                       "questions": membership})
    basis = {"schema_version": 1, "run_id": "run", "master_sha256": "m",
             "partitions_sha256": "p", "compatibility": {}, "chunks": chunks}
    plan = {**basis, "plan_sha256": sha256_json(basis)}
    for chunk in chunks:
        result = {"records": [], "exclusions": [],
                  "attempt1_by_site": {site: []}, "attempt2_by_site": {site: []}}
        for q in chunk["question_ids"]:
            value = float(int(q[1:]))
            result["records"].append({"question_id": q, "source_split": "train",
                "solver_accepted_feedback": True, "attempt1_activation_index": len(result["records"]),
                "attempt2_activation_index": len(result["records"]), "seed": int(value)})
            result["attempt1_by_site"][site].append(torch.tensor([[value, value]]))
            result["attempt2_by_site"][site].append(torch.tensor([[value+.5, value+.5]]))
        progress = tmp_path/name/"source"/chunk["chunk_id"]; progress.mkdir(parents=True)
        written = collection_progress.write_chunk(progress, start=chunk["partition_start"],
            question_ids=chunk["question_ids"], candidate_sites=[site], result=result,
            metadata_extra={"chunk_id": chunk["chunk_id"], "run_id": "run",
                "plan_sha256": plan["plan_sha256"], "membership_sha256": chunk["membership_sha256"],
                "candidate_sites": [site]})
        locations[chunk["chunk_id"]] = written
    finalize_local_chunks(plan=plan, chunk_dirs=locations, candidate_sites=[site],
        activation_root=tmp_path/name/"activations", result_root=tmp_path/name/"results",
        run_name="final", resolved_config={})
    return (ActivationStore(tmp_path/name/"activations/final/layer_08"),
            read_jsonl(tmp_path/name/"results/final/train/interactions.jsonl"))


def test_serial_and_multichunk_finalization_are_exactly_equal(tmp_path):
    serial, serial_rows = _finalize_fixture(tmp_path, "serial", [["q0", "q1", "q2"]])
    multi, multi_rows = _finalize_fixture(tmp_path, "multi", [["q0"], ["q1", "q2"]])
    assert serial_rows == multi_rows
    for split in ("train_attempt1", "train_attempt2", "train"):
        assert torch.equal(serial.load_activations(split), multi.load_activations(split))
    assert [(r["attempt1_activation_index"], r["attempt2_activation_index"],
             r["sae_attempt1_index"], r["sae_attempt2_index"]) for r in multi_rows] \
        == [(0,0,0,3), (1,1,1,4), (2,2,2,5)]
