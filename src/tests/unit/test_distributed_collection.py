import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

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
from mas_sae.experiments.provenance import experiment_compatibility
from mas_sae.experiments.distributed_collection import core, cli


PROPORTIONS = {"train": .5, "validation": .2, "test": .2, "intervention": .1}
SITE = "model.layers.8"


def resolved_config(*, hardware="cpu", max_tokens=32):
    roles = {}
    for role in ("solver", "critic", "validator"):
        roles[role] = {"model_id": f"fixture/{role}", "requested_revision": "main",
                       "resolved_revision": f"{role}-commit", "loader": "qwen3",
                       "architecture": ["Fixture"], "dtype": "torch.float32",
                       "quantization": None, "generation": {"do_sample": False, "max_new_tokens": max_tokens},
                       "chat_template_kwargs": {}, "device": hardware, "device_map": {"": hardware}}
    return {"provenance": {"roles": roles, "generation": {"solver_max_new_tokens": max_tokens},
            "prompt_versions": {"solver_solve": "V1"}, "prompt_sha256": {}, "protocol_version": "v2",
            "capabilities": {"blind_then_compare": True}, "active_conditions": ["natural"],
            "a2_presentations": ["feedback"], "environment": {"gpus": [hardware]}}}


RESOLVED = resolved_config()
COMPATIBILITY = experiment_compatibility(RESOLVED, [SITE])


def inputs(tmp_path):
    # These synthetic records are handcrafted fixtures, not a generated release.
    # Passing these tests does not validate a future producer's data contract.
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
    plan = plan_chunks(master, frozen, run_id="r", chunk_size=2, compatibility=COMPATIBILITY)
    assert all(c["partition"] != "intervention" for c in plan["chunks"])
    packages = tmp_path/"packages"; write_work_packages(packages, master, plan)
    package = json.loads(next(packages.iterdir()).read_text())
    assert package["examples"][0]["id"].count(":") == 0  # original payload is untouched
    chunk = next(c for c in package["plan"]["chunks"] if c["chunk_id"] == package["chunk_id"])
    assert chunk["question_ids"][0].count(":") == 1
    assert chunk["membership_sha256"]


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
    p1 = plan_chunks(extended, second, run_id="r", chunk_size=1, compatibility=COMPATIBILITY)
    p3 = plan_chunks(extended, second, run_id="r", chunk_size=3, compatibility=COMPATIBILITY)
    seeds = lambda p: {q["question_id"]: q["collection_seed"] for c in p["chunks"] for q in c["questions"]}
    assert seeds(p1) == seeds(p3)
    assert all(now[q]["candidate_order"] == r["candidate_order"] for q, r in old.items())


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
        if IfNoneMatch == "*" and (Bucket,Key) in self.objects:
            error = RuntimeError("precondition")
            error.response = {"Error": {"Code": "PreconditionFailed"},
                              "ResponseMetadata": {"HTTPStatusCode": 412}}
            raise error
        self.objects[(Bucket,Key)] = Body
    def get_object(self, Bucket, Key): return {"Body": io.BytesIO(self.objects[(Bucket,Key)])}


@pytest.mark.parametrize("status", [200, 403, 503])
def test_s3_reachability_accepts_http_responses(monkeypatch, status):
    client = SimpleNamespace(meta=SimpleNamespace(endpoint_url="https://s3.us-east-2.amazonaws.com"))

    def respond(request, timeout):
        assert request.get_method() == "HEAD"
        assert request.full_url == "https://s3.us-east-2.amazonaws.com/bucket"
        assert timeout == 5.0
        if status != 200:
            raise HTTPError(request.full_url, status, "response", {}, io.BytesIO())
        return io.BytesIO()

    monkeypatch.setattr(core, "urlopen", respond)
    S3Transport("bucket", "prefix", client).check_reachable()


@pytest.mark.parametrize("failure", [URLError("DNS lookup failed"), TimeoutError("timed out")])
def test_s3_reachability_reports_network_failure(monkeypatch, failure):
    client = SimpleNamespace(meta=SimpleNamespace(endpoint_url="https://s3.us-east-2.amazonaws.com"))

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(core, "urlopen", fail)
    with pytest.raises(ConnectionError, match="S3 endpoint is unreachable"):
        S3Transport("bucket", "prefix", client).check_reachable()


def test_cli_stops_before_upload_when_endpoint_is_unreachable(monkeypatch):
    class UnreachableTransport:
        def __init__(self, *args):
            pass

        def check_reachable(self):
            raise ConnectionError("S3 endpoint is unreachable")

        def upload_work_package(self, *args):
            pytest.fail("Upload must not start after failed preflight")

    monkeypatch.setattr("mas_sae.experiments.distributed_collection.S3Transport", UnreachableTransport)
    monkeypatch.setattr("sys.argv", ["distributed_collection.py", "upload-work",
        "--bucket", "bucket", "--prefix", "prefix", "--run-id", "run",
        "--chunk-id", "train-000000", "--package", "unused.json"])
    with pytest.raises(SystemExit, match="S3 endpoint is unreachable"):
        cli.main()


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


def _finalize_fixture(tmp_path, name, groups, *, candidate_orders=None, finalize=True):
    site = SITE; chunks = []; locations = {}
    for number, ids in enumerate(groups):
        start = sum(len(g) for g in groups[:number]); chunk_id = f"train-{number:06d}"
        membership = [{"question_id": q, "collection_seed": i+10, "payload_sha256": q}
                      for i, q in enumerate(ids, start=start)]
        if candidate_orders is not None:
            for row in membership:
                row["candidate_order"] = candidate_orders[row["question_id"]]
        chunks.append({"chunk_id": chunk_id, "partition": "train", "partition_start": start,
                       "question_ids": ids, "membership_sha256": sha256_json(membership),
                       "questions": membership})
    basis = {"schema_version": 1, "run_id": "run", "master_sha256": "m",
             "partitions_sha256": "p", "compatibility": COMPATIBILITY, "chunks": chunks}
    plan = {**basis, "plan_sha256": sha256_json(basis)}
    for chunk in chunks:
        result = {"records": [], "exclusions": [],
                  "attempt1_by_site": {site: []}, "attempt2_by_site": {site: []}}
        for q in chunk["question_ids"]:
            value = float(int(q[1:]))
            result["records"].append({"question_id": q, "source_split": "train",
                "solver_accepted_feedback": True, "attempt1_activation_index": len(result["records"]),
                "attempt2_activation_index": len(result["records"]), "seed": int(value)+10})
            if candidate_orders is not None:
                result["records"][-1].update(a2_presentation="candidates", candidate_order=candidate_orders[q])
            result["attempt1_by_site"][site].append(torch.tensor([[value, value]]))
            result["attempt2_by_site"][site].append(torch.tensor([[value+.5, value+.5]]))
        progress = tmp_path/name/"source"/chunk["chunk_id"]; progress.mkdir(parents=True)
        written = collection_progress.write_chunk(progress, start=chunk["partition_start"],
            question_ids=chunk["question_ids"], candidate_sites=[site], result=result,
            metadata_extra={"chunk_id": chunk["chunk_id"], "run_id": "run",
                "plan_sha256": plan["plan_sha256"], "membership_sha256": chunk["membership_sha256"],
                "candidate_sites": [site], "experiment_compatibility": COMPATIBILITY,
                "resolved_config": resolved_config(hardware=f"worker-{number}")})
        locations[chunk["chunk_id"]] = written
    if not finalize:
        return plan, locations, site
    finalize_local_chunks(plan=plan, chunk_dirs=locations, candidate_sites=[site],
        activation_root=tmp_path/name/"activations", result_root=tmp_path/name/"results",
        run_name="final", resolved_config=RESOLVED)
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


def test_finalization_rejects_empty_sites_before_writing(tmp_path):
    with pytest.raises(ValueError, match="candidate_sites must not be empty"):
        finalize_local_chunks(
            plan={"chunks": []}, chunk_dirs={}, candidate_sites=[],
            activation_root=tmp_path/"activations", result_root=tmp_path/"results",
            run_name="final", resolved_config={},
        )
    assert not (tmp_path/"activations").exists()
    assert not (tmp_path/"results").exists()


def test_planning_requires_compatibility_and_rejects_wrong_master(tmp_path):
    master = build_master_dataset(inputs(tmp_path))
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=7)
    with pytest.raises(ValueError, match="compatibility contract"):
        plan_chunks(master, frozen, run_id="r", chunk_size=2)
    frozen["master_sha256"] = "another dataset"
    with pytest.raises(ValueError, match="different master"):
        plan_chunks(master, frozen, run_id="r", chunk_size=2, compatibility=COMPATIBILITY)


@pytest.mark.parametrize("change", ["seed", "downloaded_file"])
def test_finalization_keeps_basic_data_checks(tmp_path, monkeypatch, change):
    original = collection_progress.write_chunk

    def changed_chunk(*args, **kwargs):
        root = original(*args, **kwargs)
        if change == "seed":
            path = root/"records.jsonl"
            rows = read_jsonl(path)
            rows[0]["seed"] = 999
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        else:
            receipt = {"run_id": "run", "chunk_id": kwargs["metadata_extra"]["chunk_id"],
                       "package_sha256": core.chunk_file_manifest(root)["package_sha256"]}
            (root/"download_receipt.json").write_text(json.dumps(receipt))
            with (root/"activations.pt").open('ab') as stream:
                stream.write(b'changed')
        return root

    monkeypatch.setattr(collection_progress, "write_chunk", changed_chunk)
    with pytest.raises(ValueError, match="seed differs|downloaded files changed"):
        _finalize_fixture(tmp_path, "changed", [["q0"]])


def test_worker_provenance_is_recorded_without_enforcing_versions(tmp_path, monkeypatch):
    original = collection_progress.write_chunk

    def add_provenance(*args, **kwargs):
        kwargs["metadata_extra"].update(provenance={"git_commit": "different", "package_versions": {"torch": "different"}})
        return original(*args, **kwargs)

    monkeypatch.setattr(collection_progress, "write_chunk", add_provenance)
    _finalize_fixture(tmp_path, "provenance", [["q0"]])
    workers = json.loads((tmp_path/"provenance/results/final/worker_provenance.json").read_text())["workers"]
    assert workers[0]["progress"]["git_commit"] == "different"


def test_compatibility_allows_hardware_but_rejects_generation_and_missing_identity():
    assert experiment_compatibility(resolved_config(hardware="cuda:1"), [SITE]) == COMPATIBILITY
    assert experiment_compatibility(resolved_config(max_tokens=64), [SITE]) != COMPATIBILITY
    broken = resolved_config()
    broken["provenance"]["roles"]["solver"]["resolved_revision"] = None
    with pytest.raises(ValueError, match="solver.*resolved_revision"):
        experiment_compatibility(broken, [SITE])


def test_work_package_payload_tampering_and_legacy_format_fail(tmp_path):
    master = build_master_dataset(inputs(tmp_path))
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=7)
    plan = plan_chunks(master, frozen, run_id="r", chunk_size=2, compatibility=COMPATIBILITY)
    write_work_packages(tmp_path/"packages", master, plan)
    package = json.loads(next((tmp_path/"packages").iterdir()).read_text())
    package["examples"][0]["question"] = "tampered"
    with pytest.raises(ValueError, match="payload SHA-256"):
        core.validate_work_package(package)
    with pytest.raises(ValueError, match="schema-v2"):
        core.validate_work_package({"schema_version": 1})


def test_work_package_upload_is_immutable_and_only_expected_conflict_is_idempotent(tmp_path):
    master = build_master_dataset(inputs(tmp_path))
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=7)
    plan = plan_chunks(master, frozen, run_id="r", chunk_size=2, compatibility=COMPATIBILITY)
    write_work_packages(tmp_path/"packages", master, plan)
    package_path = next((tmp_path/"packages").iterdir())
    package = json.loads(package_path.read_text())
    transport = S3Transport("bucket", "prefix", FakeS3())
    first = transport.upload_work_package("r", package["chunk_id"], package_path)
    assert transport.upload_work_package("r", package["chunk_id"], package_path) == first
    key = ("bucket", first["key"])
    transport.client.objects[key] = b"conflict"
    with pytest.raises(RuntimeError, match="conflicting immutable"):
        transport.upload_work_package("r", package["chunk_id"], package_path)


def test_upload_cli_existing_marker_prevents_attempt_upload(tmp_path, monkeypatch):
    marker = tmp_path/"marker.json"; marker.write_text("{}")
    class Transport:
        def __init__(self, *args): pass
        def check_reachable(self): pass
        def upload_attempt(self, *args): pytest.fail("upload must not start")
    monkeypatch.setattr("mas_sae.experiments.distributed_collection.S3Transport", Transport)
    monkeypatch.setattr("sys.argv", ["distributed_collection.py", "upload", "--bucket", "b", "--prefix", "p",
        "--run-id", "r", "--chunk-id", "c", "--chunk-dir", str(tmp_path), "--marker", str(marker)])
    with pytest.raises(FileExistsError):
        cli.main()


def test_accept_rejects_cli_identity_remote_marker_and_unrelated_s3_failure(tmp_path, monkeypatch):
    marker_path = tmp_path/"marker.json"
    marker_path.write_text(json.dumps({"run_id": "other", "chunk_id": "c"}))
    class Transport:
        def __init__(self, *args): pass
        def check_reachable(self): pass
        def accept_attempt(self, marker): pytest.fail("accept must not start")
    monkeypatch.setattr("mas_sae.experiments.distributed_collection.S3Transport", Transport)
    monkeypatch.setattr("sys.argv", ["distributed_collection.py", "accept", "--bucket", "b", "--prefix", "p",
        "--run-id", "r", "--chunk-id", "c", "--marker", str(marker_path)])
    with pytest.raises(ValueError, match="does not match"):
        cli.main()

    chunk = tmp_path/"chunk"; chunk.mkdir()
    for name in core.REQUIRED_CHUNK_FILES: (chunk/name).write_text(name)
    s3 = FakeS3(); transport = S3Transport("b", "p", s3)
    uploaded = transport.upload_attempt("r", "c", chunk, "a")
    changed = {**uploaded, "package_sha256": "wrong"}
    with pytest.raises(ValueError, match="remote attempt"):
        transport.accept_attempt(changed)
    class Denied(FakeS3):
        def put_object(self, *args, **kwargs):
            error = RuntimeError("access denied")
            error.response = {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}
            raise error
    denied = Denied(); denied.objects = s3.objects.copy()
    with pytest.raises(RuntimeError, match="access denied"):
        S3Transport("b", "p", denied).accept_attempt(uploaded)


def test_interrupted_finalization_requires_scoped_recovery(tmp_path, monkeypatch):
    plan, locations, site = _finalize_fixture(tmp_path, "recover", [["q0"]], finalize=False)
    activation_root = tmp_path/"recover/activations"; result_root = tmp_path/"recover/results"
    original = Path.replace
    def interrupt(path, target):
        if path.parent == result_root and path.name.startswith(".final.staging"):
            raise OSError("simulated interruption")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(OSError, match="simulated"):
        finalize_local_chunks(plan=plan, chunk_dirs=locations, candidate_sites=[site],
            activation_root=activation_root, result_root=result_root, run_name="final", resolved_config=RESOLVED)
    monkeypatch.setattr(Path, "replace", original)
    assert (activation_root/"final").is_dir() and not (result_root/"final").exists()
    finalize_local_chunks(plan=plan, chunk_dirs=locations, candidate_sites=[site],
        activation_root=activation_root, result_root=result_root, run_name="final",
        resolved_config=RESOLVED, recover_publication=True)
    assert (activation_root/"final/publication_complete.json").is_file()
    assert (result_root/"final/publication_complete.json").is_file()


def test_candidate_orders_are_frozen_balanced_and_independent_of_chunks(tmp_path):
    from collections import Counter
    master = build_master_dataset(inputs(tmp_path))
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=7)
    expected = {r["question_id"]: r["candidate_order"] for r in frozen["questions"]}
    groups = {}
    for row in frozen["questions"]:
        groups.setdefault((row["partition"], row["hop_group"]), []).append(row["candidate_order"])
    for orders in groups.values():
        counts = Counter(orders)
        assert abs(counts["a1_first"]-counts["critic_first"]) <= 1
    for size in (1, 3):
        plan = plan_chunks(master, frozen, run_id="r", chunk_size=size, compatibility=COMPATIBILITY)
        write_work_packages(tmp_path/f"packages-{size}", master, plan)
        for package_path in (tmp_path/f"packages-{size}").glob('*.json'):
            package = json.loads(package_path.read_text())
            chunk = next(c for c in package["plan"]["chunks"] if c["chunk_id"] == package["chunk_id"])
            for row in chunk["questions"]:
                assert row["candidate_order"] == expected[row["question_id"]]


def test_explicit_candidate_groups_support_namespaced_ids():
    from mas_sae.experiments.conditions import assign_candidate_orders
    ids = [f"synthetic:q{i}" for i in range(8)]
    groups = {qid: "2hop" if i < 4 else "3hop" for i, qid in enumerate(ids)}
    orders = assign_candidate_orders(ids, seed=7, hop_groups=groups)
    for group in ("2hop", "3hop"):
        selected = [orders[qid] for qid in ids if groups[qid] == group]
        assert selected.count("a1_first") == selected.count("critic_first") == 2
    with pytest.raises(ValueError, match="hop_groups is missing"):
        assign_candidate_orders(ids, seed=7, hop_groups={})


@pytest.mark.parametrize("missing_order", [False, True])
def test_candidates_worker_receives_frozen_orders_before_collection(tmp_path, monkeypatch, missing_order):
    import runpy
    from mas_sae.experiments import collection
    from mas_sae.agents.solver import Solver
    master = build_master_dataset(inputs(tmp_path))
    frozen = freeze_partitions(master, proportions=PROPORTIONS, seed=7)
    plan = plan_chunks(master, frozen, run_id="r", chunk_size=2, compatibility=COMPATIBILITY)
    write_work_packages(tmp_path/"packages", master, plan)
    package_path = next((tmp_path/"packages").glob("train-*.json"))
    package = json.loads(package_path.read_text())
    chunk = next(c for c in package["plan"]["chunks"] if c["chunk_id"] == package["chunk_id"])
    if missing_order:
        chunk["questions"][0].pop("candidate_order")
        package_path.write_text(json.dumps(package))
    output = tmp_path/"completed"
    config = {"model": {"id": "fixture"}, "dataset": {"source_split": "train", "num_questions": 2},
              "collection": {"layers": [8], "seed": 42, "active_conditions": ["natural"], "a2_presentations": ["candidates"]},
              "output": {"run_name": "fixture"}}
    worker = runpy.run_path(str(Path(__file__).resolve().parents[3]/"scripts/collect_activations.py"))["main"]

    class Agent:
        CANDIDATE_ORDERS = Solver.CANDIDATE_ORDERS
        def __init__(self, *args, **kwargs):
            pass

    def load_models(_):
        assert not missing_order, "Invalid orders should fail before loading models"
        return {r: (object(), object()) for r in ("solver", "critic", "validator")}

    replacements = {
        "parse_args": lambda: SimpleNamespace(config=tmp_path/"unused.yaml", work_package=package_path, chunk_output=output, resume=False),
        "load_collection_config": lambda _: config, "load_role_models": load_models,
        "resolve_solver_sites": lambda *args: ["model.layers.8"],
        "Solver": Agent, "Critic": Agent, "Validator": Agent, "configure_agent": lambda *args: None,
        "build_resolved_config": lambda *args, **kwargs: RESOLVED,
        "build_collection_progress_provenance": lambda *args, **kwargs: kwargs,
    }
    for key, value in replacements.items():
        monkeypatch.setitem(worker.__globals__, key, value)

    seen = {}
    def run_question(**kwargs):
        qid = kwargs["question_id"]
        seen[qid] = kwargs["candidate_order"]
        return {"attempt1_activations": {"model.layers.8": torch.ones(1, 2)}, "episodes": [{
            "record": {"question_id": qid, "episode_id": qid+"__natural__candidates", "seed": kwargs["seed"],
                       "critic_condition": "natural", "a2_presentation": "candidates",
                       "candidate_order": kwargs["candidate_order"], "solver_accepted_feedback": True},
            "attempt2_activations": {"model.layers.8": torch.ones(1, 2)},
        }]}
    monkeypatch.setattr(collection, "run_question", run_question)
    if missing_order:
        with pytest.raises(ValueError, match="frozen plan digest is invalid"):
            worker()
        assert not output.exists()
    else:
        worker()
        expected = {r["question_id"]: r["candidate_order"] for r in chunk["questions"]}
        assert seen == expected
        assert {r["question_id"]: r["candidate_order"] for r in read_jsonl(output/"records.jsonl")} == expected
        metadata = json.loads((output/"metadata.json").read_text())
        examples = [{**payload, "id": member["question_id"]} for payload, member in zip(package["examples"], chunk["questions"])]
        manifest = collection.describe_sampled_questions(examples, None, expected)
        assert metadata["provenance"]["manifest_sha256"] == collection_progress.manifest_sha256(manifest)


@pytest.mark.parametrize("wrong_order", [False, True])
def test_finalization_preserves_or_rejects_candidate_orders(tmp_path, monkeypatch, wrong_order):
    if wrong_order:
        original = collection_progress.write_chunk
        def changed(*args, **kwargs):
            kwargs["result"]["records"][0]["candidate_order"] = "critic_first"
            return original(*args, **kwargs)
        monkeypatch.setattr(collection_progress, "write_chunk", changed)
        with pytest.raises(ValueError, match="candidate order differs"):
            _finalize_fixture(tmp_path, "wrong", [["q0"]], candidate_orders={"q0": "a1_first"})
    else:
        _, rows = _finalize_fixture(tmp_path, "right", [["q0"], ["q1"]],
                                    candidate_orders={"q0": "a1_first", "q1": "critic_first"})
        assert [r["candidate_order"] for r in rows] == ["a1_first", "critic_first"]
