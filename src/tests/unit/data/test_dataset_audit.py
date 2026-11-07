"""Dataset-level audit tests use fake catalog reviewers, never real API calls."""
import json
from pathlib import Path
import random
import threading
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from mas_sae.data.synthetic.batch import run_batch
from mas_sae.data.synthetic.dataset_audit import (
    _load_catalog, _make_batches, _neighbor_pairs, _prompt, audit_dataset,
)
from mas_sae.data.synthetic.generation import ParagraphPool, assemble_candidate
from mas_sae.data.synthetic.prompt_examples import build_example


def record(number):
    raw = build_example("2hop")
    raw.update(question=f"Who founded the label that released Album {number}?",
               answer=f"Mira {number}", distractors=[])
    raw["supporting_paragraphs"] = [
        {"title": f"Album {number}", "paragraph_text": f"Album {number} was released by Label {number}."},
        {"title": f"Label {number}", "paragraph_text": f"Mira {number} founded Label {number}."},
    ]
    raw["question_decomposition"] = [
        {"question": f"Which label released Album {number}?", "answer": f"Label {number}", "support": 0},
        {"question": "Who founded #1?", "answer": f"Mira {number}", "support": 1},
    ]
    result = assemble_candidate(raw, "2hop", ParagraphPool([]), random.Random(42), total_paragraphs=2)
    result["id"] = f"2hop__fixture_{number}"
    return result


def write_records(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def response_for(payload, groups=False):
    records = payload["questions"]
    return {"batch_id": payload["batch_id"], "reviewed_ids": [r["record_id"] for r in records],
            "groups": [{"relationship": "template_variant", "record_ids": [r["record_id"] for r in records],
                        "evidence": [{"record_id": r["record_id"], "quote": r["question"]} for r in records],
                        "reason": "Same question template with different named entities."}] if groups else [],
            "observations": []}


class Factory:
    def __init__(self, callback=None):
        self.callback = callback
        self.lock = threading.Lock()
        self.clients = []
        self.calls = 0
        self.prompts = []
        self.generation_calls = 0

    def __call__(self):
        factory = self
        class Client:
            model_id = "fake-model"
            closed = False
            phase = None
            def generate(self, prompt, **kwargs):
                if kwargs["task"] == "data-gen":
                    self.phase = "generation"
                    with factory.lock:
                        factory.generation_calls += 1
                        number = factory.generation_calls
                    raw = build_example(prompt.split("Hop subtype: ", 1)[1].splitlines()[0])
                    raw["question"] += f" Fixture {number}"
                else:
                    assert all(c.closed for c in factory.clients if c.phase == "generation")
                    self.phase = "audit"
                    payload = json.JSONDecoder().raw_decode(prompt.split("Catalog JSON:\n", 1)[1])[0]
                    with factory.lock:
                        factory.calls += 1
                        number = factory.calls
                        factory.prompts.append(prompt)
                    raw = factory.callback(number, payload) if factory.callback else response_for(payload, groups=True)
                return SimpleNamespace(text=raw if isinstance(raw, str) else json.dumps(raw), model="fake-returned",
                                       stop_reason="end_turn", input_tokens=10, output_tokens=20,
                                       sdk_retry_attempts=0)
            def close(self): self.closed = True
        client = Client()
        with self.lock: self.clients.append(client)
        return client


def test_whole_catalog_reviewed_in_groups_without_modifying_input(tmp_path):
    rows = [record(i) for i in range(12)]
    source = write_records(tmp_path / "accepted.jsonl", rows)
    original_bytes = source.read_bytes()
    factory = Factory()
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake-model", client_factory=factory,
                            concurrency=2, batch_size=6)
    assert summary["status"] == "completed"
    assert summary["coverage"]["local_records_scanned"] == 12
    assert summary["coverage"]["claude_catalog_records_reviewed"] == 12
    assert summary["statistics"]["exact_duplicate_groups"] == 0
    assert source.read_bytes() == original_bytes
    assert all(c.closed for c in factory.clients)
    assert not (tmp_path / "audit/approved.jsonl").exists()
    groups = read(tmp_path / "audit/groups.jsonl")
    assert groups and all(group["relationship"] == "template_variant" for group in groups)
    assert (tmp_path / "audit/report.md").exists()
    assert (tmp_path / "audit/statistics.json").exists()
    catalog = read(tmp_path / "audit/catalog.jsonl")
    assert {r["record_id"] for r in catalog} == {r["id"] for r in rows}
    assert "paragraphs" not in catalog[0]


def test_generation_automatically_audits_dataset_not_individual_questions(tmp_path):
    original = write_records(tmp_path / "original.jsonl", [record(99)])
    # Supply enough filler paragraphs for generation.
    value = json.loads(original.read_text())
    value["paragraphs"].extend({"title": f"Filler {i}", "paragraph_text": "Unrelated."} for i in range(40))
    original.write_text(json.dumps(value) + "\n")
    factory = Factory()
    output = tmp_path / "run"
    summary = run_batch(output, original=original, model_id="fake-model", n=4,
                        concurrency=2, client_factory=factory)
    assert factory.generation_calls == 4
    assert factory.calls == 1  # Four questions reviewed together, not four independent reviews.
    assert summary["dataset_audit"]["coverage"]["claude_catalog_records_reviewed"] == 4
    assert len(read(output / "accepted.jsonl")) == 4
    assert (output / "dataset_audit/report.md").exists()
    assert not (output / "semantic_validation").exists()
    assert "approved_total" not in summary


def test_exact_duplicates_and_templates_are_separate_reported_signals(tmp_path):
    rows = [record(1), record(2), record(3)]
    rows[1]["question"] = "  " + rows[0]["question"].upper() + "  "
    source = write_records(tmp_path / "accepted.jsonl", rows)
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=Factory())
    assert summary["statistics"]["exact_duplicate_groups"] == 1
    assert summary["statistics"]["exact_duplicate_excess_records"] == 1
    assert read(tmp_path / "audit/exact_duplicates.jsonl")[0]["record_ids"] == [rows[0]["id"], rows[1]["id"]]
    assert summary["statistics"]["repeated_template_groups"] >= 1
    assert len(read(source)) == 3  # Report-only, no automatic deletion.


@pytest.mark.parametrize("mistake", ["id", "missing_id", "unknown_group_id", "fake_quote", "missing_evidence", "relationship"])
def test_malformed_model_findings_get_bounded_format_retry(tmp_path, mistake):
    source = write_records(tmp_path / "accepted.jsonl", [record(1), record(2)])
    def callback(number, payload):
        raw = response_for(payload, groups=True)
        if number > 1: return raw
        if mistake == "id": raw["batch_id"] = "different"
        elif mistake == "missing_id": raw["reviewed_ids"].pop()
        elif mistake == "unknown_group_id": raw["groups"][0]["record_ids"][0] = "invented-id"
        elif mistake == "fake_quote": raw["groups"][0]["evidence"][0]["quote"] = "Invented excerpt"
        elif mistake == "missing_evidence": raw["groups"][0]["evidence"].pop()
        elif mistake == "relationship": raw["groups"][0]["relationship"] = "approved"
        return raw
    factory = Factory(callback)
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory)
    assert summary["status"] == "completed" and summary["api_calls"] == 2
    assert "Format error:" in factory.prompts[1]
    provenance = read(tmp_path / "audit/provenance.jsonl")[0]
    assert provenance["attempts"][0]["error"].startswith("ValueError:")


def test_failed_model_reviews_do_not_claim_catalog_coverage(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(1), record(2)])
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake",
                            client_factory=Factory(lambda n, p: "not json"))
    assert summary["status"] == "completed_with_errors"
    assert summary["batches_error"] == 1
    assert summary["coverage"]["claude_catalog_records_reviewed"] == 0
    assert len(read(tmp_path / "audit/errors.jsonl")) == 1
    assert (tmp_path / "audit/groups.jsonl").read_text() == ""


def test_cross_catalog_candidates_are_brought_together_and_budget_is_visible(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(i) for i in range(8)])
    rows, _, _, _ = _load_catalog(source)
    jobs, coverage = _make_batches(rows, batch_size=2, max_prompt_chars=100000, max_neighbor_batches=0)
    assert coverage["catalog_records_planned"] == 8
    assert coverage["cross_batch_candidate_pairs"] > 0
    assert coverage["neighbor_batches_omitted_by_budget"] > 0
    assert all(len(job["indices"]) <= 2 for job in jobs)
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=Factory(),
                            batch_size=2, max_neighbor_batches=0)
    assert summary["status"] == "coverage_limited"
    assert summary["coverage"]["claude_catalog_records_reviewed"] == 8


def test_prompt_budgets_and_oversized_records_are_reported(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(1), record(2)])
    factory = Factory()
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory,
                            max_prompt_chars=10)
    assert not factory.calls
    assert summary["status"] == "coverage_limited"
    assert len(summary["coverage"]["records_without_model_comparison"]) == 2


def test_lsh_is_repeatable_and_records_bucket_limits():
    rows = [{"question": "Who founded the label that released Album?"}] * 140
    stats = {}
    first = list(_neighbor_pairs(rows, retrieval_stats=stats))
    assert first == list(_neighbor_pairs(rows))
    assert stats["lsh_bucket_entries_omitted"] > 0


def test_existing_directory_and_duplicate_ids_fail_before_api_calls(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(1), record(1)])
    factory = Factory()
    with pytest.raises(ValueError, match="Duplicate record ID"):
        audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory)
    assert factory.calls == 0
    with pytest.raises(FileExistsError):
        audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory)


def test_one_question_has_no_per_question_api_review(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(1)])
    factory = Factory()
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory)
    assert factory.calls == 0
    assert summary["statistics"]["records"] == 1
    assert summary["coverage"]["claude_catalog_records_reviewed"] == 0


def test_api_failure_is_separate_from_redundancy_findings(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(1), record(2)])
    def timeout(number, payload): raise TimeoutError("timeout")
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=Factory(timeout))
    assert summary["status"] == "completed_with_errors" and summary["batches_error"] == 1
    assert summary["model_group_counts"] == {}


def test_fatal_audit_failure_does_not_remove_generated_questions(tmp_path):
    original = write_records(tmp_path / "original.jsonl", [record(99)])
    value = json.loads(original.read_text())
    value["paragraphs"].extend({"title": f"Filler {i}", "paragraph_text": "Unrelated."} for i in range(40))
    original.write_text(json.dumps(value) + "\n")
    def denied(number, payload):
        raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "InvokeModel")
    factory = Factory(denied)
    with pytest.raises(ClientError):
        run_batch(tmp_path / "run", original=original, model_id="fake", n=2, client_factory=factory)
    assert len(read(tmp_path / "run/accepted.jsonl")) == 2
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == "failed"
    assert all(c.closed for c in factory.clients)


def test_dry_run_exposes_coverage_and_call_budget_without_a_client(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(i) for i in range(8)])
    summary = audit_dataset(source, tmp_path / "plan", model_id="fake", batch_size=4, dry_run=True)
    assert summary["status"] == "planned" and summary["api_calls"] == 0
    plan = json.loads((tmp_path / "plan/plan.json").read_text())
    assert plan["coverage"]["catalog_records_planned"] == 8
    assert plan["maximum_model_invocations"] == plan["planned_batches"] * 2
    assert summary["coverage"]["claude_catalog_records_reviewed"] == 0
    assert (tmp_path / "plan/groups.jsonl").read_text() == ""


def test_audit_workers_overlap_but_never_share_output_writers(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(i) for i in range(8)])
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = peak = 0
    def callback(number, payload):
        nonlocal active, peak
        with lock: active += 1; peak = max(peak, active)
        try:
            barrier.wait(timeout=5)
            return response_for(payload)
        finally:
            with lock: active -= 1
    factory = Factory(callback)
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory,
                            concurrency=2, batch_size=2, max_neighbor_batches=0)
    assert peak == 2
    assert summary["coverage"]["claude_catalog_records_reviewed"] == 8
    assert all(c.closed for c in factory.clients)


def test_fatal_audit_error_drains_other_running_catalog_group(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(i) for i in range(8)])
    barrier = threading.Barrier(2)
    fatal_saved = threading.Event()
    def callback(number, payload):
        barrier.wait(timeout=5)
        if number == 1: raise AttributeError("audit bug")
        assert fatal_saved.wait(5)
        return response_for(payload)
    def progress(report):
        if report["status"] == "error": fatal_saved.set()
    factory = Factory(callback)
    with pytest.raises(AttributeError, match="audit bug"):
        audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory,
                      concurrency=2, batch_size=2, max_neighbor_batches=0, progress=progress)
    summary = json.loads((tmp_path / "audit/summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["batches_submitted"] == 2
    assert summary["batches_error"] == summary["batches_completed"] == 1
    assert all(c.closed for c in factory.clients)


def test_every_submitted_prompt_respects_character_budget_even_on_format_retry(tmp_path):
    source = write_records(tmp_path / "accepted.jsonl", [record(1), record(2)])
    catalog, *_ = _load_catalog(source)
    limit = len(_prompt(catalog, "catalog_00000001")) + 30
    factory = Factory(lambda number, payload: "invalid JSON")
    summary = audit_dataset(source, tmp_path / "audit", model_id="fake", client_factory=factory,
                            max_prompt_chars=limit, max_neighbor_batches=0)
    assert all(len(prompt) <= limit for prompt in factory.prompts)
    assert summary["batches_error"] == 1
