"""Concurrency checks use events/barriers and fake clients; no paid calls."""
import io
import json
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from mas_sae.data.synthetic import batch
from mas_sae.data.synthetic.generation_cli import main, make_client
from mas_sae.data.synthetic.prompt_examples import build_example


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.fixture
def original(tmp_path):
    path = tmp_path / "original.jsonl"
    path.write_text(json.dumps({
        "id": "original", "question": "Original question?",
        "paragraphs": [{"title": f"Filler {i}", "paragraph_text": "Unrelated content."}
                       for i in range(40)],
    }) + "\n")
    return path


class Factory:
    """Each fake is worker-local; callbacks coordinate API completion order."""
    def __init__(self, callback=None, sdk_retries=0):
        self.callback = callback
        self.sdk_retries = sdk_retries
        self.lock = threading.Lock()
        self.clients = []
        self.calls = 0
        self.active = 0
        self.peak = 0

    def __call__(self):
        factory = self

        class Client:
            model_id = "fake-requested"

            def __init__(self):
                self.owner = threading.get_ident()
                self.closed = 0

            def generate(self, prompt, **kwargs):
                assert threading.get_ident() == self.owner
                with factory.lock:
                    factory.calls += 1
                    number = factory.calls
                    factory.active += 1
                    factory.peak = max(factory.peak, factory.active)
                try:
                    hop = prompt.split("Hop subtype: ", 1)[1].splitlines()[0]
                    raw = build_example(hop)
                    raw["question"] += f" Fixture {number}"
                    text = (factory.callback(number, raw) if factory.callback
                            else json.dumps(raw))
                    return SimpleNamespace(text=text, model="fake-returned", input_tokens=10,
                                           output_tokens=20, stop_reason="end_turn",
                                           sdk_retry_attempts=factory.sdk_retries)
                finally:
                    with factory.lock:
                        factory.active -= 1

            def close(self):
                self.closed += 1

        client = Client()
        with self.lock:
            self.clients.append(client)
        return client


def run(tmp_path, original, factory, **kwargs):
    options = dict(n=12, concurrency=4, seed=42)
    if "target_per_type" in kwargs:
        options.pop("n")
    options.update(kwargs)
    return batch.run_batch(tmp_path / "run", original=original, model_id="fake-requested",
                           client_factory=factory, dedup_originals=[original], **options)


def test_requests_overlap_with_bounded_concurrency_and_one_writer(tmp_path, original):
    barrier = threading.Barrier(4)

    def response(number, raw):
        barrier.wait(timeout=5)
        return json.dumps(raw)

    factory = Factory(response)
    summary = run(tmp_path, original, factory)
    assert factory.peak == 4
    assert len(factory.clients) == 4
    assert all(client.closed == 1 for client in factory.clients)
    assert summary["accepted"] == summary["generated"] == summary["results_saved"] == 12
    assert summary["api_calls"] == 12
    assert summary["input_tokens"] == 120
    assert summary["output_tokens"] == 240
    provenance = read(tmp_path / "run/provenance.jsonl")
    reports = read(tmp_path / "run/validation.jsonl")
    assert len({p["request_id"] for p in provenance}) == 12
    assert {p["record_id"] for p in provenance} == {r["record_id"] for r in reports}
    assert all(p["started_at"] <= p["finished_at"] and p["elapsed_seconds"] >= 0 for p in provenance)
    assert all(p["model_id"] == "fake-requested" and p["attempts"][0]["model"] == "fake-returned"
               for p in provenance)


def test_completed_results_are_written_without_waiting_for_first_request(tmp_path, original):
    slow_started, fast_saved = threading.Event(), threading.Event()

    def response(number, raw):
        if len(raw["question_decomposition"]) == 2:
            slow_started.set()
            assert fast_saved.wait(5), "Coordinator waited for the slow first request"
        else:
            assert slow_started.wait(5)
        return json.dumps(raw)

    def progress(report):
        if report["request_id"].startswith("3hop1"):
            fast_saved.set()

    factory = Factory(response)
    run(tmp_path, original, factory, n=2, concurrency=2, progress=progress)
    reports = read(tmp_path / "run/validation.jsonl")
    assert reports[0]["hop_type"] == "3hop1"


def test_quota_has_no_overshoot_even_with_more_workers_than_target(tmp_path, original):
    factory = Factory()
    summary = run(tmp_path, original, factory, target_per_type=3,
                  max_requests_per_type=5, concurrency=16)
    assert summary["status"] == "completed"
    assert summary["accepted_by_type"] == dict.fromkeys(batch.HOP_TYPES, 3)
    assert summary["requests_submitted"] == 18
    assert summary["accepted"] == 18


def test_cross_worker_duplicate_is_replaced_centrally(tmp_path, original):
    barrier = threading.Barrier(2)

    def response(number, raw):
        if number <= 2:
            raw["question"] = "Same normalized question?"
            barrier.wait(timeout=5)
        return json.dumps(raw)

    summary = run(tmp_path, original, Factory(response), concurrency=2,
                  hop_types=["2hop", "3hop1"], target_per_type=1, max_requests_per_type=3)
    assert summary["accepted"] == 2
    assert summary["rejected"] == 1
    assert summary["requests_submitted"] == 3
    assert sum(error["code"] == "duplicate" for report in read(tmp_path / "run/validation.jsonl")
               for error in report["errors"]) == 1


def test_original_duplicate_is_excluded(tmp_path, original):
    def response(number, raw):
        raw["question"] = " ORIGINAL   QUESTION? "
        return json.dumps(raw)

    summary = run(tmp_path, original, Factory(response), n=1, concurrency=1)
    assert summary["accepted"] == 0 and summary["rejected"] == 1
    assert read(tmp_path / "run/validation.jsonl")[0]["errors"][0]["record_id"] == "original"


def test_invalid_generation_respects_quota_request_cap(tmp_path, original):
    summary = run(tmp_path, original, Factory(lambda n, raw: "invalid"),
                  target_per_type=1, max_requests_per_type=2, max_attempts=2)
    assert summary["status"] == "quota_not_met"
    assert summary["accepted"] == 0
    assert summary["requests_submitted"] == 12
    assert summary["api_calls"] == 24


@pytest.mark.parametrize("error", [AttributeError("bug"), TimeoutError("API timeout"), KeyboardInterrupt()])
def test_fatal_error_drains_other_running_results_and_stops_scheduling(tmp_path, original, error):
    barrier = threading.Barrier(2)
    fatal_saved = threading.Event()

    def response(number, raw):
        barrier.wait(timeout=5)
        if number == 1:
            raise error
        assert fatal_saved.wait(5)
        return json.dumps(raw)

    def progress(report):
        if any(error["code"] == "fatal" for error in report["errors"]):
            fatal_saved.set()

    factory = Factory(response)
    with pytest.raises(type(error)):
        run(tmp_path, original, factory, n=10, concurrency=2, progress=progress)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == ("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
    assert summary["requests_submitted"] == summary["results_saved"] == 2
    assert summary["accepted"] == 1
    assert summary["failed"] == 1
    assert len(read(tmp_path / "run/provenance.jsonl")) == 2
    assert all(client.closed == 1 for client in factory.clients)


def test_coordinator_interrupt_drains_running_calls(tmp_path, original, monkeypatch):
    both_started, release = threading.Event(), threading.Event()
    barrier = threading.Barrier(2, action=both_started.set)

    def response(number, raw):
        barrier.wait(timeout=5)
        assert release.wait(5)
        return json.dumps(raw)

    def interrupted_wait(*args, **kwargs):
        assert both_started.wait(5)
        release.set()
        raise KeyboardInterrupt()

    monkeypatch.setattr(batch, "wait", interrupted_wait)
    factory = Factory(response)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, original, factory, n=10, concurrency=2)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert summary["results_saved"] == summary["accepted"] == 2
    assert all(client.closed == 1 for client in factory.clients)


def test_request_seeds_do_not_depend_on_concurrency(tmp_path, original):
    maps = []
    for concurrency in (1, 4):
        output = tmp_path / str(concurrency)
        batch.run_batch(output, original=original, model_id="fake", n=12,
                        concurrency=concurrency, seed=42, client_factory=Factory())
        maps.append({p["request_id"]: p["request_seed"] for p in read(output / "provenance.jsonl")})
    assert maps[0] == maps[1]
    assert len(set(maps[0].values())) == 12


def test_client_close_failure_fails_run_and_preserves_generation_error(tmp_path, original):
    class Client:
        model_id = "fake"
        def generate(self, *args, **kwargs):
            raise AttributeError("original failure")
        def close(self):
            raise RuntimeError("close failure")

    with pytest.raises(AttributeError, match="original failure"):
        run(tmp_path, original, Client, n=1, concurrency=1)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["status"] == "failed"
    assert "original failure" in summary["fatal_error"]
    assert any("close failure" in error for error in summary["secondary_errors"])


def test_existing_output_is_refused_before_any_calls(tmp_path, original):
    (tmp_path / "run").mkdir()
    factory = Factory()
    with pytest.raises(FileExistsError):
        run(tmp_path, original, factory)
    assert not factory.clients


@pytest.mark.parametrize("options", [
    {"concurrency": 0}, {"hop_types": ["2hop", "2hop"]},
    {"target_per_type": 3, "max_requests_per_type": 2},
])
def test_invalid_configuration_makes_no_calls(tmp_path, original, options):
    factory = Factory()
    with pytest.raises(ValueError):
        run(tmp_path, original, factory, **options)
    assert not factory.clients
    assert not (tmp_path / "run").exists()


def test_cli_quota_mode_and_cap_exit_status(tmp_path, original, monkeypatch):
    from mas_sae.data.synthetic import generation_cli
    factory = Factory(lambda n, raw: "invalid")
    monkeypatch.setattr(generation_cli, "make_client", lambda *a, **k: factory())
    result = main(["--target-per-hop", "1", "--max-requests-per-hop", "1",
                   "--concurrency", "2", "--hop-types", "2hop", "3hop1",
                   "--original", str(original), "--output-dir", str(tmp_path / "run")])
    assert result == 1
    assert factory.calls == 4


def test_client_transport_config_and_temperature_omission(monkeypatch):
    import boto3
    requests = []
    configs = []
    class Session:
        def client(self, name, **kwargs):
            configs.append(kwargs["config"])
            return self
        def invoke_model(self, **kwargs):
            requests.append(kwargs)
            return {"ResponseMetadata": {"RetryAttempts": 2}, "body": io.BytesIO(json.dumps({
                "model": "returned", "content": [{"type": "text", "text": "OK"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}, "stop_reason": "end_turn",
            }).encode())}
        def close(self):
            pass
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: Session())
    client = make_client("requested", connect_timeout=7, read_timeout=123, sdk_max_attempts=2)
    assert configs[0].connect_timeout == 7
    assert configs[0].read_timeout == 123
    assert configs[0].retries == {"mode": "standard", "total_max_attempts": 2}
    for temperature in (None, 0.0):
        result = client.generate("Test", temperature=temperature, task="other")
        body = json.loads(requests[-1]["body"])
        assert ("temperature" in body) == (temperature is not None)
        assert requests[-1]["modelId"] == "requested" and result.model == "returned"
        assert result.sdk_retry_attempts == 2
    client.close()


def test_injected_serial_client_is_not_closed(tmp_path, original):
    client = Factory()()
    # A caller-owned serial fake may be used by the executor's worker.
    client.owner = None
    class Adapter:
        model_id = client.model_id
        def generate(self, *args, **kwargs):
            client.owner = threading.get_ident()
            return client.generate(*args, **kwargs)
    batch.run_batch(tmp_path / "serial", original=original, model_id="fake", n=1,
                    client=Adapter())
    assert client.closed == 0


def test_sixteen_requests_really_overlap(tmp_path, original):
    barrier = threading.Barrier(16)
    def response(number, raw):
        barrier.wait(timeout=5)
        return json.dumps(raw)
    factory = Factory(response)
    summary = run(tmp_path, original, factory, n=16, concurrency=16)
    assert factory.peak == 16
    assert summary["accepted"] == 16


def test_terminal_throttle_is_recorded_without_hiding_error(tmp_path, original):
    from botocore.exceptions import ClientError
    def response(number, raw):
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "limit"}}, "InvokeModel")
    with pytest.raises(ClientError):
        run(tmp_path, original, Factory(response), n=1, concurrency=1)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["api_errors_by_code"] == {"ThrottlingException": 1}
    assert summary["status"] == "failed"


def test_sdk_retries_are_separate_from_model_invocations(tmp_path, original):
    summary = run(tmp_path, original, Factory(sdk_retries=2), n=3, concurrency=1)
    assert summary["api_calls"] == summary["completed_responses"] == 3
    assert summary["sdk_retry_attempts"] == 6
    assert all(row["attempts"][0]["sdk_retry_attempts"] == 2
               for row in read(tmp_path / "run/provenance.jsonl"))
