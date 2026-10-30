"""Bounded concurrent generation; only the coordinator writes run artifacts."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median
import threading
from time import perf_counter
from typing import Any, Callable, Sequence

from mas_sae.experiments.artifacts import write_json_atomic

from ..deduplication import DuplicateIndex
from ..hop_types import HOP_TYPES
from ..validation import validate_candidate
from ..workers import WorkerClients
from .assembly import ParagraphPool
from .prompts import PROMPT_VERSION
from .service import generate_one


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _Result:
    candidate: dict | None
    provenance: dict
    error: BaseException | None = None


def run_batch(
    output_dir: str | Path,
    *,
    original: str | Path,
    model_id: str,
    n: int | None = None,
    target_per_type: int | None = None,
    max_requests_per_type: int | None = None,
    hop_types: Sequence[str] = tuple(HOP_TYPES),
    concurrency: int = 1,
    seed: int | None = None,
    max_attempts: int = 2,
    max_tokens: int = 6000,
    temperature: float | None = None,
    dedup_originals: Sequence[str | Path] = (),
    client_factory: Callable[[], Any] | None = None,
    client: Any = None,
    transport_settings: dict | None = None,
    progress: Callable[[dict], None] | None = None,
    dataset_audit: bool = True,
    audit_client_factory: Callable[[], Any] | None = None,
    audit_model_id: str | None = None,
    audit_concurrency: int | None = None,
    audit_batch_size: int = 100,
    audit_max_prompt_chars: int = 100000,
    audit_max_neighbor_batches: int = 200,
    audit_max_attempts: int = 2,
    audit_max_tokens: int = 8192,
) -> dict:
    """Run a fixed request count OR bounded accepted quotas into a new directory.

    Factories create one owned client per worker thread, reused until shutdown.
    An injected client is caller-owned and allowed only with concurrency=1.
    Fatal errors/interrupts stop scheduling; running work is drained and saved
    before the original error is raised. Running network calls cannot be cancelled.
    Quota acceptance order and duplicate winners may depend on completion order.
    """
    def positive(name, value):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    positive("concurrency", concurrency)
    positive("max_attempts", max_attempts)
    positive("max_tokens", max_tokens)
    if type(dataset_audit) is not bool:
        raise ValueError("dataset_audit must be boolean")
    if dataset_audit:
        if audit_model_id not in (None, model_id) and audit_client_factory is None:
            raise ValueError("A different audit_model_id requires audit_client_factory")
        if audit_concurrency is None:
            audit_concurrency = 1 if client is not None and audit_client_factory is None else min(concurrency, 4)
        positive("audit_concurrency", audit_concurrency)
        positive("audit_max_attempts", audit_max_attempts)
        positive("audit_max_tokens", audit_max_tokens)
        positive("audit_batch_size", audit_batch_size)
        positive("audit_max_prompt_chars", audit_max_prompt_chars)
        if audit_batch_size < 2 or type(audit_max_neighbor_batches) is not int or audit_max_neighbor_batches < 0:
            raise ValueError("audit_batch_size must be >= 2; neighbor batch cap must be nonnegative")
        if client is not None and audit_client_factory is None and audit_concurrency != 1:
            raise ValueError("An injected audit client requires audit_concurrency=1")
    if (n is None) == (target_per_type is None):
        raise ValueError("Specify exactly one of n or target_per_type")
    if n is not None:
        positive("n", n)
        if max_requests_per_type is not None:
            raise ValueError("max_requests_per_type requires target_per_type")
    else:
        positive("target_per_type", target_per_type)
        positive("max_requests_per_type", max_requests_per_type)
        if max_requests_per_type < target_per_type:
            raise ValueError("max_requests_per_type must be >= target_per_type")
    hops = tuple(hop_types)
    if not hops or len(set(hops)) != len(hops) or any(h not in HOP_TYPES for h in hops):
        raise ValueError("hop_types must be nonempty, unique supported subtypes")
    if temperature is not None and not 0 <= temperature <= 1:
        raise ValueError("temperature must be between 0 and 1 or None")
    if client is not None and (concurrency != 1 or client_factory is not None):
        raise ValueError("An injected client requires concurrency=1 and no factory")
    if client is None and client_factory is None:
        raise ValueError("Provide client_factory or a serial client")
    original = Path(original)
    dedup_paths = tuple(Path(p) for p in dedup_originals)
    for path in (original, *dedup_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    run_started = _now()
    clock_started = perf_counter()
    effective_seed = seed if seed is not None else random.SystemRandom().getrandbits(64)
    requested, accepted, generated, inflight = Counter(), Counter(), Counter(), Counter()
    counts = Counter(generated=0, accepted=0, failed=0, rejected=0, cancelled=0,
                     api_calls=0, completed_responses=0, input_tokens=0, output_tokens=0,
                     sdk_retry_attempts=0)
    stop = threading.Event()
    clients = WorkerClients(client_factory, client)
    pending: dict[Future, dict] = {}
    secondary_errors: list[str] = []
    fatal: BaseException | None = None
    status = "running"
    durations: list[float] = []
    api_errors: Counter = Counter()
    next_hop = 0
    submitted = 0
    saved = 0

    def execute(request: dict) -> _Result:
        provenance = {**request, "record_id": None, "model_id": model_id,
                      "attempts": [], "api_calls": 0}
        if stop.is_set():
            provenance["cancelled"] = True
            return _Result(None, provenance)
        provenance["started_at"] = _now()
        started = perf_counter()
        candidate = None
        error = None
        try:
            worker_client = clients.get()
            candidate, _ = generate_one(
                worker_client, request["hop_type"], pool, random.Random(request["request_seed"]),
                max_attempts=max_attempts, max_tokens=max_tokens,
                temperature=temperature, provenance=provenance,
            )
        except BaseException as exc:
            error = exc
            provenance["error"] = f"{type(exc).__name__}: {exc}"
            response = getattr(exc, "response", None)
            if isinstance(response, dict):
                provenance["api_error_code"] = response.get("Error", {}).get("Code")
            stop.set()
        finally:
            provenance["finished_at"] = _now()
            provenance["elapsed_seconds"] = perf_counter() - started
        return _Result(candidate, provenance, error)

    def schedule(executor):
        nonlocal next_hop, submitted
        while len(pending) < concurrency and not stop.is_set():
            selected = None
            for _ in hops:
                hop = hops[next_hop % len(hops)]
                next_hop += 1
                if n is not None:
                    if submitted < n:
                        selected = hop
                        break
                elif (requested[hop] < max_requests_per_type
                      and accepted[hop] + inflight[hop] < target_per_type):
                    selected = hop
                    break
            if selected is None:
                return
            requested[selected] += 1
            submitted += 1
            request_id = f"{selected}_request_{requested[selected]:06d}"
            request_seed = int.from_bytes(hashlib.sha256(
                f"{effective_seed}:{request_id}".encode()
            ).digest()[:8], "big")
            request = {"request_id": request_id, "hop_type": selected,
                       "request_seed": request_seed, "submitted_at": _now()}
            future = executor.submit(execute, request)
            pending[future] = request
            inflight[selected] += 1

    def line(stream, value):
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()

    def persist(result):
        nonlocal saved
        provenance = result.provenance
        candidate = result.candidate
        hop = provenance["hop_type"]
        inflight[hop] -= 1
        counts["api_calls"] += provenance.get("api_calls", 0)
        if "elapsed_seconds" in provenance:
            durations.append(provenance["elapsed_seconds"])
        if provenance.get("api_error_code"):
            api_errors[provenance["api_error_code"]] += 1
        for attempt in provenance["attempts"]:
            counts["completed_responses"] += 1
            counts["sdk_retry_attempts"] += attempt.get("sdk_retry_attempts", 0)
            counts["input_tokens"] += attempt["input_tokens"]
            counts["output_tokens"] += attempt["output_tokens"]
        line(provenance_file, provenance)
        report = {"request_id": provenance["request_id"], "hop_type": hop,
                  "record_id": candidate["id"] if candidate else None,
                  "accepted": False, "errors": []}
        if provenance.get("cancelled"):
            counts["cancelled"] += 1
            report["cancelled"] = True
        elif result.error is not None:
            counts["failed"] += 1
            report["errors"] = [{"code": "fatal", "message": provenance["error"]}]
        elif candidate is None:
            counts["failed"] += 1
            report["errors"] = [{"code": "generation_exhausted",
                                 "message": "See provenance attempts."}]
        else:
            line(candidates_file, candidate)
            counts["generated"] += 1
            generated[hop] += 1
            report["errors"] = [asdict(issue) for issue in validate_candidate(candidate).errors]
            if not report["errors"]:
                duplicate = index.register_question(candidate["id"], candidate["question"],
                                                    source=str(output_dir.resolve()))
                if duplicate:
                    report["errors"] = [{"code": "duplicate", **asdict(duplicate)}]
                else:
                    line(accepted_file, candidate)
                    accepted[hop] += 1
                    counts["accepted"] += 1
                    report["accepted"] = True
            if not report["accepted"]:
                counts["rejected"] += 1
        line(validation_file, report)
        saved += 1
        if progress is not None:
            progress(report)

    try:
        pool = ParagraphPool.from_jsonl(original)
        with ExitStack() as stack:
            candidates_file = stack.enter_context((output_dir / "candidates.jsonl").open("w", encoding="utf-8"))
            accepted_file = stack.enter_context((output_dir / "accepted.jsonl").open("w", encoding="utf-8"))
            validation_file = stack.enter_context((output_dir / "validation.jsonl").open("w", encoding="utf-8"))
            provenance_file = stack.enter_context((output_dir / "provenance.jsonl").open("w", encoding="utf-8"))
            index = stack.enter_context(DuplicateIndex(output_dir / "question_index.sqlite"))
            for path in dedup_paths:
                index.import_jsonl(path, source=str(path.resolve()))
            executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="synthetic")
            try:
                schedule(executor)
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        pending.pop(future)
                        result = future.result()
                        persist(result)
                        if result.error is not None:
                            raise result.error
                    schedule(executor)
            except BaseException as exc:
                fatal = exc
            finally:
                # Keep output files open while already-running calls finish.
                stop.set()
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                for future, request in pending.items():
                    try:
                        if future.cancelled():
                            provenance = {**request, "attempts": [], "api_calls": 0,
                                          "cancelled": True}
                            result = _Result(None, provenance)
                        else:
                            result = future.result()
                        persist(result)
                        if result.error is not None and fatal is None:
                            fatal = result.error
                    except BaseException as exc:
                        if fatal is None:
                            fatal = exc
                        else:
                            secondary_errors.append(f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        if fatal is None:
            fatal = exc
        else:
            secondary_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        for exc in clients.close():
            if fatal is None:
                fatal = exc
            else:
                secondary_errors.append(f"Client close: {type(exc).__name__}: {exc}")
        if fatal is not None:
            status = "interrupted" if isinstance(fatal, KeyboardInterrupt) else "failed"
        elif target_per_type is not None and any(accepted[h] < target_per_type for h in hops):
            status = "quota_not_met"
        else:
            status = "completed"
        elapsed = perf_counter() - clock_started
        summary = {
            "status": status, "fatal_error": f"{type(fatal).__name__}: {fatal}" if fatal else None,
            "secondary_errors": secondary_errors, **counts,
            "accepted_total": counts["accepted"], "requests_submitted": submitted,
            "results_saved": saved, "concurrency": concurrency,
            "generated_by_hop_type": dict(generated),
            "requests_by_type": {h: requested[h] for h in hops},
            "accepted_by_type": {h: accepted[h] for h in hops},
            "model_id": client.model_id if client is not None else model_id,
            "prompt_version": PROMPT_VERSION, "max_tokens": max_tokens,
            "temperature": temperature, "max_attempts": max_attempts,
            "seed": seed, "effective_seed": effective_seed, "n": n,
            "target_per_type": target_per_type, "max_requests_per_type": max_requests_per_type,
            "started_at": run_started, "finished_at": _now(), "elapsed_seconds": elapsed,
            "accepted_per_minute": counts["accepted"] * 60 / elapsed if elapsed else 0,
            "request_latency_p50_seconds": median(durations) if durations else None,
            "request_latency_p95_seconds": sorted(durations)[math.ceil(.95 * len(durations)) - 1] if durations else None,
            "api_errors_by_code": dict(api_errors),
            "original_file": str(original.resolve()),
            "dedup_originals": [str(p.resolve()) for p in dedup_paths],
            "transport_settings": transport_settings,
            "output_dir": str(output_dir.resolve()),
            "dataset_audit": {"enabled": dataset_audit,
                                    "status": "not_run" if fatal else "pending" if dataset_audit else "disabled"},
        }
        try:
            write_json_atomic(output_dir / "summary.json", summary)
        except BaseException:
            if fatal is None:
                raise
            # Summary cannot be persisted (e.g. disk full); retain the original failure.
    if fatal is not None:
        raise fatal
    if dataset_audit:
        from ..audit import audit_dataset

        summary["generation_status"] = summary["status"]
        audit_dir = output_dir / "dataset_audit"
        summary["dataset_audit"] = {"enabled": True, "status": "running", "output_dir": str(audit_dir.resolve())}
        write_json_atomic(output_dir / "summary.json", summary)
        audit_error = None
        try:
            audit = audit_dataset(
                output_dir / "accepted.jsonl", audit_dir, model_id=audit_model_id or model_id,
                client_factory=audit_client_factory or (client_factory if client is None else None),
                client=client if audit_client_factory is None else None,
                concurrency=audit_concurrency, batch_size=audit_batch_size,
                max_prompt_chars=audit_max_prompt_chars, max_neighbor_batches=audit_max_neighbor_batches,
                max_attempts=audit_max_attempts, max_tokens=audit_max_tokens,
                transport_settings=transport_settings, progress=progress,
            )
            summary["dataset_audit"] = {"enabled": True, **audit}
            if audit["status"] != "completed":
                summary["status"] = "dataset_audit_incomplete"
        except BaseException as exc:
            audit_error = exc
            summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            summary["fatal_error"] = f"Dataset audit: {type(exc).__name__}: {exc}"
            if (audit_dir / "summary.json").exists():
                summary["dataset_audit"] = {"enabled": True, **json.loads((audit_dir / "summary.json").read_text())}
            else:
                summary["dataset_audit"]["status"] = "failed"
            raise
        finally:
            summary["pipeline_finished_at"] = _now()
            summary["total_elapsed_seconds"] = perf_counter() - clock_started
            try:
                write_json_atomic(output_dir / "summary.json", summary)
            except BaseException:
                if audit_error is None:
                    raise
    return summary
