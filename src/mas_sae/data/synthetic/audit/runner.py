"""Execute dataset audits and persist linked findings, coverage, and reports."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import threading
from time import perf_counter
from typing import Callable

from mas_sae.experiments.artifacts import sha256_file, sha256_json, utc_now, write_json_atomic

from ..workers import WorkerClients
from .catalog import _load_catalog
from .planning import _make_batches
from .review import AUDIT_VERSION, RELATIONSHIPS, _review_batch


def audit_dataset(
    input_path: str | Path, output_dir: str | Path, *, model_id: str,
    client_factory: Callable | None = None, client=None, concurrency: int = 16,
    batch_size: int = 100, max_prompt_chars: int = 100000, max_neighbor_batches: int = 200,
    max_attempts: int = 2, max_tokens: int = 8192,
    transport_settings: dict | None = None, progress: Callable | None = None,
    dry_run: bool = False,
) -> dict:
    """Audit every question locally and compare bounded catalog/neighbor groups.

    This reports relationships and aggregate patterns; it never removes questions
    or certifies per-question quality. Catalog coverage and approximate cross-group
    retrieval limitations are recorded. No automatic resume or paid benchmark.
    """
    for name, value in (("concurrency", concurrency), ("batch_size", batch_size),
                        ("max_prompt_chars", max_prompt_chars), ("max_attempts", max_attempts),
                        ("max_tokens", max_tokens)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if batch_size < 2 or type(max_neighbor_batches) is not int or max_neighbor_batches < 0:
        raise ValueError("batch_size must be >=2 and max_neighbor_batches must be nonnegative")
    if client is not None and (concurrency != 1 or client_factory is not None):
        raise ValueError("An injected client requires concurrency=1 and no factory")
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be boolean")
    if not dry_run and client is None and client_factory is None:
        raise ValueError("Provide client_factory or a serial client")
    input_path, output_dir = Path(input_path), Path(output_dir)
    if not input_path.is_file(): raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at, started = utc_now().isoformat(), perf_counter()
    clients = WorkerClients(client_factory, client)
    stop = threading.Event()
    pending = {}
    fatal = None
    secondary_errors = []
    counts = Counter(batches_submitted=0, batches_completed=0, batches_error=0, batches_cancelled=0,
                     api_calls=0, completed_responses=0, input_tokens=0, output_tokens=0, sdk_retry_attempts=0)
    catalog_reviewed, neighbor_reviewed = set(), set()
    all_groups, all_observations, group_keys = [], [], set()
    stats, coverage, records = {}, {}, []
    source_hash = None

    def line(stream, value):
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()

    def execute(job):
        selected = [records[i] for i in job["indices"]]
        provenance = {"batch_id": job["batch_id"], "kind": job["kind"],
                      "record_ids": [r["record_id"] for r in selected],
                      "catalog_sha256": sha256_json(selected), "model_id": model_id,
                      "audit_version": AUDIT_VERSION, "attempts": [], "api_calls": 0}
        if stop.is_set():
            provenance["cancelled"] = True
            return job, {"status": "cancelled"}, provenance, None
        clock = perf_counter()
        provenance["started_at"] = utc_now().isoformat()
        try:
            result = _review_batch(selected, job["batch_id"], clients.get(), provenance,
                                   max_attempts=max_attempts, max_tokens=max_tokens, max_prompt_chars=max_prompt_chars)
            return job, result, provenance, None
        except BaseException as exc:
            provenance["fatal_error"] = f"{type(exc).__name__}: {exc}"
            stop.set()
            return job, {"status": "error", "error": provenance["fatal_error"]}, provenance, exc
        finally:
            provenance["finished_at"] = utc_now().isoformat()
            provenance["elapsed_seconds"] = perf_counter() - clock

    def persist(outcome):
        job, result, provenance, error = outcome
        counts["api_calls"] += provenance.get("api_calls", 0)
        for attempt in provenance["attempts"]:
            counts["completed_responses"] += int("raw_text" in attempt)
            for key in ("input_tokens", "output_tokens", "sdk_retry_attempts"):
                counts[key] += attempt.get(key, 0)
        line(provenance_file, provenance)
        line(reviews_file, {"batch_id": job["batch_id"], "kind": job["kind"], **result})
        if result["status"] == "completed":
            counts["batches_completed"] += 1
            (catalog_reviewed if job["kind"] == "catalog" else neighbor_reviewed).update(provenance["record_ids"])
            for group in result["groups"]:
                key = (group["relationship"], tuple(sorted(group["record_ids"])))
                if key not in group_keys:
                    group_keys.add(key)
                    finding = {"source": "claude", "batch_id": job["batch_id"],
                               "group_id": sha256_json(key)[:16], **group}
                    all_groups.append(finding); line(groups_file, finding)
            for observation in result["observations"]:
                finding = {"batch_id": job["batch_id"], **observation}
                all_observations.append(finding); line(observations_file, finding)
        else:
            counts["batches_cancelled" if result["status"] == "cancelled" else "batches_error"] += 1
            line(errors_file, {"batch_id": job["batch_id"], "record_ids": provenance["record_ids"], **result})
        if progress is not None:
            progress({"phase": "dataset_audit", "batch_id": job["batch_id"], "status": result["status"]})

    try:
        source_hash = sha256_file(input_path)
        records, stats, exact, templates = _load_catalog(input_path)
        jobs, coverage = _make_batches(records, batch_size, max_prompt_chars, max_neighbor_batches)
        write_json_atomic(output_dir / "plan.json", {
            "audit_version": AUDIT_VERSION, "model_id": model_id, "input_sha256": source_hash,
            "planned_batches": len(jobs), "maximum_model_invocations": len(jobs) * max_attempts,
            "sdk_retries_excluded_from_call_bound": True, "coverage": coverage,
            "batches": [{"batch_id": job["batch_id"], "kind": job["kind"],
                         "record_ids": [records[i]["record_id"] for i in job["indices"]]}
                        for job in jobs],
        })
        if progress is not None:
            progress({"phase": "dataset_audit", "batch_id": "plan", "status": "planned",
                      "planned_batches": len(jobs), "maximum_model_invocations": len(jobs) * max_attempts})
        with (output_dir / "catalog.jsonl").open("w", encoding="utf-8") as stream:
            for record in records: line(stream, record)
        with (output_dir / "exact_duplicates.jsonl").open("w", encoding="utf-8") as stream:
            for question, ids in exact.items():
                if len(ids) > 1:
                    line(stream, {"source": "normalized_question", "question": question, "record_ids": ids})
        with (output_dir / "template_groups.jsonl").open("w", encoding="utf-8") as stream:
            for template, ids in templates.items():
                if len(ids) > 1:
                    line(stream, {"source": "heuristic", "template": template, "record_ids": ids})
        stats["exact_duplicate_groups"] = sum(len(ids) > 1 for ids in exact.values())
        stats["exact_duplicate_excess_records"] = sum(max(0, len(ids) - 1) for ids in exact.values())
        stats["repeated_template_groups"] = sum(len(ids) > 1 for ids in templates.values())
        write_json_atomic(output_dir / "statistics.json", stats)
        with (
            (output_dir / "groups.jsonl").open("w", encoding="utf-8") as groups_file,
            (output_dir / "observations.jsonl").open("w", encoding="utf-8") as observations_file,
            (output_dir / "reviews.jsonl").open("w", encoding="utf-8") as reviews_file,
            (output_dir / "provenance.jsonl").open("w", encoding="utf-8") as provenance_file,
            (output_dir / "errors.jsonl").open("w", encoding="utf-8") as errors_file,
        ):
            iterator = iter([] if dry_run else jobs)
            exhausted = False
            executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="dataset-audit")

            def schedule():
                nonlocal exhausted
                while not exhausted and not stop.is_set() and len(pending) < concurrency:
                    job = next(iterator, None)
                    if job is None:
                        exhausted = True
                        break
                    pending[executor.submit(execute, job)] = job
                    counts["batches_submitted"] += 1

            try:
                schedule()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        outcome = future.result()
                        persist(outcome)
                        if outcome[3] is not None: raise outcome[3]
                    schedule()
            except BaseException as exc:
                fatal = exc
            finally:
                stop.set()
                for future in pending: future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                for future, job in pending.items():
                    try:
                        if future.cancelled():
                            provenance = {"batch_id": job["batch_id"], "kind": job["kind"],
                                          "record_ids": [records[i]["record_id"] for i in job["indices"]],
                                          "api_calls": 0, "attempts": [], "cancelled": True}
                            outcome = (job, {"status": "cancelled"}, provenance, None)
                        else:
                            outcome = future.result()
                        persist(outcome)
                        if outcome[3] is not None and fatal is None: fatal = outcome[3]
                    except BaseException as exc:
                        if fatal is None: fatal = exc
                        else: secondary_errors.append(f"{type(exc).__name__}: {exc}")
        if sha256_file(input_path) != source_hash:
            raise ValueError("Audit input changed during processing")
    except BaseException as exc:
        if fatal is None: fatal = exc
        else: secondary_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        for exc in clients.close():
            if fatal is None: fatal = exc
            else: secondary_errors.append(f"Client close: {type(exc).__name__}: {exc}")
        coverage.update(claude_catalog_records_reviewed=len(catalog_reviewed),
                        claude_neighbor_records_reviewed=len(neighbor_reviewed))
        group_counts = Counter(g["relationship"] for g in all_groups)
        affected = {kind: len({rid for g in all_groups if g["relationship"] == kind for rid in g["record_ids"]})
                    for kind in sorted(RELATIONSHIPS)}
        limited = (coverage.get("neighbor_batches_omitted_by_budget", 0) > 0
                   or coverage.get("lsh_bucket_entries_omitted", 0) > 0
                   or bool(coverage.get("records_without_model_comparison")) and len(records) > 1)
        status = ("interrupted" if isinstance(fatal, KeyboardInterrupt) else "failed" if fatal else
                  "planned" if dry_run else "completed_with_errors" if counts["batches_error"] else
                  "coverage_limited" if limited else "completed")
        summary = {"status": status, "fatal_error": f"{type(fatal).__name__}: {fatal}" if fatal else None,
                   "secondary_errors": secondary_errors, "audit_version": AUDIT_VERSION,
                   "model_id": client.model_id if client is not None else model_id,
                   "concurrency": concurrency, "batch_size": batch_size, "dry_run": dry_run,
                   "max_prompt_chars": max_prompt_chars, "max_neighbor_batches": max_neighbor_batches,
                   "max_attempts": max_attempts, "max_tokens": max_tokens,
                   "transport_settings": transport_settings, "input_path": str(input_path.resolve()),
                   "input_sha256": source_hash, "output_dir": str(output_dir.resolve()),
                   "started_at": started_at, "finished_at": utc_now().isoformat(),
                   "elapsed_seconds": perf_counter() - started, **counts, "coverage": coverage,
                   "statistics": stats, "model_group_counts": dict(group_counts),
                   "records_in_model_groups": affected,
                   "limitations": ["Similarity retrieval is approximate, not all-pairs comparison.",
                                   "Claude sees questions, answers, decompositions, and support titles, not full paragraphs.",
                                   "Findings are model assessments; absence of a finding does not prove uniqueness.",
                                   "Template similarity and entity reuse do not automatically make questions invalid.",
                                   "No records are deleted, rewritten, or approved by this audit."]}
        try:
            write_json_atomic(output_dir / "summary.json", summary)
            report = ["# Synthetic dataset redundancy audit", "", f"Status: {status}",
                      f"Records scanned locally: {stats.get('records', 0)}",
                      f"Records acknowledged in valid Claude catalog reviews: {len(catalog_reviewed)}",
                      f"Exact normalized-question duplicate groups: {stats.get('exact_duplicate_groups', 0)}",
                      f"Repeated template groups (heuristic): {stats.get('repeated_template_groups', 0)}", "",
                      "## Model-reported relationships", ""]
            report.extend(f"- {kind}: {group_counts[kind]} groups; {affected[kind]} unique records"
                          for kind in sorted(RELATIONSHIPS))
            report.extend(["", "Counts overlap; they are not an overall defective-question percentage.",
                           "", "## Repeated support titles", ""])
            report.extend(f"- {item['title']}: {item['count']} records" for item in stats.get("repeated_support_titles", [])[:20])
            report.extend(["", "## Example findings", ""])
            for group in all_groups[:20]:
                report.append(f"- [{group['relationship']}] {', '.join(group['record_ids'])}: {group['reason']}")
            report.extend(["", "## Model observations (bounded-group evidence)", ""])
            for observation in all_observations[:20]:
                report.append(f"- [{observation['code']}] {', '.join(observation['record_ids'])}: {observation['reason']}")
            report.extend(["", "## Coverage and limitations", "", json.dumps(coverage, indent=2), ""])
            report.extend(f"- {limitation}" for limitation in summary["limitations"])
            (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        except BaseException:
            if fatal is None: raise
    if fatal is not None: raise fatal
    return summary
