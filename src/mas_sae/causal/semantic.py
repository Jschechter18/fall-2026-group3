"""Claude cohort validation and coordinator-only, resumable replay labeling.

GPU workers never construct an API client. Raw generations remain immutable;
semantic judgments and their audits live in a separate directory.
"""
from __future__ import annotations

import csv
import fcntl
import json
from collections import Counter
from pathlib import Path

from mas_sae.evaluation import claude_validator as cv
from mas_sae.experiments.artifacts import file_facts, sha256_file, sha256_json, staged_output, write_json_atomic
from mas_sae.experiments.records import read_jsonl, write_jsonl
from mas_sae.utils.versioning import create_versioned_run_dir

RESPONSES = {"adoption": "adopted_critic", "retention": "retained_a1",
             "agreement": "unchanged_same_position", "out": "out"}
SOURCE_FILES = ("run_settings.json", "summary.json", "raw_results.jsonl", "binary_labels.csv", "exclusions.jsonl")
PROTOCOL_KEYS = ("model_id", "client", "effort", "max_tokens", "temperature", "prompt_version",
                 "prompt_sha256", "a2_presentation", "max_attempts")
FINAL = "claude_labeled"


def source_files(cfg) -> list[Path]:
    folder = Path(cfg.capstone_data_root) / cfg.claude["source_run"]
    return [folder / name for name in SOURCE_FILES]


def read_source(cfg) -> tuple[dict, dict[str, dict]]:
    """Validate a complete Claude run against the configured collection and current prompt."""
    paths = source_files(cfg)
    for p in paths:
        if not p.is_file():
            raise ValueError(f"Missing completed Claude source artifact: {p}; run Claude labeling first.")
    settings = json.loads(paths[0].read_text())
    summary = json.loads(paths[1].read_text())
    if settings.get("episodes_sha256") != sha256_file(Path(cfg.capstone_data_root) / cfg.interactions_jsonl):
        raise ValueError("Claude source was labeled from different intervention interactions.")
    if settings.get("partition") != cfg.partition or summary.get("partition") != cfg.partition:
        raise ValueError("Claude source partition differs from replay partition.")
    presentation = settings.get("a2_presentation")
    variants = [include for include in (True, False)
                if settings.get("prompt_version") == cv.prompt_version(include, presentation)
                and settings.get("prompt_sha256") == cv.prompt_sha256(include, presentation)]
    if len(variants) != 1:
        raise ValueError("Claude source prompt differs from the current semantic labeling protocol.")
    if settings.get("client") not in {"ClaudeClient", "AdaptiveThinkingClaudeClient"}:
        raise ValueError("Unsupported Claude source client.")
    if any(k not in settings for k in PROTOCOL_KEYS):
        raise ValueError("Claude source lacks required protocol settings.")
    if any(k in summary and summary[k] != settings[k] for k in PROTOCOL_KEYS):
        raise ValueError("Claude source summary/settings protocol mismatch.")
    protocol = {k: settings[k] for k in PROTOCOL_KEYS}
    protocol["include_question"] = variants[0]
    protocol["models_returned"] = summary.get("models_returned", [])
    if not protocol["models_returned"]:
        raise ValueError("Claude source has no recorded returned model version.")
    rows = read_jsonl(paths[2])
    labels = {}
    for row in rows:
        qid = row["question_id"]
        if qid in labels or row.get("partition") != cfg.partition:
            raise ValueError("Claude source has duplicate questions or mixed partitions.")
        if row.get("prompt_version") != protocol["prompt_version"]:
            raise ValueError("Claude source rows mix prompt versions.")
        if row.get("behavior_label") not in (*cv.VALID_LABELS, None):
            raise ValueError("Claude source has an unknown semantic label.")
        if row.get("behavior_label") in {"adoption", "retention"} and row.get("label_source") != "claude":
            raise ValueError("Binary intervention cohort labels must come from Claude.")
        if row.get("behavior_label") in {"adoption", "retention"} and row.get("primary_target") != int(row["behavior_label"] == "adoption"):
            raise ValueError("Claude source binary target contradicts its semantic judgment.")
        labels[qid] = row
    if len(rows) != summary.get("n_episodes") or settings.get("limit") is not None:
        raise ValueError("Claude cohort requires a complete full-partition labeling run, not a pilot.")
    errors = [q for q, r in labels.items() if r.get("behavior_label") is None]
    if errors and cfg.claude.get("source_errors", "stop") != "exclude":
        raise ValueError(f"Claude source has {len(errors)} technical failures; retry or explicitly set source_errors: exclude.")
    return protocol, labels


def packaged_labels(cfg, source: dict[str, dict]) -> dict[str, dict]:
    """Refuse rule packages, stale exports, missing rows, and technical failures disguised as labels."""
    path = Path(cfg.capstone_data_root) / cfg.labels_csv
    if not path.is_file():
        raise ValueError(f"Missing Claude label package {path}; run --prepare-inputs first.")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = {r["question_id"]: r for r in rows}
    expected = {q: r for q, r in source.items() if r.get("behavior_label") is not None}
    if len(rows) != len(labels) or set(labels) != set(expected):
        raise ValueError("Claude label package does not cover exactly the successful source labels.")
    for qid, row in labels.items():
        original = expected[qid]
        binary = original["behavior_label"] in {"adoption", "retention"}
        if (row.get("behavior_label") != original["behavior_label"] or row.get("partition") != cfg.partition
                or row.get("label_source") != original["label_source"]
                or row.get("prompt_version") != original["prompt_version"]
                or row.get("eligible_primary") != str(binary)
                or row.get("primary_target") != (str(original["primary_target"]) if binary else "")):
            raise ValueError(f"{qid}: Claude package differs from the source judgment.")
    return labels


def generation_label(ep: dict, answer: str) -> dict:
    """Reuse only an exact stored answer; every changed answer awaits Claude."""
    if answer == ep["stored_solver_attempt_2"]:
        return {"solver_response": ep["stored_solver_response"], "adopted": ep["stored_adopted"],
                "behavior_label": ep["stored_behavior_label"], "label_source": "claude:stored_answer"}
    return {"solver_response": "pending_claude", "adopted": None,
            "behavior_label": None, "label_source": "pending_claude"}


def _identity(run: Path) -> dict:
    meta = json.loads((run / "replay_config.json").read_text())
    from .replay import ReplayConfig
    cfg = ReplayConfig.from_dict(meta["config"])
    if cfg.label_policy != "claude":
        raise ValueError("--label-only requires label_policy: claude in the frozen run config.")
    protocol, source = read_source(cfg)
    identity = {"protocol": protocol, "source_sha256": {str(p): sha256_file(p) for p in source_files(cfg)},
                "generation_sha256": sha256_file(run / "per_row.jsonl"),
                "metadata_sha256": sha256_file(run / "replay_config.json"),
                "episodes_sha256": sha256_file(run / "episodes.csv"),
                "generated_errors": cfg.claude.get("generated_errors", "stop")}
    return {"config": cfg, "meta": meta, "identity": identity, "source": source}


def _create_client(cfg, protocol):
    """Create a Bedrock client only on the coordinator, using the frozen source protocol."""
    import boto3
    from botocore.config import Config
    from mas_sae.models.claude import ClaudeClient
    from mas_sae.evaluation.claude_adaptive_client import AdaptiveThinkingClaudeClient
    session = boto3.Session(profile_name=cfg.claude.get("aws_profile"))
    classes = {"ClaudeClient": ClaudeClient, "AdaptiveThinkingClaudeClient": AdaptiveThinkingClaudeClient}
    extra = {"effort": protocol["effort"]} if protocol["client"] == "AdaptiveThinkingClaudeClient" else {}
    return classes[protocol["client"]](
        session=session, model_id=protocol["model_id"], region=cfg.claude.get("region", "us-east-1"),
        config=Config(max_pool_connections=cfg.claude.get("concurrency", 1),
                      retries={"mode": "adaptive", "max_attempts": 8}), **extra)


def label_run(run_dir: Path, *, client=None, dry_run: bool = False) -> dict:
    """Deduplicate semantic inputs, reuse successful audits, retry failures, then publish atomically."""
    run = Path(run_dir)
    if dry_run:
        return _label_run(run, client=client, dry_run=True)
    # Kernel locks release on crashes; a leftover filename does not block resuming.
    with (run / "claude_labeling.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another coordinator is labeling this run.") from exc
        return _label_run(run, client=client)


def _label_run(run: Path, *, client=None, dry_run=False) -> dict:
    from . import replay
    state = _identity(run)
    cfg, meta, identity = state["config"], state["meta"], state["identity"]
    final = run / FINAL
    if final.exists():
        load_labeled_rows(run)
        return {**json.loads((final / "summary.json").read_text()), "n_requests_this_invocation": 0}
    episodes = replay.load_episodes(cfg).set_index("question_id").to_dict("index")
    rows = replay.read_rows(run / "per_row.jsonl")
    conds = {(c["experiment"], c["condition"], c["draw"]) for c in meta["conditions"]}
    expected = {(q, *c) for q in episodes for c in conds}
    keys = [replay.key(r) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Claude labeling requires exact episode/condition/draw generation coverage.")
    requests, row_requests = {}, {}
    for row in rows:
        ep = episodes[row["question_id"]]
        if row["solver_attempt_2"] == ep["stored_solver_attempt_2"]:
            continue
        payload = {k: ep[k] for k in ("question", "solver_attempt_1", "critic_advocated_answer",
                                      "critic_noncommittal", "a2_presentation", "candidate_order")}
        payload["solver_attempt_2"] = row["solver_attempt_2"]
        cache_key = sha256_json({"protocol": identity["protocol"], "payload": payload})
        requests[cache_key] = {**payload, "episode_id": cache_key, "question_id": row["question_id"],
                               "partition": cfg.partition}
        row_requests[replay.key(row)] = cache_key
    work = run / "claude_labeling"
    manifest = work / "manifest.json"
    cache = {}
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Claude labeling inputs or protocol changed; use a new run.")
        for audit in sorted((work / "attempts").glob("*/raw_results.jsonl")):
            if not dry_run:
                replay.repair_tail(audit)
            for r in read_jsonl(audit):
                if r["episode_id"] not in requests or r.get("prompt_version") != identity["protocol"]["prompt_version"]:
                    raise ValueError("Foreign or incompatible Claude cache row.")
                cache[r["episode_id"]] = r
    pending = [ep for k, ep in requests.items() if cache.get(k, {}).get("behavior_label") not in cv.VALID_LABELS]
    report = {"n_generations": len(rows), "n_unique_changed_inputs": len(requests),
              "n_pending_requests": len(pending), "n_reused_stored_rows": len(rows) - len(row_requests),
              "source_technical_exclusions": {q: r.get("exclusion_reason") for q, r in state["source"].items()
                                              if r.get("behavior_label") is None}}
    if dry_run:
        return {**report, "status": "dry_run", "n_requests_this_invocation": 0}
    work.mkdir(exist_ok=True)
    if not manifest.exists():
        write_json_atomic(manifest, identity, overwrite=False)
    if pending:
        attempt = create_versioned_run_dir(work / "attempts")
        write_jsonl(attempt / "requests.jsonl", pending)
        own_client = client is None
        client = client if client is not None else _create_client(cfg, identity["protocol"])
        try:
            result = cv.validate_batch(client, pending, output_path=attempt / "raw_results.jsonl",
                                       max_tokens=identity["protocol"]["max_tokens"],
                                       temperature=identity["protocol"]["temperature"],
                                       max_attempts=identity["protocol"]["max_attempts"],
                                       include_question=identity["protocol"]["include_question"],
                                       concurrency=cfg.claude.get("concurrency", 1))
        finally:
            if own_client:
                client.close()
        cache.update({r["episode_id"]: r for r in result["results"]})
    failed_questions, counts, labeled = {}, Counter(), []
    for row in rows:
        ep = episodes[row["question_id"]]
        request = row_requests.get(replay.key(row))
        if request is None:
            label = generation_label(ep, row["solver_attempt_2"])
        else:
            r = cache[request]
            behavior = r.get("behavior_label")
            if behavior is not None and r.get("claude_model") not in identity["protocol"]["models_returned"]:
                raise ValueError("Claude returned a different model version from the source labeling run.")
            label = {"solver_response": RESPONSES.get(behavior, "claude_error"),
                     "behavior_label": behavior, "adopted": None if behavior is None else int(behavior == "adoption"),
                     "label_source": r.get("label_source"), "claude_request_key": request,
                     "claude_exclusion_reason": r.get("exclusion_reason")}
            if behavior is None:
                failed_questions[row["question_id"]] = r.get("exclusion_reason")
        counts[label.get("behavior_label") or "error"] += 1
        labeled.append({**row, **label})
    failed_requests = sum(r.get("behavior_label") not in cv.VALID_LABELS for r in cache.values())
    stopping = bool(failed_questions) and cfg.claude.get("generated_errors", "stop") == "stop"
    report.update(label_counts=dict(counts), excluded_questions=failed_questions,
                  n_analysis_episodes=len(episodes) - len(failed_questions),
                  n_requests_this_invocation=len(pending), n_failed_requests=failed_requests,
                  n_pending_requests=failed_requests if stopping else 0,
                  status="incomplete" if stopping else "complete_with_exclusions" if failed_questions else "complete")
    if stopping:
        write_json_atomic(work / "pending_errors.json", report)
        raise ValueError(f"Claude failed on {len(failed_questions)} episodes; audits saved. Repeat --label-only to retry.")
    with staged_output(final) as stage:
        write_jsonl(stage / "per_row.jsonl", labeled)
        write_json_atomic(stage / "summary.json", report)
        write_json_atomic(stage / "identity.json", identity)
        write_json_atomic(stage / "complete.json", {"files": file_facts(stage), "audit_files": file_facts(work)})
    return report


def load_labeled_rows(run: Path) -> tuple[list[dict], dict]:
    final = Path(run) / FINAL
    if not (final / "complete.json").exists():
        raise ValueError("Claude judgments are not complete; run scripts/causal_replay.py --label-only RUN first.")
    complete = json.loads((final / "complete.json").read_text())
    if complete["files"] != file_facts(final, exclude_names={"complete.json"}):
        raise ValueError("Published Claude label artifacts changed.")
    current_identity = _identity(Path(run))["identity"]
    if json.loads((final / "identity.json").read_text()) != current_identity:
        raise ValueError("Claude labels belong to different generations, inputs or protocol.")
    manifest = Path(run) / "claude_labeling/manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text()) != current_identity:
        raise ValueError("Claude labeling audit manifest changed or is missing.")
    if complete["audit_files"] != file_facts(Path(run) / "claude_labeling"):
        raise ValueError("Claude labeling audit files changed.")
    return read_jsonl(final / "per_row.jsonl"), json.loads((final / "summary.json").read_text())
