"""Combine Claude validation runs over one growing cohort and summarize the result.

A cohort is labeled in batches (100, +400, +500, plus retries of technical errors),
each in its own run directory.  ``combine_runs`` joins them by ``episode_id``: a later
run may replace an earlier row only when that row ended in a technical error, so no
episode is ever labeled twice silently.  ``milestone_summary`` reports the four
behavioral labels, technical errors and Critic-refusal metadata for the combined rows.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from mas_sae.data.musique import hop_group
from mas_sae.evaluation.claude_validator import VALID_LABELS, is_technical_error
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.records import read_jsonl


def combine_runs(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Return ``(rows, sources, totals)`` for the runs in order.

    ``sources`` maps episode_id to the unmodified source record (read from each run's
    ``episodes_path`` after checking its sha256).  ``totals`` holds billed tokens and
    wall-clock time over every row of every run, superseded rows included.
    """
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    totals = {"input_tokens": 0, "output_tokens": 0, "wall_clock_seconds": 0.0, "runs": []}
    for run_dir in map(Path, run_dirs):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        episodes_path = Path(summary["episodes_path"])
        if sha256_file(episodes_path) != summary["episodes_sha256"]:
            raise ValueError(f"{episodes_path} no longer matches the sha256 recorded by {run_dir}")
        source = {r["episode_id"]: r for r in read_jsonl(episodes_path)}
        replaced = 0
        for row in read_jsonl(run_dir / "raw_results.jsonl"):
            eid = row["episode_id"]
            if eid in rows:
                if not is_technical_error(rows[eid]):
                    raise ValueError(f"{eid} already has a result ({rows[eid]['exclusion_reason'] or rows[eid]['behavior_label']}); "
                                     f"{run_dir} would label it twice")
                replaced += 1
            rows[eid] = row
            sources[eid] = source[eid]
            totals["input_tokens"] += row["input_tokens"]
            totals["output_tokens"] += row["output_tokens"]
        totals["wall_clock_seconds"] += summary.get("wall_clock_seconds") or 0.0
        totals["runs"].append({"run_dir": str(run_dir), "episodes_path": str(episodes_path),
                               "episodes_sha256": summary["episodes_sha256"],
                               "git_commit": summary.get("git_commit"),
                               "n_rows": summary["n_episodes"], "replaced_technical_errors": replaced})
    return list(rows.values()), sources, totals


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(r["behavior_label"] or "error" for r in rows)
    return {label: c[label] for label in (*VALID_LABELS, "error")}


def critic_refusal(source: dict[str, Any]) -> bool:
    """Critic refused or rejected the premise, per the behavior metadata already on the record."""
    label = source.get("label") or {}
    return bool(label.get("critic_refusal_flag") or label.get("critic_premise_rejection_flag"))


def milestone_summary(rows: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Four-label counts, Retention rate, technical errors and Critic metadata for combined rows."""
    counts = _counts(rows)
    eligible = counts["adoption"] + counts["retention"]
    refusal = {r["episode_id"] for r in rows if critic_refusal(sources[r["episode_id"]])}
    return {
        "n_episodes": len(rows),
        "n_questions": len({r["question_id"] for r in rows}),
        "label_counts": counts,
        "label_sources": dict(Counter(r["label_source"] for r in rows if r["behavior_label"])),
        "technical_errors": dict(Counter(r["exclusion_reason"] for r in rows if is_technical_error(r))),
        "data_errors": dict(Counter(r["exclusion_reason"] for r in rows
                                    if r["behavior_label"] is None and not is_technical_error(r))),
        "binary_eligible": eligible,
        "retention_pct": round(100 * counts["retention"] / eligible, 2) if eligible else None,
        "critic_noncommittal": sum(bool(sources[r["episode_id"]].get("critic_noncommittal")) for r in rows),
        "critic_refusal_or_premise_rejection": len(refusal),
        "adoption_with_critic_refusal": sum(r["behavior_label"] == "adoption" and r["episode_id"] in refusal
                                            for r in rows),
        "retried_episodes": sum(r.get("claude_attempts", 1) > 1 for r in rows),
        "by_hop": {hop: _counts([r for r in rows if hop_group(r["question_id"]) == hop])
                   for hop in sorted({hop_group(r["question_id"]) for r in rows})},
    }
