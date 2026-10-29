"""Tests for combining Claude validation runs into milestones."""

import json

import pytest

pytest.importorskip("datasets")  # mas_sae.data.musique (hop_group) imports it

from mas_sae.evaluation.claude_milestones import combine_runs, milestone_summary
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.records import write_jsonl


def _source(qid, refusal=False, noncommittal=False):
    return {"episode_id": f"{qid}__natural", "question_id": qid, "critic_noncommittal": noncommittal,
            "label": {"critic_refusal_flag": refusal, "critic_premise_rejection_flag": False}}


def _row(qid, label=None, reason=None, tokens=10):
    return {"episode_id": f"{qid}__natural", "question_id": qid, "behavior_label": label,
            "label_source": "claude" if label else None, "exclusion_reason": reason,
            "input_tokens": tokens, "output_tokens": 1}


def _run(tmp_path, name, sources, rows):
    episodes = tmp_path / f"{name}_episodes.jsonl"
    write_jsonl(episodes, sources)
    run = tmp_path / name
    run.mkdir()
    write_jsonl(run / "raw_results.jsonl", rows)
    (run / "summary.json").write_text(json.dumps({
        "episodes_path": str(episodes), "episodes_sha256": sha256_file(episodes),
        "n_episodes": len(rows), "wall_clock_seconds": 2.0, "git_commit": "abc"}))
    return run


def test_retry_replaces_only_technical_errors(tmp_path):
    src = [_source("2hop__a", refusal=True), _source("3hop1__b")]
    first = _run(tmp_path, "first", src, [_row("2hop__a", "adoption"), _row("3hop1__b", reason="parse_error")])
    retry = _run(tmp_path, "retry", src, [_row("3hop1__b", "retention")])

    rows, sources, totals = combine_runs([first, retry])
    summary = milestone_summary(rows, sources)

    assert summary["label_counts"] == {"adoption": 1, "retention": 1, "agreement": 0, "out": 0, "error": 0}
    assert summary["retention_pct"] == 50.0
    assert summary["adoption_with_critic_refusal"] == 1
    assert summary["by_hop"]["3hop"]["retention"] == 1
    assert totals["input_tokens"] == 30  # superseded attempt still billed
    assert totals["runs"][1]["replaced_technical_errors"] == 1


def test_relabeling_a_labeled_episode_is_refused(tmp_path):
    src = [_source("2hop__a")]
    first = _run(tmp_path, "first", src, [_row("2hop__a", "agreement")])
    again = _run(tmp_path, "again", src, [_row("2hop__a", "adoption")])
    with pytest.raises(ValueError, match="label it twice"):
        combine_runs([first, again])


def test_changed_source_file_is_refused(tmp_path):
    run = _run(tmp_path, "first", [_source("2hop__a")], [_row("2hop__a", "adoption")])
    write_jsonl(tmp_path / "first_episodes.jsonl", [_source("2hop__z")])
    with pytest.raises(ValueError, match="no longer matches"):
        combine_runs([run])


def test_technical_and_data_errors_reported_separately(tmp_path):
    src = [_source("2hop__a"), _source("2hop__b"), _source("2hop__c", noncommittal=True)]
    rows = [_row("2hop__a", reason="api_error"), _row("2hop__b", reason="missing_field:solver_attempt_2"),
            dict(_row("2hop__c", "out"), label_source="rule:critic_noncommittal")]
    combined, sources, _ = combine_runs([_run(tmp_path, "r", src, rows)])
    summary = milestone_summary(combined, sources)
    assert summary["technical_errors"] == {"api_error": 1}
    assert summary["data_errors"] == {"missing_field:solver_attempt_2": 1}
    assert summary["label_sources"] == {"rule:critic_noncommittal": 1}
    assert summary["critic_noncommittal"] == 1


def test_legacy_schema_rejected_with_clear_message(tmp_path):
    legacy = {k: v for k, v in _row("2hop__a", "adoption").items() if k not in ("behavior_label", "label_source")}
    run = _run(tmp_path, "old", [_source("2hop__a")], [legacy])
    with pytest.raises(ValueError, match="predates behavior_label"):
        combine_runs([run])


def test_retry_with_a_different_interaction_rejected(tmp_path):
    first = _run(tmp_path, "first", [_source("2hop__a")], [_row("2hop__a", reason="parse_error")])
    changed = dict(_source("2hop__a"), solver_attempt_2="something else")
    retry = _run(tmp_path, "retry", [changed], [_row("2hop__a", "adoption")])
    with pytest.raises(ValueError, match="different interaction"):
        combine_runs([first, retry])


def test_unknown_hops_rejected_in_summary(tmp_path):
    run = _run(tmp_path, "r", [_source("musique:2hop__a")], [_row("musique:2hop__a", "adoption")])
    rows, sources, _ = combine_runs([run])
    with pytest.raises(ValueError, match="no parseable hop group"):
        milestone_summary(rows, sources)
