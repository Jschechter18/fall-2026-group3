"""Claude replay uses semantic cohorts and never requires API credentials on GPU workers."""
import csv
import dataclasses
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from mas_sae.causal import distributed, replay, semantic
from mas_sae.evaluation import claude_validator as cv
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.experiments.records import read_jsonl, write_jsonl
from mas_sae.models.claude import ClaudeResult
from mas_sae.probe.feature_selection import ActivationBasis
from repo_paths import REPO_ROOT

MODEL = "claude-sonnet-4-5-20250929"


def result(label="out", stop="end_turn"):
    return ClaudeResult(text=json.dumps({"label": label, "reason": "semantic fixture"}), model=MODEL,
                        input_tokens=10, output_tokens=10, task="labeling", stop_reason=stop)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    episodes = []
    for i, (label, answer) in enumerate(zip(["adoption", "retention", "agreement", "out", None],
                                          ["Lyon", "Paris", "Lyon", "Berlin", "Rome"])):
        episodes.append({"question_id": f"q{i}", "episode_id": f"q{i}__natural__candidates",
                         "source_split": "intervention", "partition": "intervention", "question": "Which city?",
                         "solver_attempt_1": "Paris", "critic_advocated_answer": "Lyon",
                         "solver_attempt_2": answer, "critic_feedback": "Lyon", "critic_noncommittal": False,
                         "critic_condition": "natural", "a2_presentation": "candidates", "candidate_order": "a1_first",
                         "attempt2_activation_index": i})
    write_jsonl(tmp_path / "interactions.jsonl", episodes)
    labels = []
    for ep, label in zip(episodes, ["adoption", "retention", "agreement", "out", None]):
        client = MagicMock()
        client.generate.return_value = result(label or "out", "end_turn" if label else "max_tokens")
        labels.append(cv.validate_episode(client, ep))
    settings = {"model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "client": "ClaudeClient",
                "effort": None, "max_tokens": 512, "temperature": 0., "max_attempts": 3,
                "prompt_version": cv.prompt_version(True, "candidates"),
                "prompt_sha256": cv.prompt_sha256(True, "candidates"), "a2_presentation": "candidates",
                "episodes_sha256": sha256_file(tmp_path / "interactions.jsonl"),
                "partition": "intervention", "limit": None}
    (source / "run_settings.json").write_text(json.dumps(settings))
    (source / "summary.json").write_text(json.dumps({**settings, "n_episodes": len(labels),
                                                     "models_returned": [MODEL]}))
    write_jsonl(source / "raw_results.jsonl", labels)
    cv.export_binary_labels(labels, source / "binary_labels.csv")
    cv.export_exclusions(labels, source / "exclusions.jsonl")
    fields = ["question_id", "partition", "behavior_label", "label_source", "prompt_version",
              "eligible_primary", "primary_target"]
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in labels:
            if row["behavior_label"] is not None:
                writer.writerow({k: row[k] for k in fields})
    (tmp_path / "features.json").write_text("{}")
    cfg = replay.ReplayConfig.from_dict({
        "capstone_data_root": str(tmp_path), "interactions_jsonl": "interactions.jsonl", "labels_csv": "labels.csv",
        "feature_selection_summary": "features.json", "label_policy": "claude", "partition": "intervention",
        "layer": 1, "saes": {"S": {"kind": "activations", "path": "center.npy"}},
        "reconstruction": False, "error_preserving": True, "n_random_draws": 1,
        "claude": {"source_run": "source", "source_errors": "exclude", "generated_errors": "stop",
                   "concurrency": 1}, "n_bootstrap": 10, "n_permutations": 20,
    })
    return cfg


def fake_generation(comp, ep, cond, control_token=None):
    answer = (ep["stored_solver_attempt_2"] if cond["experiment"] == replay.BASELINE
              else "Marseille" if cond["condition"].startswith("targeted") else "Berlin")
    return {**{k: cond.get(k) for k in ("experiment", "condition", "draw", "mode", "strength", "fraction", "edit_norm",
                                      "norm_edit_policy", "probe_direction", "direction_seed")},
            "question_id": ep["question_id"], "solver_attempt_2": answer,
            "stored_solver_attempt_2": ep["stored_solver_attempt_2"],
            "stored_solver_response": ep["stored_solver_response"], "stored_adopted": ep["stored_adopted"],
            **semantic.generation_label(ep, answer), "first_token_id": 10,
            "first_token_logprob": -1., "control_first_token_logprob": -1.,
            "input_ids_sha256": "fixture", "prompt_matches_stored": True,
            "removed_rel_norm": 0.01, "raw_edit_rel_norm": .02, "edit_scale": .5,
            "n_ablated_active": 1, "recon_rel_error": 0., "seconds": .01, "reached_token_budget": False}


def components():
    return SimpleNamespace(features={"S": {"targeted": [0], "random": [[1]]}},
                           saes={"S": ActivationBasis(np.zeros(2))}, site="layer", info={})


def bundle(cfg, folder, monkeypatch):
    comp = components()
    monkeypatch.setattr(replay, "generate", fake_generation)
    # an episode's edited conditions go through generate_episode (#214): one fake generation each
    monkeypatch.setattr(replay, "generate_episode", lambda comp, items, first: (
        fake_generation(comp, ep, cond) for ep, cond in items))
    replay.replay(cfg, replay.load_episodes(cfg), comp, folder)
    return folder


def client():
    c = MagicMock()
    c.generate.side_effect = lambda prompt, **kwargs: result("out" if "Marseille" in prompt else "retention")
    return c


def test_claude_cohort_ignores_old_rules_and_audits_source_failures(inputs, monkeypatch):
    monkeypatch.setattr(replay.bl, "classify_candidate", MagicMock(side_effect=AssertionError("old rules called")))
    episodes = replay.load_episodes(inputs).set_index("question_id")
    assert set(episodes.index) == {"q0", "q1"}
    assert episodes.loc["q0", "stored_adopted"] == 1 and episodes.loc["q1", "stored_adopted"] == 0
    with pytest.raises(ValueError, match="technical failures"):
        semantic.read_source(dataclasses.replace(inputs, claude={**inputs.claude, "source_errors": "stop"}))
    provenance = replay.provenance(inputs, None, components())
    assert provenance["label_policy"] == cv.prompt_version(True, "candidates")
    assert "source/raw_results.jsonl" in provenance["input_sha256"]


def test_rule_package_or_changed_source_is_refused(inputs):
    path = Path(inputs.capstone_data_root) / "labels.csv"
    path.write_text(path.read_text().replace("claude", "rule"))
    with pytest.raises(ValueError, match="differs from the source"):
        replay.load_episodes(inputs)
    path = Path(inputs.capstone_data_root) / "interactions.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="different intervention"):
        semantic.read_source(inputs)


def test_workers_keep_changed_answers_pending_and_preflight_fails_closed(inputs, tmp_path, monkeypatch):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    rows = read_jsonl(run / "per_row.jsonl")
    assert all(r["adopted"] is None and r["solver_response"] == "pending_claude"
               for r in rows if r["experiment"] != replay.BASELINE)
    assert json.loads((run / "replay_check.json").read_text())["passed"]
    with pytest.raises(ValueError, match="judgments are not complete"):
        replay.analyse(run)
    manifest = pd.read_csv(run / "episodes.csv")
    baseline = pd.DataFrame([r for r in rows if r["experiment"] == replay.BASELINE])
    baseline.loc[0, ["solver_attempt_2", "solver_response", "adopted", "label_source"]] = ["changed", "pending_claude", None, "pending_claude"]
    check = replay.replay_check(baseline, manifest, .98)
    assert not check["passed"] and check["pending_claude_labels"] == 1
    assert check["mismatch_counts"]["pending_claude"] == 1


def test_coordinator_deduplicates_and_preserves_condition_identity(inputs, tmp_path, monkeypatch):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    api = client()
    report = semantic.label_run(run, client=api)
    assert api.generate.call_count == 2  # six rows, but only two distinct changed semantic inputs
    assert report["n_reused_stored_rows"] == 2 and report["source_technical_exclusions"] == {"q4": "incomplete_response:max_tokens"}
    assert report["n_pending_requests"] == 0 and report["status"] == "complete"
    labeled, _ = semantic.load_labeled_rows(run)
    assert len({replay.key(r) for r in labeled}) == len(labeled) == 6
    assert all(r["behavior_label"] == "out" for r in labeled if r["condition"].startswith("targeted"))
    table = replay.analyse(run)
    assert set(table.N) == {2}
    summary = json.loads((run / "causal_replay_summary.json").read_text())
    assert summary["validated"] and summary["semantic_labeling"]["label_counts"]["out"] == 2
    repeated = semantic.label_run(run, client=api)
    assert repeated["n_requests_this_invocation"] == 0
    assert api.generate.call_count == 2  # completed labeling is idempotent
    raw = run / "per_row.jsonl"
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="different generations"):
        semantic.load_labeled_rows(run)


def test_dry_run_never_constructs_a_client_or_writes_outputs(inputs, tmp_path, monkeypatch):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    monkeypatch.setattr(semantic, "_create_client", MagicMock(side_effect=AssertionError("network")))
    report = semantic.label_run(run, dry_run=True)
    assert report["n_pending_requests"] == 2
    assert not (run / "claude_labeling").exists() and not (run / "claude_labeling.lock").exists()


def test_coordinator_cli_dry_run_uses_the_frozen_run_config(inputs, tmp_path, monkeypatch, capsys):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    script = REPO_ROOT / "scripts/causal_replay.py"
    monkeypatch.setattr(semantic, "_create_client", MagicMock(side_effect=AssertionError("network")))
    monkeypatch.setattr(sys, "argv", [str(script), "--label-only", str(run), "--dry-run"])
    runpy.run_path(str(script), run_name="__main__")
    assert '"n_pending_requests": 2' in capsys.readouterr().out
    assert not (run / "claude_labeling").exists()


def test_failed_judgments_retry_without_relabeling_successes(inputs, tmp_path, monkeypatch):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    api = client()
    api.generate.side_effect = lambda prompt, **kwargs: result("out", "max_tokens") if "Marseille" in prompt else result("retention")
    with pytest.raises(ValueError, match="Repeat --label-only"):
        semantic.label_run(run, client=api)
    assert not (run / semantic.FINAL).exists()
    pending = json.loads((run / "claude_labeling/pending_errors.json").read_text())
    assert set(pending["excluded_questions"]) == {"q0", "q1"}
    good = client()
    report = semantic.label_run(run, client=good)
    assert good.generate.call_count == 1 and report["n_requests_this_invocation"] == 1
    assert report["n_pending_requests"] == 0
    assert not report["excluded_questions"]


def test_interrupted_api_run_reuses_written_prefix(inputs, tmp_path, monkeypatch):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    bad = client()
    bad.generate.side_effect = [result("out"), ClientError({"Error": {"Code": "ExpiredTokenException", "Message": "expired"}}, "InvokeModel")]
    with pytest.raises(ClientError):
        semantic.label_run(run, client=bad)
    good = client()
    semantic.label_run(run, client=good)
    assert good.generate.call_count == 1


def test_distributed_generation_and_central_labeling_match_serial(inputs, tmp_path, monkeypatch):
    serial = bundle(inputs, tmp_path / "serial", monkeypatch)
    semantic.label_run(serial, client=client())
    expected = replay.analyse(serial)
    comp = components()
    preflight = tmp_path / "preflight"
    episodes = replay.load_episodes(inputs)
    replay.replay(inputs, episodes, comp, preflight, preflight_only=True)
    manifest = distributed.prepare_distributed_run(preflight, num_shards=2)
    monkeypatch.setattr(replay, "load_components", lambda *args: comp)
    monkeypatch.setattr(semantic, "_create_client", MagicMock(side_effect=AssertionError("worker made API call")))
    for shard in range(2):
        distributed.run_shard(manifest, shard)
    merged = distributed.reduce_shards(manifest)
    assert not (merged / "causal_table.csv").exists()
    semantic.label_run(merged, client=client())
    pd.testing.assert_frame_equal(expected, replay.analyse(merged))


def test_protocol_change_invalidates_partial_label_cache(inputs, tmp_path, monkeypatch):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    bad = client()
    bad.generate.return_value = result("out", "max_tokens")
    bad.generate.side_effect = None
    with pytest.raises(ValueError):
        semantic.label_run(run, client=bad)
    path = Path(inputs.capstone_data_root) / "source/run_settings.json"
    settings = json.loads(path.read_text())
    settings["max_tokens"] = 1024
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="protocol"):
        semantic.label_run(run, client=client())


def test_explicit_generated_error_exclusion_is_paired_across_all_conditions(inputs, tmp_path, monkeypatch):
    cfg = dataclasses.replace(inputs, claude={**inputs.claude, "generated_errors": "exclude_episode"})
    run = bundle(cfg, tmp_path / "run", monkeypatch)
    rows = read_jsonl(run / "per_row.jsonl")
    # Only q0's target creates this failing semantic input; q1's target remains valid.
    for row in rows:
        if row["question_id"] == "q0" and row["condition"].startswith("targeted"):
            row["solver_attempt_2"] = "Rome"
    write_jsonl(run / "per_row.jsonl", rows)
    api = client()
    # Prompt formatting is intentionally delegated to the production validator.
    def classify(prompt, **kw):
        return result("out", "max_tokens") if "Rome" in prompt else result("out")
    api.generate.side_effect = classify
    report = semantic.label_run(run, client=api)
    assert set(report["excluded_questions"]) == {"q0"} and report["n_analysis_episodes"] == 1
    table = replay.analyse(run)
    assert set(table.N) == {1}
    summary = json.loads((run / "causal_replay_summary.json").read_text())
    assert summary["n_episodes_planned"] == 2 and summary["semantic_labeling"]["excluded_questions"]


def test_completed_audit_tampering_is_detected(inputs, tmp_path, monkeypatch):
    run = bundle(inputs, tmp_path / "run", monkeypatch)
    semantic.label_run(run, client=client())
    audit = next((run / "claude_labeling/attempts").glob("*/raw_results.jsonl"))
    audit.write_text(audit.read_text() + "\n")
    with pytest.raises(ValueError, match="audit files changed"):
        semantic.load_labeled_rows(run)


def test_prompt_changes_fail_before_any_coordinator_api_call(inputs, monkeypatch):
    monkeypatch.setattr(cv, "LABELING_SYSTEM", cv.LABELING_SYSTEM + " different")
    with pytest.raises(ValueError, match="prompt differs"):
        semantic.read_source(inputs)


def test_claude_package_preparation_never_invokes_rule_builder(inputs, tmp_path, monkeypatch):
    from mas_sae.causal import prepare
    root = Path(inputs.capstone_data_root)
    (root / "labels.csv").unlink()
    cfg = dataclasses.replace(inputs, partition="intervention", interactions_jsonl="intervention/interactions.jsonl",
                              input_preparation={"collection_results": "."})
    (root / "intervention").mkdir()
    (root / "interactions.jsonl").rename(root / "intervention/interactions.jsonl")
    calls = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(replay, "load_episodes", lambda cfg: pd.DataFrame({"question_id": ["q"]}))
    monkeypatch.setattr(prepare, "sae_files", lambda *args: [])
    monkeypatch.setattr(replay, "feature_sets", lambda *args: {})
    prepare.prepare_inputs(cfg)
    assert len(calls) == 1 and calls[0][1].endswith("export_final_labels_for_probes.py")
    assert "--claude-runs" in calls[0] and not any("prepare_behavior_v01" in str(a) for a in calls[0])
