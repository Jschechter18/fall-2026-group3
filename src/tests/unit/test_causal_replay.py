"""Causal replay (issue #90): the splice hook, episode loading checks, and the run/resume/table mechanics."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch

from mas_sae.causal import offline, replay
from mas_sae.causal.smoke import smoke_components, tiny_features, tiny_saes
from mas_sae.evaluation import behavior_labels as bl


class Echo(torch.nn.Module):
    def __init__(self, as_tuple=False):
        super().__init__()
        self.as_tuple = as_tuple

    def forward(self, h):
        return (h, "cache") if self.as_tuple else h


def _host(as_tuple=False):
    m = torch.nn.Module()
    m.layer = Echo(as_tuple)
    return m


@pytest.mark.parametrize("as_tuple", [False, True])
def test_splice_changes_only_the_last_prefill_position(as_tuple):
    sae = tiny_saes()["TopK SAE"]
    m = _host(as_tuple)
    prefill, step = torch.randn(1, 5, 16), torch.randn(1, 1, 16)
    targeted = [1, 2, 3]
    with replay.SAESplice(m, "layer", sae, targeted) as sp:
        out = m.layer(prefill)
        later = m.layer(step)
    out, later = (out[0], later[0]) if as_tuple else (out, later)
    assert torch.equal(out[:, :-1], prefill[:, :-1]) and torch.equal(later, step)
    x = prefill[0, -1].numpy()[None]
    expected = offline.decode(sae, offline.ablate(offline.encode(sae, x), targeted))[0]
    assert np.allclose(out[0, -1].numpy(), expected, atol=1e-6)
    assert sp.calls == 2 and set(sp.stats) == {"n_active", "recon_rel_error", "n_ablated_active",
                                                "removed_rel_norm"}
    assert m.layer._forward_hooks == {}                 # hook removed on exit


def test_splice_without_ablation_is_the_plain_reconstruction():
    sae = tiny_saes()["Pretrained SAE"]
    m = _host()
    h = torch.randn(1, 3, 16)
    with replay.SAESplice(m, "layer", sae, []) as sp:
        out = m.layer(h)
    assert torch.allclose(out[0, -1], sae.decoder(sae.encoder(h[:, -1]))[0])
    assert "n_ablated_active" not in sp.stats


def test_splice_refuses_batches_single_tokens_and_unused_hooks():
    sae, m = tiny_saes()["L1 SAE"], _host()
    with pytest.raises(ValueError, match="single-prompt prefill"):
        with replay.SAESplice(m, "layer", sae):
            m.layer(torch.randn(1, 1, 16))
    with pytest.raises(ValueError, match="single-prompt prefill"):
        with replay.SAESplice(m, "layer", sae):
            m.layer(torch.randn(2, 4, 16))
    with pytest.raises(RuntimeError, match="never ran"):
        with replay.SAESplice(m, "layer", sae):
            pass
    with pytest.raises(ValueError, match="not found"):
        replay.SAESplice(m, "nope", sae)


# ---- episodes and feature sets

def _episode(qid, a1="Paris", critic="Lyon", a2="Lyon", partition="validation"):
    return {"question_id": qid, "source_split": "validation", "question": f"q {qid}?",
            "solver_attempt_1": a1, "critic_condition": "natural", "critic_advocated_answer": critic,
            "critic_feedback": f"The reviewer disagrees with your previous answer. {critic}",
            "critic_noncommittal": False, "solver_attempt_2": a2, "partition": partition}


def _write_inputs(tmp_path, episodes, label_overrides=None, partitions=None):
    rows = []
    for ep in episodes:
        label = bl.classify_candidate(ep, bl.V012_QC)
        label.update((label_overrides or {}).get(ep["question_id"], {}))
        rows.append({"question_id": ep["question_id"], "source_split": "validation",
                     "partition": (partitions or {}).get(ep["question_id"], ep["partition"]),
                     "solver_behavior": "x", "label": label})
    bl.write_labels(tmp_path / "labels.csv", rows)
    (tmp_path / "interactions.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))


def _cfg(tmp_path, **kw):
    base = dict(capstone_data_root=tmp_path, interactions_jsonl="interactions.jsonl", labels_csv="labels.csv",
                feature_selection_summary="fs.json", layer=33, partition="validation", n_random_draws=2,
                n_bootstrap=50, n_permutations=200, output_dir=tmp_path / "out",
                on_replay_check_fail="mark_unvalidated")         # a random tiny model cannot reproduce collection
    return replay.ReplayConfig(**{**base, **kw})


def _episodes(n=6):
    eps = [_episode(f"q{i}") for i in range(n)]
    eps[1]["solver_attempt_2"] = "Paris"                  # retained A1
    eps[2]["solver_attempt_2"] = "Marseille"              # third answer
    eps[3]["critic_advocated_answer"] = "Paris"           # no conflict: not eligible
    return eps


def test_load_episodes_keeps_eligible_rows_in_a_seeded_order(tmp_path):
    _write_inputs(tmp_path, _episodes())
    df = replay.load_episodes(_cfg(tmp_path))
    assert sorted(df.question_id) == ["q0", "q1", "q2", "q4", "q5"]
    assert df.set_index("question_id").loc[["q0", "q1", "q2"], "stored_adopted"].tolist() == [1, 0, 0]
    assert df.question_id.tolist() == replay.load_episodes(_cfg(tmp_path)).question_id.tolist()
    assert replay.load_episodes(_cfg(tmp_path, limit=2)).question_id.tolist() == df.question_id.tolist()[:2]


@pytest.mark.parametrize("problem, match", [
    ("relabel", "relabeling the stored A2"),
    ("duplicate", "more than one episode"),
    ("missing", "no stored episode"),
    ("condition", "Natural-condition"),
    ("version", "rule_version"),
])
def test_load_episodes_refuses_inconsistent_inputs(tmp_path, problem, match):
    eps = _episodes()
    overrides = {}
    if problem == "relabel":
        overrides = {"q1": {"solver_response": "adopted_critic", "primary_target": 1}}
    if problem == "version":
        overrides = {e["question_id"]: {"rule_version": "behavior_v0.1.1_candidate"} for e in eps}
    _write_inputs(tmp_path, eps, overrides)
    lines = (tmp_path / "interactions.jsonl").read_text().splitlines()
    if problem == "duplicate":
        lines.append(lines[0])
    if problem == "missing":
        lines = lines[1:]
    if problem == "condition":
        lines[0] = lines[0].replace('"natural"', '"controlled"')
    (tmp_path / "interactions.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=match):
        replay.load_episodes(_cfg(tmp_path))


def _summary(tmp_path, sae_hash="h", layer=33):
    s = {"config": {"layer": layer},
         "provenance": {"sae_sha256": {"TopK SAE": {"ckpt": "h"}}},
         "saes": {"TopK SAE": {"targeted_features": [1, 2], "random_control_draws": [[3, 4], [5, 6], [7, 8]]}}}
    (tmp_path / "fs.json").write_text(json.dumps(s))
    return {"TopK SAE": {"ckpt": sae_hash}}


def test_feature_sets_come_from_the_summary_of_the_same_saes(tmp_path):
    cfg = _cfg(tmp_path, saes={"TopK SAE": {"kind": "custom", "path": "x"}})
    assert replay.feature_sets(cfg, _summary(tmp_path)) == {"TopK SAE": {"targeted": [1, 2], "random": [[3, 4], [5, 6]]}}
    with pytest.raises(ValueError, match="differ"):
        replay.feature_sets(cfg, _summary(tmp_path, sae_hash="other"))
    _summary(tmp_path, layer=25)
    with pytest.raises(ValueError, match="layer 25"):
        replay.feature_sets(cfg, {"TopK SAE": {"ckpt": "h"}})
    _summary(tmp_path)
    with pytest.raises(ValueError, match="random draws"):
        replay.feature_sets(_cfg(tmp_path, saes=cfg.saes, n_random_draws=4), {"TopK SAE": {"ckpt": "h"}})


def test_config_requires_paths_and_valid_saes(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("interactions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\nlayer: 33\n")
    with pytest.raises(ValueError, match="capstone_data_root"):
        replay.ReplayConfig.from_yaml(p)
    paths = "capstone_data_root: .\ninteractions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\n"
    p.write_text(paths)
    with pytest.raises(ValueError, match="layer"):                  # no silent default layer
        replay.ReplayConfig.from_yaml(p)
    p.write_text(paths + "layer: -1\n")
    with pytest.raises(ValueError, match="nonnegative"):
        replay.ReplayConfig.from_yaml(p)
    p.write_text(paths + "layer: 33\nsaes:\n  Baseline:\n    kind: custom\n    path: x\n")
    with pytest.raises(ValueError, match="Baseline"):
        replay.ReplayConfig.from_yaml(p)


def test_committed_configs_load():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    real = replay.ReplayConfig.from_yaml(repo / "configs/causal_replay_layer33.yaml")
    smoke = replay.ReplayConfig.from_yaml(repo / "configs/causal_replay_smoke_validation.yaml")
    assert set(real.saes) == {"TopK SAE", "L1 SAE", "Pretrained SAE"} == set(smoke.saes)
    assert (real.partition, real.layer, real.solver["revision"]) == ("intervention", 33, "093f9f388b31de276ce2de164bdc2081324b9767")
    assert real.feature_selection_summary == smoke.feature_selection_summary
    assert (real.on_replay_check_fail, smoke.on_replay_check_fail) == ("stop", "mark_unvalidated")
    for layer in (33, 25):
        steer = replay.ReplayConfig.from_yaml(repo / f"configs/causal_steering_layer{layer}.yaml")
        assert (steer.layer, steer.ablation, steer.steer_strengths, steer.n_random_draws) == (layer, False, [-2, 2, 5], 7)
        assert all(f"layer_{layer}" in spec["path"] for spec in steer.saes.values())
        assert Path(steer.feature_reference).parent == Path(steer.feature_selection_summary).parent
    assert smoke.steer_strengths and smoke.ablation


@pytest.mark.parametrize("change, match", [
    (lambda f: f["TopK SAE"].update(targeted=[1, 1, 2]), "repeat"),
    (lambda f: f["TopK SAE"].update(targeted=[1, 2, 64]), r"\[0, 64\)"),
    (lambda f: f["TopK SAE"].update(targeted=[1, 2, -1]), r"\[0, 64\)"),
    (lambda f: f["TopK SAE"].update(targeted=[1, 2, 3.0]), "integers"),
    (lambda f: f["TopK SAE"].update(targeted=[]), "integers"),
    (lambda f: f["TopK SAE"]["random"][0].pop(), "targeted set has 3"),
    (lambda f: f["TopK SAE"]["random"][1].__setitem__(0, f["TopK SAE"]["targeted"][0]), "includes targeted"),
    (lambda f: f.pop("L1 SAE"), "Feature sets are for"),
])
def test_validate_features_refuses_unfair_or_out_of_range_sets(change, match):
    saes = tiny_saes()
    feats = tiny_features(saes, n_random_draws=2)
    replay.validate_features(feats, saes)
    change(feats)
    with pytest.raises(ValueError, match=match):
        replay.validate_features(feats, saes)


# ---- end to end on the tiny model

def _smoke(tmp_path, layer=33, n_layers=None):
    _write_inputs(tmp_path, _episodes())
    _summary(tmp_path)
    cfg = _cfg(tmp_path, layer=layer)
    episodes = replay.load_episodes(cfg)
    paragraphs = {q: [{"idx": 0, "title": "T", "paragraph_text": "text"}] for q in episodes.question_id}
    saes = tiny_saes()
    comp = smoke_components(paragraphs, layer, saes, tiny_features(saes, n_random_draws=2), max_new_tokens=2,
                            n_layers=n_layers)
    return cfg, episodes, comp


N_CONDITIONS = 1 + 3 * ((2 + 2) + (1 + 2))      # plain + per SAE: reconstruction, error-preserving


def test_replay_writes_the_table_and_resumes_without_regenerating(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    run_dir = tmp_path / "out" / "run"

    replay.replay(cfg, episodes.head(3), comp, run_dir)
    per_row = run_dir / "per_row.jsonl"
    n_conditions = N_CONDITIONS
    assert len(per_row.read_text().splitlines()) == 3 * n_conditions

    replay.replay(cfg, episodes, comp, run_dir)                     # resume and extend
    assert len(per_row.read_text().splitlines()) == len(episodes) * n_conditions

    table = replay.analyse(run_dir)
    assert list(table.columns) == replay.TABLE_COLUMNS
    assert table.Condition.tolist()[:2] == ["No SAE (stored A2)", "No SAE (replayed A2)"]
    assert len(table) == 2 + 6 * 3 and (table.N == len(episodes)).all()
    assert table.loc[0, "Acceptance Rate (%)"] == pytest.approx(100 * episodes.stored_adopted.mean())
    assert set(table.loc[table.Condition == "No ablation", "Δ vs. SAE Control (pp)"]) == {"Reference"}
    for f in ("causal_table.csv", "per_row.csv", "causal_replay_summary.json", "causal_effects.svg", "episodes.csv"):
        assert (run_dir / f).exists()
    meta = json.loads((run_dir / "replay_config.json").read_text())
    assert meta["provenance"]["label_policy"] == bl.V012_QC.version

    with pytest.raises(ValueError, match="different config"):
        replay.replay(_cfg(tmp_path, seed=7), episodes, comp, run_dir)


def test_replay_at_another_layer_hooks_that_layer(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path, layer=5)
    assert comp.site == "model.language_model.layers.5"
    assert len(comp.model.model.language_model.layers) == 6
    run_dir = replay.replay(cfg, episodes.head(1), comp, tmp_path / "run")   # the hook must fire
    rows = replay._read_rows(run_dir / "per_row.jsonl")
    assert len(rows) == N_CONDITIONS and all(r["recon_rel_error"] is not None for r in rows
                                             if r["experiment"] != replay.BASELINE)
    meta = json.loads((run_dir / "replay_config.json").read_text())
    assert meta["config"]["layer"] == 5 and meta["provenance"]["site"] == comp.site


def test_resume_after_a_truncated_write_regenerates_the_lost_record(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    run_dir = replay.replay(cfg, episodes.head(2), comp, tmp_path / "run")
    per_row = run_dir / "per_row.jsonl"
    complete = per_row.read_text()
    last = complete.splitlines()[-1]
    per_row.write_text(complete[:-len(last) - 1] + last[:len(last) // 2])   # killed mid-write
    assert len(replay._read_rows(per_row)) == 2 * N_CONDITIONS - 1

    replay.replay(cfg, episodes.head(2), comp, run_dir)
    rows = replay._read_rows(per_row)
    assert per_row.read_text().endswith("\n") and len(rows) == 2 * N_CONDITIONS
    assert len({replay.key(r) for r in rows}) == 2 * N_CONDITIONS
    timeless = lambda r: {k: v for k, v in r.items() if k != "seconds"}
    assert timeless(json.loads(per_row.read_text().splitlines()[-1])) == timeless(json.loads(last))


def test_corrupt_interior_records_are_errors_not_skipped(tmp_path):
    path = tmp_path / "per_row.jsonl"
    path.write_text('{"a": 1}\n{"broken\n{"a": 2}\n')
    with pytest.raises(ValueError, match="line 2 .*corrupt"):
        replay._read_rows(path)


def test_resume_refuses_to_shrink_reorder_or_change_inputs(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    run_dir = replay.replay(cfg, episodes.head(2), comp, tmp_path / "run")
    manifest = (run_dir / "episodes.csv").read_text()
    with pytest.raises(ValueError, match="shrink"):
        replay.replay(cfg, episodes.head(1), comp, run_dir)
    with pytest.raises(ValueError, match="does not match"):
        replay.replay(cfg, episodes.iloc[[1, 0, 2]], comp, run_dir)
    assert (run_dir / "episodes.csv").read_text() == manifest            # manifest left as it was

    labels = tmp_path / "labels.csv"
    labels.write_text(labels.read_text() + "\n")                          # input edited in place
    with pytest.raises(ValueError, match="input_sha256"):
        replay.replay(cfg, episodes.head(2), comp, run_dir)


def test_generations_outside_the_manifest_are_refused(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    run_dir = replay.replay(cfg, episodes.head(1), comp, tmp_path / "run")
    per_row = run_dir / "per_row.jsonl"
    row = json.loads(per_row.read_text().splitlines()[0])
    per_row.write_text(per_row.read_text() + json.dumps({**row, "question_id": "elsewhere"}) + "\n")
    with pytest.raises(ValueError, match="outside this run"):
        replay.replay(cfg, episodes.head(1), comp, run_dir)


def test_analyse_uses_complete_episodes_and_averages_random_draws(tmp_path):
    feats = {"S": {"targeted": [1], "random": [[2], [3]]}}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "replay_config.json").write_text(json.dumps({
        "config": {"seed": 0, "n_bootstrap": 50, "n_permutations": 200, "layer": 33, "partition": "validation"},
        "conditions": replay.conditions(feats), "features": feats}))
    pd.DataFrame({"question_id": ["a", "b", "c"], "stored_solver_response": ["adopted_critic"] * 3}).to_csv(
        run_dir / "episodes.csv", index=False)
    adopted = {"no_sae": 1, "no_ablation": 1, "targeted_ablation": 0}
    rows = []
    for q, rand in (("a", [1, 0]), ("b", [1, 1]), ("c", [0, 0])):
        for c in replay.conditions(feats):
            if q == "c" and c["condition"] == "targeted_ablation":
                continue                                            # c is incomplete: dropped
            a = rand[c["draw"]] if c["condition"] == "random_ablation" else adopted[c["condition"]]
            rows.append({"question_id": q, "experiment": c["experiment"], "condition": c["condition"],
                         "draw": c["draw"], "adopted": a, "stored_adopted": 1, "solver_attempt_2": "x",
                         "stored_solver_attempt_2": "x", "a2_matches_stored": True,
                         "solver_response": "adopted_critic" if a else "retained_a1",
                         "reached_token_budget": False, "seconds": 0.1,
                         "n_ablated_active": None, "recon_rel_error": None})
    (run_dir / "per_row.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"truncated')
    t = replay.analyse(run_dir).set_index(["Experiment", "Condition"])
    assert (t.N == 2).all()
    assert t.loc[("S", "Random ablation"), "Acceptance Rate (%)"] == pytest.approx(75.0)
    assert t.loc[("S", "Random ablation"), "Δ vs. SAE Control (pp)"] == pytest.approx(-25.0)
    assert t.loc[("S", "Targeted ablation"), "Δ vs. SAE Control (pp)"] == pytest.approx(-100.0)
    s = json.loads((run_dir / "causal_replay_summary.json").read_text())
    assert s["n_episodes_complete"] == 2 and s["n_episodes_planned"] == 3
    assert s["replay_check"]["a2_exact_match_rate"] == 1.0


# ---- intervention variants and first-token telemetry

def test_error_preserving_splice_keeps_the_sae_error():
    sae = tiny_saes()["TopK SAE"]
    m = _host()
    h = torch.randn(1, 4, 16)
    with replay.SAESplice(m, "layer", sae, [], mode="error_preserving"):
        unchanged = m.layer(h)
    assert torch.equal(unchanged, h)                    # no ablation: the original activation, exactly
    targeted = [1, 2, 3]
    with replay.SAESplice(m, "layer", sae, targeted, mode="error_preserving") as sp:
        out = m.layer(h)
    x = h[0, -1].numpy()[None]
    z = offline.encode(sae, x)
    expected = x + offline.decode(sae, offline.ablate(z, targeted)) - offline.decode(sae, z)
    assert np.allclose(out[0, -1].numpy(), expected[0], atol=1e-5)
    removed = np.linalg.norm(offline.decode(sae, z) - offline.decode(sae, offline.ablate(z, targeted)))
    assert sp.stats["removed_rel_norm"] == pytest.approx(removed / np.linalg.norm(x), rel=1e-4)
    with pytest.raises(ValueError, match="mode"):
        replay.SAESplice(m, "layer", sae, [], mode="decode_all_steps")


def test_every_control_is_generated_before_what_is_compared_with_it():
    feats = tiny_features(tiny_saes(), n_random_draws=2)
    conds = replay.conditions(feats, error_preserving=True)
    seen = set()
    for c in conds:
        ctl = replay.control_key(c)
        assert ctl is None or ctl in seen
        seen.add((c["experiment"], c["condition"], c["draw"]))
    ep = [c for c in conds if c["mode"] == "error_preserving"]
    assert {replay.control_key(c) for c in ep} == {(replay.BASELINE, "no_sae", None)}
    assert not any(c["condition"] == "no_ablation" for c in ep)       # that would be the plain replay again
    assert len(replay.conditions(feats)) == 1 + 3 * (2 + 2)            # reconstruction only


def test_rows_record_the_first_token_and_the_control_token_log_prob(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    run_dir = replay.replay(cfg, episodes.head(2), comp, tmp_path / "run")
    rows = {replay.key(r): r for r in replay._read_rows(run_dir / "per_row.jsonl")}
    for k, r in rows.items():
        assert isinstance(r["first_token_id"], int) and r["first_token_logprob"] <= 0
        ctl = replay.control_key({"experiment": r["experiment"], "condition": r["condition"], "mode": r["mode"]})
        if ctl is None:
            assert r["control_first_token_logprob"] is None
            continue
        c = rows[(r["question_id"], *ctl)]
        # the control's own token, scored under this condition; equal to this row's own if it chose it too
        assert r["control_first_token_logprob"] <= r["first_token_logprob"] + 1e-6
        if c["first_token_id"] == r["first_token_id"]:
            assert r["control_first_token_logprob"] == pytest.approx(r["first_token_logprob"])
    plain = next(r for k, r in rows.items() if k[1] == replay.BASELINE)
    assert plain["mode"] is None and plain["prompt_matches_stored"] is None   # collection stored no prompt hash
    meta = json.loads((run_dir / "replay_config.json").read_text())
    assert meta["provenance"]["intervention"] == replay.INTERVENTION


# ---- replay check before the SAE conditions

class _ReturnsStored:
    """The tiny Solver (so hooks and telemetry are real) answering with each episode's stored A2."""

    def __init__(self, solver, answers):
        self.solver, self.answers = solver, answers

    def revise(self, question, *args):
        self.solver.revise(question, *args)
        return self.answers[question]

    def __getattr__(self, name):
        return getattr(self.solver, name)


def _faithful(comp):
    eps = {e["question"]: e["solver_attempt_2"] for e in _episodes()}
    comp.solver = _ReturnsStored(comp.solver, eps)
    return comp


def test_failed_replay_check_stops_before_any_sae_condition(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    cfg = dataclasses.replace(cfg, on_replay_check_fail="stop")
    with pytest.raises(RuntimeError, match="does not reproduce collection"):
        replay.replay(cfg, episodes, comp, tmp_path / "run")
    rows = replay._read_rows(tmp_path / "run" / "per_row.jsonl")
    assert len(rows) == len(episodes) and {r["experiment"] for r in rows} == {replay.BASELINE}
    check = json.loads((tmp_path / "run" / "replay_check.json").read_text())
    assert not check["passed"] and check["n_episodes"] == len(episodes)
    mism = pd.read_csv(tmp_path / "run" / "replay_check_mismatches.csv")
    assert len(mism) == len(episodes) - check["mismatch_counts"].get("exact_match", 0)


def test_preflight_then_resume_runs_the_sae_conditions_once_the_check_passes(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    cfg = dataclasses.replace(cfg, on_replay_check_fail="stop")
    comp = _faithful(comp)
    run_dir = replay.replay(cfg, episodes, comp, tmp_path / "run", preflight_only=True)
    check = json.loads((run_dir / "replay_check.json").read_text())
    assert check["passed"] and check["solver_response_match_rate"] == 1.0
    assert len(replay._read_rows(run_dir / "per_row.jsonl")) == len(episodes)
    replay.replay(cfg, episodes, comp, run_dir)
    assert len(replay._read_rows(run_dir / "per_row.jsonl")) == len(episodes) * N_CONDITIONS
    replay.analyse(run_dir)
    s = json.loads((run_dir / "causal_replay_summary.json").read_text())
    assert s["validated"] and s["intervention"]["name"] == replay.INTERVENTION


@pytest.mark.parametrize("replayed, response, prompt_ok, kind", [
    ("Lyon", "adopted_critic", None, "exact_match"),
    ("lyon.", "adopted_critic", None, "formatting_only"),
    ("Lyon, France", "adopted_critic", None, "text_differs_same_label"),
    ("Paris", "retained_a1", None, "label_changed"),
    ("Lyon", "adopted_critic", False, "prompt_differs"),
])
def test_replay_check_separates_kinds_of_mismatch(replayed, response, prompt_ok, kind):
    manifest = pd.DataFrame({"question_id": ["a", "b"], "stored_solver_response": ["adopted_critic"] * 2})
    rows = pd.DataFrame([{"question_id": q, "experiment": replay.BASELINE, "solver_attempt_2": a,
                          "stored_solver_attempt_2": "Lyon", "solver_response": r, "adopted": int(r == "adopted_critic"),
                          "prompt_matches_stored": p}
                         for q, a, r, p in (("a", "Lyon", "adopted_critic", True),
                                            ("b", replayed, response, prompt_ok))])
    check = replay.replay_check(rows, manifest, min_label_match=0.9)
    assert check["mismatch_counts"].get(kind) == (2 if kind == "exact_match" else 1)
    assert check["passed"] == (kind not in ("label_changed", "prompt_differs"))
    assert check["n_prompt_hashes_compared"] == (2 if prompt_ok is not None else 1)
    partial = replay.replay_check(rows.head(1), manifest, min_label_match=0.9)
    assert not partial["passed"] and "no plain replay yet" in partial["failures"][0]


def test_config_checks_the_replay_check_settings(tmp_path):
    p = tmp_path / "c.yaml"
    base = "capstone_data_root: .\ninteractions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\nlayer: 33\n"
    for extra, match in (("min_replay_label_match: 0\n", "min_replay_label_match"),
                         ("on_replay_check_fail: ignore\n", "on_replay_check_fail"),
                         ("error_preserving: 1\n", "error_preserving")):
        p.write_text(base + extra)
        with pytest.raises(ValueError, match=match):
            replay.ReplayConfig.from_yaml(p)


# ---- statistics

def test_targeted_vs_random_is_a_direct_paired_comparison():
    rng = np.random.default_rng(0)
    n = 200
    control = rng.integers(0, 2, n).astype(float)
    random = np.repeat(control[:, None], 4, axis=1)
    random[:10, 0] = 1 - random[:10, 0]                         # one draw moves a little
    targeted = control.copy()
    targeted[control == 1] = rng.integers(0, 2, int(control.sum()))   # targeted removes some adoption
    r = offline.targeted_vs_random(targeted, random, control, np.arange(n), 300, 500, 0)
    assert r["delta_pp"] == pytest.approx(100 * (targeted - random.mean(1)).mean())
    assert r["ci95_pp"][1] < 0 and r["p_value"] < 0.01
    lo, hi = r["ci95_pp_questions_and_draws"]
    assert lo <= r["delta_pp"] <= hi
    assert r["n_random_draws"] == 4 and len(r["random_draw_vs_control_pp"]) == 4
    assert r["draw_tail_fraction"] == pytest.approx(1 / 5)       # no draw moves as much as targeted


def test_holm_adjusts_in_input_order():
    assert offline.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert offline.holm([0.5]) == [0.5]


def test_analyse_writes_the_direct_comparison_and_diagnostics(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    run_dir = replay.replay(cfg, episodes, comp, tmp_path / "run")
    replay.analyse(run_dir)
    tvr = pd.read_csv(run_dir / "targeted_vs_random.csv")
    assert list(tvr.columns) == replay.TVR_COLUMNS and len(tvr) == 6
    assert tvr.loc[tvr.Splice == "reconstruction", "p-value (Holm)"].notna().all()
    assert tvr.loc[tvr.Splice == "error_preserving", "p-value (Holm)"].isna().all()
    assert len(pd.read_csv(run_dir / "random_draws.csv")) == 6 * 2
    dmg = pd.read_csv(run_dir / "reconstruction_damage.csv")
    assert dmg.Experiment.tolist() == ["TopK SAE", "L1 SAE", "Pretrained SAE"]
    diag = pd.read_csv(run_dir / "diagnostics.csv")
    assert {"Answer changed vs. control (%)", "First token changed vs. control (%)", "Empty answer (%)",
            "Edited features all inactive before the edit (%)", "Mean edit norm (rel. to |x|)"} <= set(diag.columns)
    trans = pd.read_csv(run_dir / "transitions.csv")
    per_cond = trans.groupby(["experiment", "condition"])["count"].sum()
    assert (per_cond.xs("random_ablation", level="condition") == 2 * len(episodes)).all()   # 2 draws pooled
    changed = pd.read_csv(run_dir / "changed_answers.csv")
    assert (changed.solver_attempt_2.fillna("") != changed.solver_attempt_2_ctl.fillna("")).all()
    t = pd.read_csv(run_dir / "causal_table.csv")
    assert (t.loc[t.Experiment.str.endswith(replay.EP_SUFFIX), "Condition"]
            .eq("No ablation (= plain replay)").sum() == 3)
    s = json.loads((run_dir / "causal_replay_summary.json").read_text())
    assert s["validated"] is False                                # the tiny model cannot reproduce collection


# ---- follow-up review: generated first token, stored feature ids, match rates, layer-aware scope

def test_first_token_is_the_one_generation_chose_not_the_raw_argmax(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    ep = episodes.iloc[0].to_dict()
    plain = replay.conditions(comp.features)[0]
    with replay.FirstTokenLogits(comp.model) as f:
        comp.solver.revise(ep["question"], comp.paragraphs[ep["question_id"]], ep["solver_attempt_1"],
                           ep["critic_feedback"])
    raw = int(f.logprobs.argmax())
    comp.solver.generation_settings["suppress_tokens"] = [raw]       # logits processing changes the choice
    row = replay.generate(comp, ep, plain)
    assert row["first_token_id"] == comp.solver.last_generated_token_ids[0] != raw
    assert row["first_token_logprob"] == pytest.approx(float(f.logprobs[row["first_token_id"]]), abs=1e-5)
    assert row["first_token_logprob"] < float(f.logprobs[raw])     # scored by the model's raw log-probs


@pytest.mark.parametrize("targeted, random", [
    ([1.9, 2], [[3, 4], [5, 6]]),
    ([1, 2], [[3.8, 4], [5, 6]]),
    ([True, 2], [[3, 4], [5, 6]]),
    (["1", 2], [[3, 4], [5, 6]]),
])
def test_stored_feature_ids_are_validated_as_stored_before_the_model_loads(tmp_path, monkeypatch, targeted, random):
    from mas_sae.models import roles
    from mas_sae.probe import feature_selection as fs
    (tmp_path / "fs.json").write_text(json.dumps({
        "config": {"layer": 33}, "provenance": {"sae_sha256": {"TopK SAE": {}}},
        "saes": {"TopK SAE": {"targeted_features": targeted, "random_control_draws": random}}}))
    cfg = _cfg(tmp_path, saes={"TopK SAE": {"kind": "custom", "path": "x"}})
    assert replay.feature_sets(cfg, {"TopK SAE": {}})["TopK SAE"]["targeted"] == targeted   # not coerced
    monkeypatch.setattr(fs, "sae_files", lambda spec, root: [])
    monkeypatch.setattr(fs, "load_sae", lambda spec, root, layer: (tiny_saes()["TopK SAE"], {}))
    monkeypatch.setattr(roles, "load_spec", lambda spec: pytest.fail("loaded the model before validating"))
    with pytest.raises(ValueError, match="integers"):
        replay.load_components(cfg, pd.DataFrame({"question_id": [], "source_split": []}))


def test_replay_check_match_rates_do_not_depend_on_the_mismatch_kind():
    manifest = pd.DataFrame({"question_id": ["a", "b"], "stored_solver_response": ["adopted_critic"] * 2})
    rows = pd.DataFrame([{"question_id": q, "experiment": replay.BASELINE, "solver_attempt_2": a,
                          "stored_solver_attempt_2": "Lyon", "solver_response": r, "adopted": int(r == "adopted_critic"),
                          "prompt_matches_stored": p}
                         for q, a, r, p in (("a", "Lyon", "adopted_critic", True),
                                            ("b", "Paris", "retained_a1", False))])   # prompt AND label differ
    check = replay.replay_check(rows, manifest, min_label_match=0.9)
    assert check["mismatch_counts"] == {"exact_match": 1, "prompt_differs": 1}
    assert check["solver_response_match_rate"] == 0.5 and check["a2_exact_match_rate"] == 0.5
    assert len(check["failures"]) == 2                                    # prompt difference and label match


def test_intervention_description_depends_on_whether_the_layer_is_final(tmp_path):
    assert "first token" in replay.intervention_description(33, 34)
    assert "keys and values" in replay.intervention_description(2, 4)
    assert "keys and values" in replay.intervention_description(33, None)
    cfg, episodes, comp = _smoke(tmp_path, layer=2, n_layers=4)
    assert comp.site == "model.language_model.layers.2" and replay.num_layers(comp.model) == 4
    run_dir = replay.replay(cfg, episodes.head(1), comp, tmp_path / "run")
    replay.analyse(run_dir)
    s = json.loads((run_dir / "causal_replay_summary.json").read_text())
    assert s["provenance"]["num_layers"] == 4 and "not the final layer" in s["intervention"]["description"]


# ---- steering: clamping features to s * ref

def test_splice_clamps_features_to_the_given_values():
    sae = tiny_saes()["TopK SAE"]
    m = _host()
    h = torch.randn(1, 4, 16)
    feats, values = [1, 2, 3], [2.0, -1.5, 0.5]
    with replay.SAESplice(m, "layer", sae, feats, values=values) as sp:
        out = m.layer(h)
    z = offline.encode(sae, h[0, -1].numpy()[None])
    z[:, feats] = values
    assert np.allclose(out[0, -1].numpy(), offline.decode(sae, z)[0], atol=1e-5)
    assert set(sp.stats) == {"n_active", "recon_rel_error", "n_ablated_active", "removed_rel_norm"}
    with replay.SAESplice(m, "layer", sae, feats, values=[0.0] * 3) as zeros:
        a = m.layer(h)
    with replay.SAESplice(m, "layer", sae, feats) as ablate:
        b = m.layer(h)
    assert torch.equal(a, b) and zeros.stats == ablate.stats             # clamping to 0 is ablation
    z = offline.encode(sae, h[0, -1].numpy()[None])
    with replay.SAESplice(m, "layer", sae, feats, mode="error_preserving", values=z[0, feats].tolist()):
        same = m.layer(h)
    assert torch.allclose(same, h, atol=1e-5)                            # setting features to their own value: no edit
    with pytest.raises(ValueError, match="3 features but 2 values"):
        replay.SAESplice(m, "layer", sae, feats, values=[1.0, 2.0])


def test_steering_conditions_clamp_each_set_to_strength_times_ref():
    feats = tiny_features(tiny_saes(), n_random_draws=2)
    conds = replay.conditions(feats, error_preserving=True, ablation=False, steer_strengths=[-2, 5])
    per_sae = (1 + 2) * 2 * 2 + 1                     # (targeted + draws) x strengths x splices + reconstruction only
    assert len(conds) == 1 + 3 * per_sae
    assert not any("ablation" in c["condition"] and c["condition"] != "no_ablation" for c in conds)
    t = next(c for c in conds if c["condition"] == "targeted_steer_-2" and c["experiment"] == "L1 SAE")
    ref = feats["L1 SAE"]["ref"]
    assert t["features"] == feats["L1 SAE"]["targeted"] and t["strength"] == -2
    assert t["values"] == pytest.approx([-2 * ref[str(i)] for i in t["features"]])
    r = next(c for c in conds if c["condition"] == "random_steer_+5" and c["draw"] == 1 and c["mode"] == "error_preserving")
    assert r["values"] == pytest.approx([5 * feats[r["sae"]]["ref"][str(i)] for i in feats[r["sae"]]["random"][1]])
    seen = set()
    for c in conds:                                    # controls still come first
        ctl = replay.control_key(c)
        assert ctl is None or ctl in seen
        seen.add((c["experiment"], c["condition"], c["draw"]))
    assert replay.condition_label("targeted_steer_+5") == "Targeted steer (+5× ref)"
    assert replay.split_condition("random_steer_-2") == ("random", "steer_-2")
    assert replay.split_condition("no_ablation") is None
    both = replay.conditions(feats, steer_strengths=[2])
    assert [c["condition"] for c in both[1:1 + 1 + 3 + 3]] == (
        ["no_ablation", "targeted_ablation", "random_ablation", "random_ablation",
         "targeted_steer_+2", "random_steer_+2", "random_steer_+2"])


def test_config_checks_the_steering_settings(tmp_path):
    p = tmp_path / "c.yaml"
    base = "capstone_data_root: .\ninteractions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\nlayer: 25\n"
    for extra, match in (("steer_strengths: [0, 2]\nfeature_reference: r\n", "nonzero"),
                         ("steer_strengths: [2, 2]\nfeature_reference: r\n", "distinct"),
                         ("steer_strengths: 2\nfeature_reference: r\n", "list"),
                         ("steer_strengths: [2]\n", "feature_reference"),
                         ("ablation: false\n", "Nothing to run"),
                         ("max_hours: 0\n", "max_hours")):
        p.write_text(base + extra)
        with pytest.raises(ValueError, match=match):
            replay.ReplayConfig.from_yaml(p)
    p.write_text(base + "ablation: false\nsteer_strengths: [-2, 2.5]\nfeature_reference: r\nmax_hours: 23\n")
    cfg = replay.ReplayConfig.from_yaml(p)
    assert (cfg.layer, cfg.steer_strengths, cfg.max_hours) == (25, [-2, 2.5], 23)


def _reference(tmp_path, layer=33, summary_sha=None, sae_hash="h"):
    from mas_sae.experiments.artifacts import sha256_file
    ref = {"layer": layer, "feature_selection_summary_sha256": summary_sha or sha256_file(tmp_path / "fs.json"),
           "saes": {"TopK SAE": {"sae_sha256": {"ckpt": sae_hash},
                                 "features": {str(i): {"ref": 0.5 * i} for i in range(1, 9)}}}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))


def test_feature_sets_attach_reference_activations_of_the_same_inputs(tmp_path):
    hashes = _summary(tmp_path)
    cfg = _cfg(tmp_path, saes={"TopK SAE": {"kind": "custom", "path": "x"}}, steer_strengths=[2],
               feature_reference="ref.json")
    _reference(tmp_path)
    f = replay.feature_sets(cfg, hashes)["TopK SAE"]
    assert f["ref"] == {"1": 0.5, "2": 1.0, "3": 1.5, "4": 2.0, "5": 2.5, "6": 3.0}   # only features in use
    for kw, match in ((dict(layer=25), "layer 25"), (dict(summary_sha="x"), "different feature-selection"),
                      (dict(sae_hash="other"), "reference activations used")):
        _reference(tmp_path, **kw)
        with pytest.raises(ValueError, match=match):
            replay.feature_sets(cfg, hashes)


def test_validate_features_needs_a_positive_ref_for_every_steered_feature():
    saes = tiny_saes()
    feats = tiny_features(saes, n_random_draws=2)
    replay.validate_features(feats, saes)
    feats["TopK SAE"]["ref"][str(feats["TopK SAE"]["random"][1][0])] = 0.0
    with pytest.raises(ValueError, match="no positive reference"):
        replay.validate_features(feats, saes)
    del feats["L1 SAE"]["ref"][str(feats["L1 SAE"]["targeted"][0])]
    feats["TopK SAE"]["ref"] = {k: 1.0 for k in feats["TopK SAE"]["ref"]}
    with pytest.raises(ValueError, match="L1 SAE targeted: no positive reference"):
        replay.validate_features(feats, saes)


@pytest.mark.parametrize("layer, n_layers", [(33, None), (5, 8)])
def test_steering_run_writes_dose_response_and_bidirectional_tables(tmp_path, layer, n_layers):
    cfg, episodes, comp = _smoke(tmp_path, layer=layer, n_layers=n_layers)
    cfg = dataclasses.replace(cfg, steer_strengths=[-2, 2], max_hours=1e-9)
    run_dir = replay.replay(cfg, episodes.head(3), comp, tmp_path / "run")
    rows = replay._read_rows(run_dir / "per_row.jsonl")
    n_conds = 1 + 3 * ((1 + 3 * 3) + 3 * 3)              # ablation + 2 strengths, (targeted + 2 draws) each
    assert len(rows) == 3 * n_conds
    steered = [r for r in rows if "steer" in r["condition"]]
    assert {r["strength"] for r in steered} == {-2, 2} and all(r["removed_rel_norm"] > 0 for r in steered)
    table = replay.analyse(run_dir)
    assert len(table) == 2 + 6 * (1 + 2 * 3)
    assert "Targeted steer (+2× ref)" in set(table.Condition)
    tvr = pd.read_csv(run_dir / "targeted_vs_random.csv")
    assert list(tvr.columns) == replay.TVR_COLUMNS and len(tvr) == 6 * 3
    assert tvr.loc[tvr.Splice == "reconstruction", "p-value (Holm)"].notna().all()
    dose = pd.read_csv(run_dir / "dose_response.csv")
    assert len(dose) == 6 * 3
    assert dose[dose.Experiment == "TopK SAE"]["Strength (× ref)"].tolist() == [-2.0, 0.0, 2.0]
    bi = pd.read_csv(run_dir / "steering_bidirectional.csv")
    assert len(bi) == 6 and (bi["Strength (× ref)"] == 2).all()
    s = json.loads((run_dir / "causal_replay_summary.json").read_text())
    assert s["intervention"]["arms"] == ["ablation", "steer_-2", "steer_+2"]
    assert meta_layer(run_dir) == layer


def meta_layer(run_dir):
    return json.loads((run_dir / "replay_config.json").read_text())["config"]["layer"]


def test_steering_only_run_skips_ablation(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    cfg = dataclasses.replace(cfg, ablation=False, steer_strengths=[3])
    run_dir = replay.replay(cfg, episodes.head(2), comp, tmp_path / "run")
    assert not any("ablation" in r["condition"] and r["condition"] != "no_ablation"
                   for r in replay._read_rows(run_dir / "per_row.jsonl"))
    replay.analyse(run_dir)
    assert pd.read_csv(run_dir / "targeted_vs_random.csv").Arm.unique().tolist() == ["steer_+3"]
    assert pd.read_csv(run_dir / "steering_bidirectional.csv").empty      # no -3 to compare with


def test_reference_activations_are_mean_nonzero_values_on_the_fit_partition(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mas_sae.probe import feature_selection as fs
    sae = tiny_saes()["TopK SAE"]
    X = np.random.default_rng(0).normal(size=(40, 16)).astype(np.float32)
    part = np.array(["train"] * 30 + ["validation"] * 10)
    monkeypatch.setattr(fs.ls, "load_inputs", lambda c: (pd.DataFrame({"p": part}), None, SimpleNamespace(partition="p")))
    monkeypatch.setattr(fs.bd, "load_layer_matrix", lambda *a: X)
    monkeypatch.setattr(fs, "sae_files", lambda spec, root: [])
    monkeypatch.setattr(fs, "load_sae", lambda spec, root, layer: (sae, {}))
    Z = offline.encode(sae, X[:30])
    live = [int(i) for i in np.flatnonzero((Z > 0).any(0))][:4]
    summary = {"config": {"layer": 25}, "provenance": {"sae_sha256": {"TopK SAE": {}}},
               "saes": {"TopK SAE": {"targeted_features": live[:2], "random_control_draws": [live[2:]]}}}
    cfg = fs.FeatureSelectionConfig(capstone_data_root=tmp_path, activations_dir="a", results_dir="r", labels_csv="l",
                                    manifest_csv="m", layer=25, saes={"TopK SAE": {"kind": "custom", "path": "x"}})
    out = fs.reference_activations(cfg, summary)["TopK SAE"]["features"]
    assert sorted(out) == sorted(str(i) for i in live)
    for i in live:
        col = Z[:, i]
        assert out[str(i)]["ref"] == pytest.approx(col[col > 0].mean())
        assert out[str(i)]["firing_frequency"] == pytest.approx((col > 0).mean())
    with pytest.raises(ValueError, match="layer 25, not 33"):
        fs.reference_activations(dataclasses.replace(cfg, layer=33), summary)
    dead = int(np.flatnonzero(~(Z > 0).any(0))[0])
    summary["saes"]["TopK SAE"]["targeted_features"] = [dead]
    with pytest.raises(ValueError, match="never fires"):
        fs.reference_activations(cfg, summary)
