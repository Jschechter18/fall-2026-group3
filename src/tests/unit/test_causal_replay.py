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
                                                "removed_rel_norm", "raw_edit_rel_norm", "edit_scale"}
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
        main = replay.ReplayConfig.from_yaml(repo / f"configs/causal_edit_size_layer{layer}.yaml")
        l1 = replay.ReplayConfig.from_yaml(repo / f"configs/causal_edit_size_l1_layer{layer}.yaml")
        for c in (main, l1):
            assert (c.layer, c.random_controls, c.reconstruction, c.error_preserving) == (layer, "contribution", False, True)
            assert (c.feature_selection_summary, c.feature_reference) == (steer.feature_selection_summary, steer.feature_reference)
            assert c.ablation and not c.steer_strengths and all(c.saes[n] == steer.saes[n] for n in c.saes)
        assert set(main.saes) == set(steer.saes) and set(l1.saes) == {"L1 SAE"}
        assert (main.limit, l1.limit, main.random_directions, l1.random_directions) == (None, None, 3, 0)
        summary = json.loads((repo / steer.feature_selection_summary).read_text())
        (full, alone), = [(set(summary["saes"]["L1 SAE"]["targeted_features"]), s) for s in l1.targeted_subsets.values()]
        one, rest = alone.values()
        assert len(one) == 1 and set(one) | set(rest) == full and not set(one) & set(rest)
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
    assert set(sp.stats) == {"n_active", "recon_rel_error", "n_ablated_active", "removed_rel_norm",
                             "raw_edit_rel_norm", "edit_scale"}
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
    every = tvr[tvr.Rows == "all"]
    assert list(tvr.columns) == replay.TVR_COLUMNS and len(every) == 6 * 3 and (every.N == 3).all()
    assert every.loc[every.Splice == "reconstruction", "p-value (Holm)"].notna().all()
    edited = tvr[tvr.Rows == "edited only"]                   # its own Holm family, on the primary splice too
    assert edited.loc[edited.Splice == "reconstruction", "p-value (Holm)"].notna().all() and (tvr.N <= 3).all()
    assert (tvr.Control == "matched features").all() and tvr["Random no-op (%)"].between(0, 100).all()
    dose = pd.read_csv(run_dir / "dose_response.csv")
    assert len(dose) == 6 * 3
    topk = dose[dose.Experiment == "TopK SAE"]
    assert topk.Edit.tolist() == ["ablation", "steer", "steer"] and topk.Strength.tolist() == [0.0, -2.0, 2.0]
    bi = pd.read_csv(run_dir / "steering_bidirectional.csv")
    assert len(bi) == 6 and (bi.Edit == "steer").all() and (bi.Strength == 2).all()
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
    full = fs.reference_activations(cfg, summary)["TopK SAE"]
    out = full["features"]
    assert sorted(out) == sorted(str(i) for i in live)
    assert full["latents"]["n_rows"] == 30 and len(full["latents"]["rms"]) == sae.latent_dim
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


# ---- edit size (issue #146): scaling, norm-matched edits, matched controls, subsets

def test_splice_scales_features_and_full_suppression_is_ablation():
    sae = tiny_saes()["L1 SAE"]
    m = _host()
    h = torch.randn(1, 4, 16)
    feats = [1, 2, 3]
    with replay.SAESplice(m, "layer", sae, feats, fraction=0.5) as sp:
        out = m.layer(h)
    z = offline.encode(sae, h[0, -1].numpy()[None])
    z[:, feats] *= 1.5
    assert np.allclose(out[0, -1].numpy(), offline.decode(sae, z)[0], atol=1e-5) and sp.stats["edit_scale"] == 1.0
    with replay.SAESplice(m, "layer", sae, feats, fraction=-1) as scaled:
        a = m.layer(h)
    with replay.SAESplice(m, "layer", sae, feats) as ablated:
        b = m.layer(h)
    assert torch.allclose(a, b, atol=1e-6)
    assert scaled.stats["removed_rel_norm"] == pytest.approx(ablated.stats["removed_rel_norm"], rel=1e-5)
    for kw, match in ((dict(values=[1.0] * 3, fraction=0.5), "at most one"), (dict(fraction=-1.5), ">= -1"),
                      (dict(edit_norm=0.0), "nonzero")):
        with pytest.raises(ValueError, match=match):
            replay.SAESplice(m, "layer", sae, feats, **kw)


@pytest.mark.parametrize("mode", replay.MODES)
@pytest.mark.parametrize("r", [-0.25, 0.25])
def test_norm_matched_edit_has_the_requested_size_along_the_ablation_direction(mode, r):
    sae = tiny_saes()["TopK SAE"]
    m = _host()
    h = torch.randn(1, 4, 16)
    x = h[0, -1].numpy()
    z = offline.encode(sae, x[None])
    active = [int(i) for i in np.flatnonzero(z[0] > 0)[:3]]
    ablation = offline.decode(sae, offline.ablate(z, active))[0] - offline.decode(sae, z)[0]
    with replay.SAESplice(m, "layer", sae, active, mode=mode, edit_norm=r) as sp:
        out = m.layer(h)[0, -1].numpy()
    delta = out - (offline.decode(sae, z)[0] if mode == "reconstruction" else x)
    cos = delta @ ablation / (np.linalg.norm(delta) * np.linalg.norm(ablation))
    assert np.linalg.norm(delta) / np.linalg.norm(x) == pytest.approx(0.25, rel=1e-4)
    assert cos == pytest.approx(-1.0 if r > 0 else 1.0, abs=1e-5)       # r < 0 removes, r > 0 adds
    raw = np.linalg.norm(ablation) / np.linalg.norm(x)
    assert sp.stats["removed_rel_norm"] == pytest.approx(0.25, rel=1e-5)
    assert sp.stats["raw_edit_rel_norm"] == pytest.approx(raw, rel=1e-4)
    assert sp.stats["edit_scale"] == pytest.approx(0.25 / raw, rel=1e-4)


def test_norm_matched_edit_of_features_that_are_not_firing_is_a_recorded_no_op():
    sae = tiny_saes()["TopK SAE"]
    m = _host()
    h = torch.randn(1, 4, 16)
    z = offline.encode(sae, h[0, -1].numpy()[None])
    inactive = [int(i) for i in np.flatnonzero(z[0] == 0)[:3]]
    with replay.SAESplice(m, "layer", sae, inactive, mode="error_preserving", edit_norm=-0.5) as sp:
        out = m.layer(h)
    assert torch.equal(out, h)
    assert sp.stats["n_ablated_active"] == 0 and sp.stats["removed_rel_norm"] == 0 and sp.stats["edit_scale"] is None


def _with_subset(feats, sae="L1 SAE", label="first"):
    f = feats[sae]
    f["subsets"] = {label: {"targeted": f["targeted"][:1], "random": [d[:1] for d in f["random"]]}}
    return feats


def test_conditions_for_scaling_norm_matched_edits_and_subsets():
    feats = _with_subset(tiny_features(tiny_saes(), n_random_draws=2))
    conds = replay.conditions(feats, error_preserving=True, ablation=False, scale_fractions=[-0.5, 0.5],
                              edit_norms=[-0.25, 0.25], reconstruction=False)
    assert len(conds) == 1 + (3 + 1) * (1 + 2) * 4            # (3 SAEs + 1 subset) x (targeted + draws) x arms
    assert not any(c["condition"] == "no_ablation" for c in conds)
    assert all(replay.control_key(c) == (replay.BASELINE, "no_sae", None) for c in conds[1:])
    sub = [c for c in conds if c["experiment"] == "L1 SAE · first · error-preserving"]
    assert len(sub) == 12 and {len(c["features"]) for c in sub} == {1} and {c["sae"] for c in sub} == {"L1 SAE"}
    n = next(c for c in conds if c["condition"] == "targeted_norm_-0.25" and c["experiment"] == "TopK SAE · error-preserving")
    assert (n["edit_norm"], n["fraction"], n["values"], n["strength"]) == (-0.25, None, None, None)
    s = next(c for c in conds if c["condition"] == "random_scale_+0.5" and c["draw"] == 1)
    assert s["fraction"] == 0.5 and s["features"] == feats[s["sae"]]["random"][1]
    both = replay.conditions(feats, error_preserving=True, ablation=True, edit_norms=[0.1])
    recon_sub = next(c for c in both if c["experiment"] == "L1 SAE · first")
    assert replay.control_key(recon_sub) == ("L1 SAE", "no_ablation", None)
    seen = set()
    for c in both:                                    # controls still come first
        ctl = replay.control_key(c)
        assert ctl is None or ctl in seen
        seen.add((c["experiment"], c["condition"], c["draw"]))
    assert replay.condition_label("targeted_norm_-0.25") == "Targeted suppress (|Δ| = 0.25·|x|)"
    assert replay.condition_label("random_norm_+1") == "Random amplify (|Δ| = 1·|x|)"
    assert replay.condition_label("random_scale_-0.5") == "Random scale (z × 0.5)"
    assert [replay.arm_family(a) for a in ("ablation", "steer_+2", "scale_-1", "norm_+0.1")] == list(replay.FAMILIES)


def test_contribution_matched_controls_match_frequency_first_then_contribution():
    # feature 0 is targeted; 1 and 5 fire as often (5 also matches its contribution), 2 fires 2x as often
    freq = np.array([0.01, 0.01, 0.02, 0.5, 0.0, 0.011])
    contribution = np.array([10.0, 1.0, 10.0, 10.0, 10.0, 9.0])
    assert offline.frequency_matched_controls(freq, [0], 3, seed=0, pool_size=1) == [[1]] * 3
    assert offline.contribution_matched_controls(freq, contribution, [0], 3, seed=0, pool_size=1) == [[5]] * 3
    # within the frequency band contribution decides; outside it (2 fires 2x as often) it does not count
    assert offline.contribution_matched_controls(freq, contribution, [0], 3, seed=0, pool_size=1, exclude=[5]) == [[1]] * 3
    assert offline.contribution_matched_controls(freq, contribution, [0], 3, seed=0, pool_size=1, exclude=[5],
                                                 freq_tolerance=2.5) == [[2]] * 3
    # nothing fires as often as 3 (0.5): the pool falls back to the nearest frequencies
    assert offline.contribution_matched_controls(freq, contribution, [3], 2, seed=0, pool_size=1) == [[2]] * 2
    q = offline.match_quality(freq, contribution, [0], [[5]])
    assert q == [{"draw": 0, "targeted": 0, "control": 5, "targeted_frequency": 0.01, "control_frequency": 0.011,
                  "frequency_ratio": pytest.approx(1.1), "targeted_contribution": 10.0, "control_contribution": 9.0,
                  "contribution_ratio": pytest.approx(0.9)}]
    draws = offline.contribution_matched_controls(freq, contribution, [0, 2], 4, seed=1, pool_size=3)
    assert draws == offline.contribution_matched_controls(freq, contribution, [0, 2], 4, seed=1, pool_size=3)
    assert all(len(set(d)) == 2 and not set(d) & {0, 2, 4} for d in draws)    # dead feature 4 is never drawn
    with pytest.raises(ValueError, match="must fire"):
        offline.contribution_matched_controls(freq, contribution, [4], 1, seed=0)


def test_targeted_vs_random_leaves_out_missing_draws():
    t = np.array([1.0, 0.0, 1.0, 1.0])
    r = np.array([[0.0, np.nan], [1.0, 0.0], [np.nan, 0.0], [1.0, 1.0]])
    res = offline.targeted_vs_random(t, r, np.zeros(4), np.array(["a", "b", "c", "d"]), 100, 500, 0)
    assert res["delta_pp"] == pytest.approx(100 * np.mean(t - [0.0, 0.5, 0.0, 1.0]))
    assert res["random_draw_vs_control_pp"] == pytest.approx([100 * 2 / 3, 100 / 3])
    assert np.isfinite(res["ci95_pp_questions_and_draws"]).all()
    with pytest.raises(ValueError, match="at least one random draw"):
        offline.targeted_vs_random(t, np.where([[True, True]] * 4, np.nan, r), np.zeros(4), np.arange(4), 10, 10, 0)


def test_latent_stats_cover_every_latent():
    from mas_sae.probe import feature_selection as fs
    X = np.random.default_rng(1).normal(size=(30, 16)).astype(np.float32)
    for name, sae in tiny_saes().items():
        st = fs.latent_stats(sae, X, batch_size=7)
        Z = offline.encode(sae, X)
        assert st["n_rows"] == 30 and len(st["firing_frequency"]) == sae.latent_dim
        assert np.allclose(st["firing_frequency"], (Z > 0).mean(0), rtol=1e-5)
        assert np.allclose(st["rms"], np.sqrt((Z.astype(np.float64) ** 2).mean(0)), rtol=1e-5)
        for i in range(sae.latent_dim):
            on = Z[:, i][Z[:, i] > 0]
            assert (st["ref"][i] is None) == (len(on) == 0)
            if len(on):
                assert st["ref"][i] == pytest.approx(on.mean(), rel=1e-5)
        e = np.eye(sae.latent_dim, dtype=np.float32)[:5]
        cols = offline.decode(sae, e) - offline.decode(sae, np.zeros((1, sae.latent_dim), np.float32))
        assert np.allclose(st["decoder_norm"][:5], np.linalg.norm(cols, axis=1), rtol=1e-5)
    pre = tiny_saes()["Pretrained SAE"]
    assert np.allclose(fs.latent_stats(pre, X)["decoder_norm"], pre.w_dec.norm(dim=1).numpy(), rtol=1e-5)


def _reference_with_latents(tmp_path, n=12, latents=True):
    from mas_sae.experiments.artifacts import sha256_file
    rng = np.random.default_rng(0)
    lat = {"n_rows": 100, "firing_frequency": rng.uniform(0.05, 1, n).round(4).tolist(),
           "ref": rng.uniform(0.5, 2, n).round(4).tolist(), "rms": rng.uniform(0.1, 1, n).round(4).tolist(),
           "decoder_norm": [1.0] * n}
    sae = {"sae_sha256": {"ckpt": "h"}, "features": {str(i): {"ref": 1.0} for i in range(1, 9)}}
    ref = {"layer": 33, "feature_selection_summary_sha256": sha256_file(tmp_path / "fs.json"),
           "saes": {"TopK SAE": {**sae, **({"latents": lat} if latents else {})}}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    return lat


def test_feature_sets_redraw_contribution_matched_controls_and_draw_for_each_subset(tmp_path):
    hashes = _summary(tmp_path)                                 # targeted [1, 2]
    lat = _reference_with_latents(tmp_path)
    saes = {"TopK SAE": {"kind": "custom", "path": "x"}}
    cfg = _cfg(tmp_path, saes=saes, random_controls="contribution", feature_reference="ref.json",
               targeted_subsets={"TopK SAE": {"one": [1]}}, steer_strengths=[2])
    f = replay.feature_sets(cfg, hashes)["TopK SAE"]
    contribution = np.array(lat["rms"]) * np.array(lat["decoder_norm"])
    expected = offline.contribution_matched_controls(lat["firing_frequency"], contribution, [1, 2], 2, cfg.seed,
                                                     exclude=[1, 2])
    assert f["targeted"] == [1, 2] and f["random"] == expected
    sub = f["subsets"]["one"]
    assert sub["targeted"] == [1] and len(sub["random"]) == 2 and all(len(d) == 1 and d[0] not in (1, 2) for d in sub["random"])
    assert f["match"] == offline.match_quality(lat["firing_frequency"], contribution, [1, 2], expected)
    assert [m["control"] for m in sub["match"]] == [d[0] for d in sub["random"]]
    used = {i for s in (f, sub) for i in [*s["targeted"], *(i for d in s["random"] for i in d)]}
    assert f["ref"] == {str(i): lat["ref"][i] for i in sorted(used)}   # refs of the new draws come from the latents
    for subsets, match in (({"TopK SAE": {"all": [1, 2]}}, "proper subset"), ({"TopK SAE": {"x": [9]}}, "proper subset")):
        with pytest.raises(ValueError, match=match):
            replay.feature_sets(dataclasses.replace(cfg, targeted_subsets=subsets), hashes)
    _reference_with_latents(tmp_path, latents=False)
    with pytest.raises(ValueError, match="per-latent statistics"):
        replay.feature_sets(cfg, hashes)


def test_config_checks_the_edit_size_settings(tmp_path):
    p = tmp_path / "c.yaml"
    base = ("capstone_data_root: .\ninteractions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\nlayer: 25\n"
            "saes:\n  L1 SAE:\n    kind: custom\n    path: x\n")
    for extra, match in (("scale_fractions: [-1.5]\n", ">= -1"), ("scale_fractions: [0]\n", "not 0"),
                         ("edit_norms: [0]\n", "nonzero"), ("edit_norms: [0.1, 0.1]\n", "distinct"),
                         ("reconstruction: false\nerror_preserving: false\n", "no splice"),
                         ("random_controls: other\n", "random_controls must be"),
                         ("random_controls: contribution\n", "needs feature_reference"),
                         ("targeted_subsets: {L1 SAE: {one: [7025]}}\nfeature_reference: r\n", "needs random_controls"),
                         ("random_controls: contribution\nfeature_reference: r\ntargeted_subsets: {TopK SAE: {one: [1]}}\n",
                          "targeted_subsets must map")):
        p.write_text(base + extra)
        with pytest.raises(ValueError, match=match):
            replay.ReplayConfig.from_yaml(p)
    p.write_text(base + "ablation: false\nreconstruction: false\nedit_norms: [-0.1, 0.1]\nscale_fractions: [-1]\n"
                        "random_controls: contribution\nfeature_reference: r\ntargeted_subsets: {L1 SAE: {f7025: [7025]}}\n")
    cfg = replay.ReplayConfig.from_yaml(p)
    assert (cfg.edit_norms, cfg.scale_fractions, cfg.targeted_subsets) == ([-0.1, 0.1], [-1], {"L1 SAE": {"f7025": [7025]}})


def test_norm_matched_run_reports_edit_sizes_no_ops_and_both_directions(tmp_path):
    cfg, episodes, comp = _smoke(tmp_path)
    _with_subset(comp.features)
    cfg = dataclasses.replace(cfg, ablation=False, reconstruction=False, scale_fractions=[-0.5, 0.5],
                              edit_norms=[-0.25, 0.25], random_directions=2)
    run_dir = replay.replay(cfg, episodes.head(3), comp, tmp_path / "run")
    rows = replay._read_rows(run_dir / "per_row.jsonl")
    assert len(rows) == 3 * (1 + 4 * 3 * 4 + 2 * 2)          # + 2 edit norms x 2 random directions
    directed = [r for r in rows if r["experiment"] == replay.DIRECTION]
    assert len(directed) == 3 * 4 and all(r["removed_rel_norm"] == pytest.approx(0.25, rel=1e-4) for r in directed)
    normed = [r for r in rows if "_norm_" in r["condition"]]
    assert all(r["removed_rel_norm"] == pytest.approx(0.25, rel=1e-4) or (r["removed_rel_norm"] == 0 and r["edit_scale"] is None)
               for r in normed)
    assert {r["edit_norm"] for r in normed} == {-0.25, 0.25} and all(r["fraction"] is None for r in normed)
    table = replay.analyse(run_dir)
    assert len(table) == 2 + 4 * (1 + 2 * 4) + (1 + 2)
    tvr = pd.read_csv(run_dir / "targeted_vs_random.csv")
    every = tvr[(tvr.Rows == "all") & (tvr.Control == "matched features")]
    assert len(every) == 4 * 4 and every["p-value (Holm)"].notna().all()     # error-preserving is primary here
    assert "L1 SAE · first · error-preserving" in set(every.Experiment)
    direction = tvr[(tvr.Rows == "all") & (tvr.Control == "random direction")]
    assert len(direction) == 4 * 2 and set(direction.Arm) == {"norm_-0.25", "norm_+0.25"}
    assert (direction["Random no-op (%)"] == 0).all() and direction["p-value (Holm)"].notna().all()
    assert replay.DIRECTION in set(table.Experiment) and replay.DIRECTION not in set(pd.read_csv(run_dir / "dose_response.csv").Experiment)
    assert list(pd.read_csv(run_dir / "control_match.csv").columns) == replay.MATCH_COLUMNS
    bi = pd.read_csv(run_dir / "steering_bidirectional.csv")
    assert sorted(bi.Edit.unique()) == ["norm", "scale"] and len(bi) == 4 * 2
    diag = pd.read_csv(run_dir / "diagnostics.csv")
    edits = diag[diag.Condition.str.contains("suppress|amplify")]
    assert {"No-op edits (%)", "Mean raw edit norm (rel. to |x|)", "Mean edit scale"} <= set(diag.columns)
    assert edits["No-op edits (%)"].notna().all()
    assert pd.read_csv(run_dir / "reconstruction_damage.csv").empty
    s = json.loads((run_dir / "causal_replay_summary.json").read_text())
    assert s["intervention"]["arms"] == ["scale_-0.5", "scale_+0.5", "norm_-0.25", "norm_+0.25"]
    assert "error_preserving splice" in s["multiple_comparisons"]


def test_random_direction_edit_has_the_requested_size_and_is_fixed_by_its_seed():
    m = _host()
    h = torch.randn(1, 4, 16)
    x = h[0, -1]
    out = {}
    for r, seed in ((-0.25, 7), (0.25, 7), (0.25, 8)):
        with replay.DirectionSplice(m, "layer", r, seed) as sp:
            out[r, seed] = m.layer(h)[0, -1] - x
        assert sp.stats["removed_rel_norm"] == pytest.approx(abs(r), rel=1e-5) and sp.stats["edit_scale"] is None
    assert (out[0.25, 7].norm() / x.norm()).item() == pytest.approx(0.25, rel=1e-5)
    assert torch.allclose(out[-0.25, 7], -out[0.25, 7], atol=1e-6)            # same direction, opposite sign
    with replay.DirectionSplice(m, "layer", 0.25, 7):
        again = m.layer(h)[0, -1] - x
    assert torch.equal(again, out[0.25, 7]) and not torch.allclose(out[0.25, 8], out[0.25, 7])
    with pytest.raises(ValueError, match="nonzero"):
        replay.DirectionSplice(m, "layer", 0.0, 1)


def test_random_direction_conditions():
    feats = tiny_features(tiny_saes(), n_random_draws=2)
    conds = replay.conditions(feats, error_preserving=True, edit_norms=[-0.1, 0.1], random_directions=3, seed=42)
    d = [c for c in conds if c["experiment"] == replay.DIRECTION]
    assert len(d) == 2 * 3 and {c["direction_seed"] for c in d} == {42000, 42001, 42002}
    assert all(c["features"] == [] and c["mode"] == "error_preserving" and replay.control_key(c) == (replay.BASELINE, "no_sae", None)
               for c in d)
    # the control uses the error-preserving splice, so a reconstruction-only run has none
    assert not any(c["experiment"] == replay.DIRECTION for c in replay.conditions(feats, edit_norms=[0.1], random_directions=3))


def test_config_checks_the_random_direction_settings(tmp_path):
    p = tmp_path / "c.yaml"
    base = "capstone_data_root: .\ninteractions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\nlayer: 25\n"
    for extra, match in (("random_directions: -1\nedit_norms: [0.1]\n", "nonnegative"),
                         ("random_directions: 2\n", "needs edit_norms"),
                         ("random_directions: 2\nedit_norms: [0.1]\nerror_preserving: false\n", "needs edit_norms")):
        p.write_text(base + extra)
        with pytest.raises(ValueError, match=match):
            replay.ReplayConfig.from_yaml(p)
    p.write_text(base + "random_directions: 2\nedit_norms: [0.1]\n")
    assert replay.ReplayConfig.from_yaml(p).random_directions == 2
