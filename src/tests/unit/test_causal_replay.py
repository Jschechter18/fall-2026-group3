"""Causal replay (issue #90): the splice hook, episode loading checks, and the run/resume/table mechanics."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from mas_sae.causal import offline, replay
from mas_sae.causal.smoke import smoke_components, tiny_saes
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
    assert sp.calls == 2 and set(sp.stats) == {"n_active", "recon_rel_error", "n_ablated_active"}
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
                feature_selection_summary="fs.json", partition="validation", n_random_draws=2,
                n_bootstrap=50, n_permutations=200, output_dir=tmp_path / "out")
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
    p.write_text("interactions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\n")
    with pytest.raises(ValueError, match="capstone_data_root"):
        replay.ReplayConfig.from_yaml(p)
    p.write_text("capstone_data_root: .\ninteractions_jsonl: a\nlabels_csv: b\nfeature_selection_summary: c\n"
                 "saes:\n  Baseline:\n    kind: custom\n    path: x\n")
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


# ---- end to end on the tiny model

def test_replay_writes_the_table_and_resumes_without_regenerating(tmp_path):
    _write_inputs(tmp_path, _episodes())
    _summary(tmp_path)
    cfg = _cfg(tmp_path)
    episodes = replay.load_episodes(cfg)
    paragraphs = {q: [{"idx": 0, "title": "T", "paragraph_text": "text"}] for q in episodes.question_id}
    comp = smoke_components(paragraphs, n_random_draws=2, max_new_tokens=2)
    run_dir = tmp_path / "out" / "run"

    replay.replay(cfg, episodes.head(3), comp, run_dir)
    per_row = run_dir / "per_row.jsonl"
    n_conditions = 1 + 3 * (2 + 2)
    assert len(per_row.read_text().splitlines()) == 3 * n_conditions

    replay.replay(cfg, episodes, comp, run_dir)                     # resume and extend
    assert len(per_row.read_text().splitlines()) == len(episodes) * n_conditions

    table = replay.analyse(run_dir)
    assert list(table.columns) == replay.TABLE_COLUMNS
    assert table.Condition.tolist()[:2] == ["No SAE (stored A2)", "No SAE (replayed A2)"]
    assert len(table) == 2 + 3 * 3 and (table.N == len(episodes)).all()
    assert table.loc[0, "Acceptance Rate (%)"] == pytest.approx(100 * episodes.stored_adopted.mean())
    assert set(table.loc[table.Condition == "No ablation", "Δ vs. SAE Control (pp)"]) == {"Reference"}
    for f in ("causal_table.csv", "per_row.csv", "causal_replay_summary.json", "causal_effects.svg", "episodes.csv"):
        assert (run_dir / f).exists()
    meta = json.loads((run_dir / "replay_config.json").read_text())
    assert meta["provenance"]["label_policy"] == bl.V012_QC.version

    with pytest.raises(ValueError, match="different config"):
        replay.replay(_cfg(tmp_path, seed=7), episodes, comp, run_dir)


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
