"""Controls added on top of #207 reuse the serial and distributed replay contracts."""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from mas_sae.causal import distributed, prepare, replay
from mas_sae.causal.smoke import smoke_components, tiny_features, tiny_saes
from mas_sae.evaluation import behavior_labels as bl
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.probe.feature_selection import ActivationBasis
from mas_sae.probe import feature_selection
from mas_sae.probe.matched_edits import direction_scores


def cfg(tmp_path, **kw):
    return replay.ReplayConfig.from_dict({
        "capstone_data_root": str(tmp_path), "interactions_jsonl": "validation/interactions.jsonl",
        "labels_csv": "labels/labels.csv", "label_policy": "v0.2.0-candidate",
        "feature_selection_summary": "summary.json", "layer": 1, "partition": "validation",
        "n_random_draws": 1, "reconstruction": False, "error_preserving": True,
        "edit_norms": [-.01, -.02], "random_directions": 2, "probe_direction": "direction.npz",
        "n_bootstrap": 20, "n_permutations": 30, "on_replay_check_fail": "mark_unvalidated", **kw})


def host():
    model = torch.nn.Module()
    model.layer = torch.nn.Identity()
    return model


@pytest.mark.parametrize("policy", ["allow", "cap_at_ablation"])
def test_norm_policy_and_offline_hook_agreement(policy):
    sae, model = ActivationBasis([0., 0.]), host()
    x = np.array([[1., 10.]], dtype=np.float32)
    h = torch.from_numpy(np.stack([x, x], axis=1))
    with replay.SAESplice(model, "layer", sae, [0], "error_preserving", edit_norm=-.5,
                         norm_edit_policy=policy) as splice:
        edited = model.layer(h)[0, -1].numpy()
    w = np.array([2., 3.])
    shift, _, _, _ = direction_scores(np.array([[-1., 0.]]), x, w)
    if policy == "allow":
        assert (edited - x[0]) @ w == pytest.approx(.5 * shift[0])
        assert edited[0] < 0 and splice.stats["overshoot"]
    else:
        assert edited[0] == 0 and splice.stats["edit_capped"]
        assert splice.stats["removed_rel_norm"] < .5 and not splice.stats["overshoot"]
    assert splice.stats["would_overshoot"] and splice.stats["requested_edit_scale"] > 1


def test_signed_raw_activity_and_zero_edits_are_not_fabricated():
    model, sae = host(), ActivationBasis([2., 3., 4.])
    h = torch.tensor([[[1., 1., 1.], [1., 2., 4.]]])
    with replay.SAESplice(model, "layer", sae, [0, 1], "error_preserving") as splice:
        out = model.layer(h)
    assert torch.equal(out[0, -1], torch.tensor([2., 3., 4.]))
    assert splice.stats["n_ablated_active"] == 2
    with replay.SAESplice(model, "layer", sae, [2], "error_preserving", edit_norm=-.02,
                         norm_edit_policy="cap_at_ablation") as splice:
        out = model.layer(h)
    assert torch.equal(out, h) and splice.stats["edit_scale"] is None


def test_supplied_direction_has_correct_sign_norm_and_hook_scope():
    model, w = host(), np.array([1., -2., 3.])
    h, step = torch.ones(1, 3, 3), torch.ones(1, 1, 3)
    with replay.DirectionSplice(model, "layer", -.02, direction=w):
        out, later = model.layer(h), model.layer(step)
    delta = out[0, -1] - h[0, -1]
    assert float(delta.norm() / h[0, -1].norm()) == pytest.approx(.02, rel=1e-5)
    assert float(delta @ torch.tensor(w, dtype=delta.dtype)) < 0
    assert torch.equal(later, step) and torch.equal(out[:, :-1], h[:, :-1])
    with pytest.raises(ValueError, match="nonzero vector"):
        replay.DirectionSplice(model, "layer", -.01, direction=[0., 0., 0.])


def test_probe_hash_and_layer_are_checked(tmp_path):
    path = tmp_path / "direction.npz"
    np.savez(path, w=np.ones(16), layer=1)
    summary = {"saes": {"Raw": {"probe_direction": {"sha256": sha256_file(path)}}}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert np.array_equal(replay.load_probe_direction(cfg(tmp_path)), np.ones(16))
    np.savez(path, w=np.ones(16), layer=2)
    with pytest.raises(ValueError, match="exported"):
        replay.load_probe_direction(cfg(tmp_path))
    summary["saes"]["Raw"]["probe_direction"]["sha256"] = sha256_file(path)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="layer"):
        replay.load_probe_direction(cfg(tmp_path))


def test_config_preparation_reuses_builders_without_solver_loading(tmp_path, monkeypatch):
    c = cfg(tmp_path, random_controls="contribution", feature_reference="feature_reference_activations.json",
            input_preparation={"collection_results": ".", "feature_selection_config": "features.yaml"})
    calls = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda args, **kw: calls.append(args))
    monkeypatch.setattr(replay, "load_episodes", lambda c: pd.DataFrame({"question_id": ["q"]}))
    monkeypatch.setattr(replay, "feature_sets", lambda *args: {})
    monkeypatch.setattr(replay, "load_probe_direction", lambda *args: None)
    prepare.prepare_inputs(c)
    assert len(calls) == 2
    assert calls[0][1].endswith("prepare_behavior_v01.py") and "v0.2.0-candidate" in calls[0]
    assert calls[1][1].endswith("feature_reference_activations.py")
    assert str(tmp_path / "features.yaml") in calls[1]


def test_reference_cli_reports_signed_neuron_statistics(tmp_path, monkeypatch, capsys):
    config, summary = tmp_path / "config.yaml", tmp_path / "summary.json"
    config.write_text("fixture")
    summary.write_text("{}")
    monkeypatch.setattr(feature_selection.FeatureSelectionConfig, "from_yaml",
                        lambda path: SimpleNamespace(layer=1, fit_partition="train"))
    monkeypatch.setattr(feature_selection, "reference_activations", lambda *args: {
        "Raw": {"features": {"0": {"std": 2.}}, "latents": {"basis": "activations"}},
        "SAE": {"features": {"0": {"ref": 1.}}, "latents": {}}})
    script = Path(__file__).resolve().parents[3] / "scripts/feature_reference_activations.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--config", str(config), "--summary", str(summary)])
    runpy.run_path(str(script), run_name="__main__")
    assert "Raw: 1 features, std median" in capsys.readouterr().out
    result = json.loads((tmp_path / "feature_reference_activations.json").read_text())
    assert result["saes"]["Raw"]["features"]["0"]["std"] == 2.


def test_all_controls_generate_and_analyse_with_one_shared_random_set(tmp_path, monkeypatch):
    c = cfg(tmp_path, norm_edit_policy="cap_at_ablation")
    eps = [{"episode_id": q + "__natural", "question_id": q, "partition": "validation",
            "source_split": "validation", "question": "Which city?", "solver_attempt_1": "Paris",
            "critic_advocated_answer": "Lyon", "solver_attempt_2": a2, "critic_feedback": "Lyon",
            "critic_condition": "natural", "critic_noncommittal": False,
            "a2_presentation": "candidates", "candidate_order": "a1_first"}
           for q, a2 in [("a", "Lyon"), ("b", "Paris")]]
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation/interactions.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    (tmp_path / "labels").mkdir()
    bl.write_labels(tmp_path / "labels/labels.csv", [
        {**e, "solver_behavior": "fixture", "label": bl.classify_candidate(e, bl.V020_CANDIDATE)} for e in eps])
    saes = tiny_saes()
    saes["Adapted pretrained SAE"] = tiny_saes(seed=1)["Pretrained SAE"]
    saes["Raw activations"] = ActivationBasis(np.zeros(16))
    features = tiny_features(saes, n_targeted=2, n_random_draws=1)
    features["Raw activations"].pop("ref")
    comp = smoke_components({e["question_id"]: [] for e in eps}, 1, saes, features, max_new_tokens=2)
    comp.probe_vector = np.arange(1., 17.)
    monkeypatch.setattr(replay, "provenance", lambda *args: {})
    run = replay.replay(c, replay.load_episodes(c), comp, tmp_path / "run")
    conds = replay.configured_conditions(c, features)
    rows = replay.read_rows(run / "per_row.jsonl")
    assert len(rows) == 2 * len(conds)
    assert sum(x["experiment"] == replay.DIRECTION for x in conds) == 4
    assert sum(x["experiment"] == replay.PROBE_DIRECTION for x in conds) == 2
    distributed._validate_condition_metadata(rows, conds)
    table = replay.analyse(run)
    probe = table[table.Experiment == replay.PROBE_DIRECTION]
    assert probe["Acceptance Rate (%)"].notna().all()
    assert "Targeted ablation" not in set(probe.Condition)
    assert any("≤" in s for s in table[table.Experiment != replay.PROBE_DIRECTION].Condition)
    assert not any("≤" in s for s in probe.Condition)
    tvr = pd.read_csv(run / "targeted_vs_random.csv")
    p = tvr[tvr.Experiment == replay.PROBE_DIRECTION]
    assert len(p) == 2 and set(p.Control) == {"random direction"}
    assert not p["Targeted − random (pp)"].isna().any()


def test_presets_and_worker_policy_metadata(tmp_path):
    root = Path(__file__).resolve().parents[3]
    broad = replay.ReplayConfig.from_yaml(root / "configs/causal_mixed_layer25.yaml")
    focused = replay.ReplayConfig.from_yaml(root / "configs/causal_topk_mixed_layer25.yaml")
    assert len(broad.saes) == 5 and broad.probe_direction
    assert focused.n_random_draws == 100 and focused.edit_norms == [-.01, -.02]
    assert len(focused.targeted_subsets["TopK SAE"]) == 2
    conds = replay.configured_conditions(cfg(tmp_path, norm_edit_policy="cap_at_ablation"),
                                         {"S": {"targeted": [0], "random": [[1]]}})
    condition = next(x for x in conds if x.get("norm_edit_policy"))
    with pytest.raises(ValueError, match="metadata differs"):
        distributed._validate_condition_metadata([{**condition, "question_id": "q", "norm_edit_policy": "allow"}], conds)


def test_pretrained_and_reconstruction_control_presets():
    # #220: Exp 2 (original and decoder-fine-tuned pretrained SAEs) at the TopK run's standard, and Control 1
    # (each SAE with no ablation, on the reconstruction splice) as its own run
    root = Path(__file__).resolve().parents[3]
    focused = replay.ReplayConfig.from_yaml(root / "configs/causal_topk_mixed_layer25.yaml")
    exp2 = replay.ReplayConfig.from_yaml(root / "configs/causal_pretrained_layer25.yaml")
    control = replay.ReplayConfig.from_yaml(root / "configs/causal_reconstruction_control_layer25.yaml")
    original = "results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big"
    adapted = "results/sae/decoder_finetuning/20261102T205853702830Z/best"
    assert {n: (s["kind"], s["path"]) for n, s in exp2.saes.items()} == {
        "Pretrained SAE": ("gemma_scope", original), "Adapted pretrained SAE": ("gemma_scope", adapted)}
    assert not exp2.targeted_subsets and exp2.claude["generated_errors"] == "exclude_episode"
    same = ("partition", "layer", "interactions_jsonl", "contexts_jsonl", "labels_csv", "label_policy", "solver",
            "random_controls", "n_random_draws", "random_directions", "reconstruction", "error_preserving", "ablation",
            "edit_norms", "norm_edit_policy", "min_replay_label_match", "on_replay_check_fail", "seed",
            "n_bootstrap", "n_permutations")
    assert {k: getattr(exp2, k) for k in same} == {k: getattr(focused, k) for k in same}
    assert {k: v for k, v in exp2.claude.items() if k != "generated_errors"} == \
           {k: v for k, v in focused.claude.items() if k != "generated_errors"}

    assert control.reconstruction and not control.error_preserving and control.ablation
    assert control.edit_norms == [] and control.random_directions == 0 and control.n_random_draws == 1
    assert control.saes == {"TopK SAE": focused.saes["TopK SAE"], **exp2.saes}
    for k in ("feature_selection_summary", "feature_reference", "labels_csv", "interactions_jsonl", "seed", "solver"):
        assert getattr(control, k) == getattr(exp2, k)
    assert len({focused.output_dir, exp2.output_dir, control.output_dir}) == 3

    toy = lambda c: {n: {"targeted": list(range(10)), "random": [list(range(10 + 10 * d, 20 + 10 * d))
                                                                 for d in range(c.n_random_draws)]} for n in c.saes}
    exp2_conds = replay.configured_conditions(exp2, toy(exp2))
    control_conds = replay.configured_conditions(control, toy(control))
    assert len(exp2_conds) == 1 + 2 * 3 * (1 + 100) + 2 * 3 == 613
    assert len(control_conds) == 1 + 3 * 3 == 10
    assert sorted((c["experiment"], c["condition"]) for c in control_conds if c["condition"] == "no_ablation") == \
           sorted((n, "no_ablation") for n in control.saes)
    assert all(c["mode"] in (None, "reconstruction") for c in control_conds)
