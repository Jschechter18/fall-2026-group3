"""Mechanics of SAE feature selection and the offline causal proxy (small hand-built arrays)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from mas_sae.causal import offline
from mas_sae.probe import feature_selection as fs
from mas_sae.probe import layer_selection as ls
from mas_sae.sae.pretrained import JumpReLUSAE, load_gemma_scope_sae


def _jump_relu(d_in=4, d_sae=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(w_enc=torch.randn(d_in, d_sae, generator=g), b_enc=torch.zeros(d_sae),
                w_dec=torch.randn(d_sae, d_in, generator=g), b_dec=torch.zeros(d_in),
                threshold=torch.full((d_sae,), 0.5))


def test_jump_relu_zeroes_features_below_threshold():
    sae = JumpReLUSAE(**_jump_relu())
    x = torch.randn(10, 4)
    pre = x @ sae.w_enc + sae.b_enc
    z = sae.encoder(x)
    assert torch.equal(z == 0, pre <= sae.threshold)
    assert torch.allclose(z[z != 0], pre[z != 0])
    assert torch.allclose(sae(x)[1], z @ sae.w_dec + sae.b_dec)


def test_jump_relu_never_emits_negative_latents_with_a_negative_threshold():
    params = _jump_relu()
    params["threshold"] = torch.full((6,), -1.0)
    sae = JumpReLUSAE(**params)
    pre = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 3.0, -1.0]])
    sae.w_enc.data, sae.b_enc.data = torch.zeros(4, 6), pre[0]   # pre-activation = b_enc for x = 0
    z = sae.encoder(torch.zeros(1, 4))
    assert torch.equal(z, torch.tensor([[0.0, 0.0, 0.0, 0.5, 3.0, 0.0]]))


def _write_gemma_scope(path, hook="model.layers.33.output", arch="jump_relu"):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"architecture": arch, "hf_hook_point_in": hook}))
    save_file(_jump_relu(), str(path / "params.safetensors"))


def test_gemma_scope_loader_checks_the_hook_point(tmp_path):
    _write_gemma_scope(tmp_path / "ok")
    sae, cfg = load_gemma_scope_sae(tmp_path / "ok", expected_layer=33)
    assert (sae.input_dim, sae.latent_dim) == (4, 6) and cfg["architecture"] == "jump_relu"
    _write_gemma_scope(tmp_path / "wrong", hook="model.layers.31.output")
    with pytest.raises(ValueError, match="not layer 33"):
        load_gemma_scope_sae(tmp_path / "wrong", expected_layer=33)


def test_gemma_scope_loader_rejects_other_architectures(tmp_path):
    _write_gemma_scope(tmp_path / "topk", arch="topk")
    with pytest.raises(ValueError, match="jump_relu"):
        load_gemma_scope_sae(tmp_path / "topk")


def test_ablate_zeroes_only_the_chosen_features_and_copies():
    Z = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    out = offline.ablate(Z, [1, 3])
    assert (out[:, [1, 3]] == 0).all() and np.array_equal(out[:, [0, 2]], Z[:, [0, 2]])
    assert (Z != 0).all()


def test_encode_and_decode_match_the_sae_in_batches():
    sae = JumpReLUSAE(**_jump_relu())
    X = np.random.default_rng(0).normal(size=(7, 4)).astype(np.float32)
    Z = offline.encode(sae, X, batch_size=3)
    assert np.allclose(Z, sae.encoder(torch.from_numpy(X)).numpy(), atol=1e-6)
    assert np.allclose(offline.decode(sae, Z, batch_size=2), sae.decoder(torch.from_numpy(Z)).numpy(), atol=1e-6)


def test_random_controls_are_matched_live_and_disjoint_from_targets():
    freq = np.array([0.0, 0.5, 0.4, 0.01, 0.02, 0.45, 0.0, 0.015, 0.6, 0.03])
    draws = offline.frequency_matched_controls(freq, targeted=[1, 3], n_draws=20, seed=0, pool_size=2)
    assert draws == offline.frequency_matched_controls(freq, [1, 3], 20, seed=0, pool_size=2)
    for d in draws:
        assert len(d) == 2 and len(set(d)) == 2
        assert not {1, 3} & set(d) and all(freq[f] > 0 for f in d)
        assert d[0] in {5, 8} and d[1] in {7, 4}           # nearest firing frequencies (log scale)


def test_paired_effect_reports_pp_ci_and_p_value():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.6, 0.9, 400)
    groups = np.arange(400).astype(str)
    same = offline.paired_effect(ref, ref, groups, 200, 2000, seed=0)
    assert same["delta_pp"] == 0 and same["p_value"] == 1.0
    shifted = offline.paired_effect(ref - 0.05 + rng.normal(0, 0.01, 400), ref, groups, 200, 2000, seed=0)
    assert shifted["delta_pp"] == pytest.approx(-5, abs=0.3)
    assert shifted["ci95_pp"][0] < -5 < shifted["ci95_pp"][1] < 0 and shifted["p_value"] < 0.001
    with pytest.raises(ValueError, match="one row per question"):
        offline.paired_effect(ref[:2], ref[:2], np.array(["q", "q"]), 10, 10, seed=0)


def test_feature_pushing_toward_acceptance_gets_a_positive_coefficient():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.5).astype(int)
    Z = rng.exponential(1.0, size=(600, 5)).astype(np.float32)
    Z[y == 1, 2] += 2.0                                    # feature 2 fires more when accepting
    Z[y == 0, 4] += 2.0                                    # feature 4 fires more when not accepting
    cfg = fs.FeatureSelectionConfig(**{k: "." for k in fs.PATH_KEYS}, c_grid=[1.0], inner_folds=2)
    model, *_ = ls._fit_counting_warnings(lambda c: ls._activation_model(c, 0), Z, y,
                                          np.arange(600).astype(str), cfg)
    r = fs.rank_features(model, Z).set_index("feature")
    assert r.loc[2, "coef_toward_accept"] > 0 > r.loc[4, "coef_toward_accept"]
    assert set(r["mean_abs_shap"].nlargest(2).index) == {2, 4}


def test_readout_probabilities_are_for_acceptance():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.8).astype(int)
    X = rng.normal(size=(400, 3)) + y[:, None]
    readout, _ = offline.fit_readout(X, y, np.arange(400).astype(str), [1.0], 2, seed=0)
    p = offline.p_accept(readout, X)
    assert p[y == 1].mean() > p[y == 0].mean()


def test_results_table_has_the_presentation_columns():
    cond = {"no_ablation": {"acceptance_pct": 90.0},
            "random_ablation": {"acceptance_pct": 89.5, "delta_pp": -0.5, "ci95_pp": [-1.0, 0.0], "p_value": 0.2},
            "targeted_ablation": {"acceptance_pct": 85.0, "delta_pp": -5.0, "ci95_pp": [-6.0, -4.0], "p_value": 0.0001}}
    t = fs.results_table(91.0, 90.5, {"TopK SAE": {"conditions": cond}})
    assert list(t.columns) == ["Experiment", "Condition", "Outcome", "Acceptance Rate (%)",
                               "Δ vs. SAE Control (pp)", "95% CI for Δ", "p-value"]
    topk = t[t.Experiment == "TopK SAE"].set_index("Condition")
    assert topk.loc["No ablation", "Δ vs. SAE Control (pp)"] == "Reference"
    assert topk.loc["Targeted ablation", "95% CI for Δ"] == "[-6.00, -4.00]"
    assert t.loc[0, "Outcome"] == "observed"


PATHS_YAML = "".join(f"{k}: .\n" for k in fs.PATH_KEYS)


def test_config_requires_the_data_paths(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(PATHS_YAML.replace("labels_csv: .\n", ""))
    with pytest.raises(ValueError, match="labels_csv"):
        fs.FeatureSelectionConfig.from_yaml(p)


def test_config_rejects_unknown_sae_kinds(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(PATHS_YAML + "saes:\n  X:\n    kind: something\n    path: a\n")
    with pytest.raises(ValueError, match="kind"):
        fs.FeatureSelectionConfig.from_yaml(p)


def test_config_passes_excluded_episodes_to_the_loader(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(PATHS_YAML + 'excluded_episodes:\n  "2hop__89162_77346": "no frozen partition"\n')
    cfg = fs.FeatureSelectionConfig.from_yaml(p)
    assert cfg.layer_selection_config().excluded_episodes == {"2hop__89162_77346": "no frozen partition"}


def test_committed_config_loads_with_three_saes():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    cfg = fs.FeatureSelectionConfig.from_yaml(repo / "configs/feature_selection_layer33.yaml")
    assert set(cfg.saes) == {"TopK SAE", "L1 SAE", "Pretrained SAE"}
    assert (cfg.layer, cfg.fit_partition, cfg.eval_partition) == (33, "train", "validation")


def test_provenance_hashes_every_input_and_sae_file(tmp_path, monkeypatch):
    for f in ["acts/layer_33/train_attempt2.pt", "acts/layer_33/validation_attempt2.pt",
              "res/train/interactions.jsonl", "res/validation/interactions.jsonl",
              "topk/checkpoints/best_checkpoint.pt", "gs/config.json", "gs/params.safetensors"]:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(f)
    (tmp_path / "cfg.yaml").write_text("x")
    monkeypatch.setattr(fs.ls, "provenance", lambda cfg: {"git_commit": "abc"})
    monkeypatch.setattr(fs, "get_untracked_files", lambda *d: ["src/new.py"])
    monkeypatch.setattr(fs, "get_untracked_sha256", lambda *d: "u")
    cfg = fs.FeatureSelectionConfig(capstone_data_root=tmp_path, activations_dir="acts", results_dir="res",
                                    labels_csv="l", manifest_csv="m",
                                    saes={"T": {"kind": "custom", "path": "topk"},
                                          "G": {"kind": "gemma_scope", "path": "gs"}})
    prov = fs.provenance(cfg, tmp_path / "cfg.yaml")
    assert prov["git_commit"] == "abc" and prov["untracked_code_files"] == ["src/new.py"]
    assert prov["config_sha256"] == fs.sha256_file(tmp_path / "cfg.yaml")
    assert set(prov["input_sha256"]) == {"acts/layer_33/train_attempt2.pt", "acts/layer_33/validation_attempt2.pt",
                                         "res/train/interactions.jsonl", "res/validation/interactions.jsonl"}
    assert set(prov["sae_sha256"]["T"]) == {"topk/checkpoints/best_checkpoint.pt"}
    assert set(prov["sae_sha256"]["G"]) == {"gs/config.json", "gs/params.safetensors"}
