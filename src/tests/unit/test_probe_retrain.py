"""Issue #191: the Adoption-vs-Retention probe contract, family grouping, selectivity, saved probes, and
raw activations as a feature-selection basis (the control the SAEs are compared against).

The fake tree has question families (two related variants each), Claude-style binary labels with
ineligible Agreement episodes, correctness flags, and a planted Adoption signal in dimension 0.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mas_sae.causal import offline
from mas_sae.probe import behavior_data as bd
from mas_sae.probe import feature_selection as fs
from mas_sae.probe import layer_selection as ls

CLASSES = ("adoption", "retention")   # adoption sorts first: decision_function points the other way


def binary_tree(root: Path, n_families=(80, 30), per_family=2, signal=2.5, seed=0, d=8):
    """results/<split>/interactions.jsonl, acts/layer_17/<split>_attempt2.pt, labels.csv, manifest.csv."""
    rng = np.random.default_rng(seed)
    labels, manifest = [], []
    for split, n_fam in zip(("train", "validation"), n_families):
        recs, X = [], []
        for f in range(n_fam):
            for k in range(per_family):
                qid = f"synthetic:{split}{f:03d}_{k}"
                agreement = rng.random() < 0.15
                label = "agreement" if agreement else str(rng.choice(CLASSES))
                x = rng.normal(size=d)
                x[0] += signal * (label == "adoption")
                row = len(recs)
                recs.append({"episode_id": f"{qid}__natural", "question_id": qid,
                             "attempt2_activation_index": row, "question": "q", "solver_attempt_1": "a1",
                             "critic_advocated_answer": "c", "candidate_order": str(rng.choice(["a1_first", "critic_first"])),
                             "split_group_id": f"family:{split}{f:03d}", "dataset_source": "synthetic",
                             "hop_group": "2hop", "solver_attempt_1_correct": bool(rng.random() < 0.5),
                             "critic_feedback_correct": bool(rng.random() < 0.7)})
                X.append(x)
                labels.append({"question_id": qid, "source_split": split, "partition": split,
                               "eligible_primary": not agreement, "behavior_label": label,
                               "primary_target": np.nan if agreement else float(label == "adoption")})
                manifest.append({"question_id": qid, "source_split": split, "partition": split, "full_index": row})
        (root / "results" / split).mkdir(parents=True)
        (root / "results" / split / "interactions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (root / "acts" / "layer_17").mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(np.array(X), dtype=torch.float32), root / "acts" / "layer_17" / f"{split}_attempt2.pt")
    pd.DataFrame(labels).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False)
    return root


def ls_cfg(root: Path, **kw) -> ls.LayerSelectionConfig:
    base = dict(capstone_data_root=root, activations_dir="acts", results_dir="results", labels_csv="labels.csv",
                manifest_csv="manifest.csv", layers=[17], target_labels=list(CLASSES), label_column="behavior_label",
                text_fields=["solver_attempt_1"], n_bootstrap=50, c_grid=[0.1, 1.0], output_dir=root / "out",
                metadata_baseline_fields=["candidate_order", "correctness_cell"])
    return ls.LayerSelectionConfig(**{**base, **kw})


def _selected_frame(labels, targets, eligible=None):
    n = len(labels)
    return pd.DataFrame({"partition": ["train"] * n, "eligible_primary": eligible or [True] * n,
                         "primary_target": targets, "behavior_label": labels,
                         "ep_question_id": [f"q{i}" for i in range(n)]})


# ----------------------------------------------------------------------------- binary contract

@pytest.mark.parametrize("bad", ["agreement", "out", "third_answer"])
def test_only_adoption_and_retention_may_enter_the_binary_target(bad):
    cols = bd.Columns(label="behavior_label")
    df = _selected_frame(["adoption", "retention", bad], [1.0, 0.0, 0.0])
    with pytest.raises(bd.AlignmentError, match="Only 'adoption'"):
        bd.select_partitions(df, cols, ["train"], class_names=CLASSES)


def test_a_target_that_disagrees_with_its_label_is_refused():
    cols = bd.Columns(label="behavior_label")
    df = _selected_frame(["adoption", "retention"], [1.0, 1.0])
    with pytest.raises(bd.AlignmentError, match="disagrees with their label"):
        bd.select_partitions(df, cols, ["train"], class_names=CLASSES)


def test_ineligible_outcomes_are_dropped_and_classes_carry_the_contract_names():
    cols = bd.Columns(label="behavior_label")
    df = _selected_frame(["adoption", "retention", "agreement"], [1.0, 0.0, np.nan], eligible=[True, True, False])
    out, _ = bd.select_partitions(df, cols, ["train"], class_names=CLASSES)
    assert out["target_name"].tolist() == ["adoption", "retention"]


def test_label_column_must_exist_when_the_contract_is_on(tmp_path):
    root = binary_tree(tmp_path)
    pd.read_csv(root / "labels.csv").drop(columns="behavior_label").to_csv(root / "labels.csv", index=False)
    with pytest.raises(bd.AlignmentError, match="no 'behavior_label' column"):
        ls.load_inputs(ls_cfg(root))


def test_config_requires_two_target_labels_for_the_contract(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("label_column: behavior_label\n")
    with pytest.raises(ValueError, match="needs target_labels"):
        ls.LayerSelectionConfig.from_yaml(p)
    p.write_text("target_labels: [adoption]\n")
    with pytest.raises(ValueError, match="two different classes"):
        ls.LayerSelectionConfig.from_yaml(p)


# ----------------------------------------------------------------------------- family grouping

def test_groups_are_families_and_fall_back_to_questions_only_when_no_episode_has_one():
    df = pd.DataFrame({"ep_question_id": ["q1", "q2", "q3"], "meta_split_group_id": ["f1", "f1", "f2"]})
    groups, name = bd.episode_groups(df, "split_group_id")
    assert name == "split_group_id" and groups.tolist() == ["f1", "f1", "f2"]
    df["meta_split_group_id"] = None
    groups, name = bd.episode_groups(df, "split_group_id")
    assert name == "question_id" and groups.tolist() == ["q1", "q2", "q3"]
    df.loc[0, "meta_split_group_id"] = "f1"
    with pytest.raises(bd.AlignmentError, match="while the others do"):
        bd.episode_groups(df, "split_group_id")


def test_holdout_refuses_a_family_on_both_sides():
    y = np.array(["adoption", "retention"] * 4)
    groups = np.array(["f1", "f1", "f2", "f2", "f3", "f3", "f1", "f4"])
    with pytest.raises(ValueError, match="appear in both the fit and the evaluation"):
        ls.compare_layers_holdout({"layer_17": np.zeros((8, 2))}, y, groups, np.arange(8) < 6, None, ls_cfg(Path(".")))


def test_inner_cv_and_bootstrap_never_split_a_family(tmp_path, monkeypatch):
    seen = []
    real = ls.make_group_folds
    monkeypatch.setattr(ls, "make_group_folds", lambda y, groups, *a: seen.append(np.asarray(groups)) or real(y, groups, *a))
    run_dir = ls.run(ls_cfg(binary_tree(tmp_path)))
    assert seen and all(str(g).startswith("family:") for groups in seen for g in groups)
    summary = json.loads((run_dir / "layer_selection_summary.json").read_text())
    assert summary["data_diagnostics"]["group_column"] == "split_group_id"
    rows = pd.read_csv(run_dir / "rows_and_folds.csv")
    assert rows.groupby("split_group_id")["partition"].nunique().max() == 1


# ----------------------------------------------------------------------------- selectivity

def test_shuffle_keeps_a_question_together_only_when_its_rows_share_one_label():
    rng = np.random.default_rng(0)
    y = np.array(["a", "a", "b", "b", "a", "b"])
    units = np.array(["q1", "q1", "q2", "q2", "q3", "q4"])
    out, unit = ls._shuffle_labels(y, units, rng)
    per_question = lambda labels: pd.Series(labels).groupby(units).agg(set)
    assert unit == "question" and (per_question(out).map(len) == 1).all()
    # class counts are preserved across questions (= across rows when each question has one episode)
    assert sorted(per_question(out).map(min)) == sorted(per_question(y).map(min))
    y_mixed = np.array(["a", "b", "b", "b", "a", "b"])        # q1's two episodes differ
    out, unit = ls._shuffle_labels(y_mixed, units, rng)
    assert unit == "episode" and sorted(out) == sorted(y_mixed)


def test_label_shuffle_control_reports_selectivity_for_balanced_accuracy_and_auroc(tmp_path):
    run_dir = ls.run(ls_cfg(binary_tree(tmp_path), n_permutations=3))
    control = json.loads((run_dir / "layer_selection_summary.json").read_text())["comparison"]["label_shuffle_control"]["layer_17"]
    assert control["shuffle_unit"] == "question"
    assert control["selectivity"] == pytest.approx(control["observed"] - control["null_mean"]) and control["selectivity"] > 0.2
    assert control["auroc"]["selectivity"] > 0.2


# ----------------------------------------------------------------------------- AUROC, probes, breakdowns

def test_auroc_points_toward_adoption_and_saved_weights_reproduce_the_probe(tmp_path):
    root = binary_tree(tmp_path)
    run_dir = ls.run(ls_cfg(root))
    res = json.loads((run_dir / "layer_selection_summary.json").read_text())["comparison"]["results"]["layer_17"]
    assert res["auroc"] > 0.9 and res["auroc_ci95"][0] < res["auroc"] < res["auroc_ci95"][1] + 1e-9
    probe = np.load(run_dir / "probes" / "layer_17.npz")
    assert str(probe["positive_class"]) == "adoption"
    X = bd.load_tensor(root / "acts" / "layer_17" / "validation_attempt2.pt")
    labels = pd.read_csv(root / "labels.csv").query("partition == 'validation' and eligible_primary")
    scores = X[pd.read_csv(root / "manifest.csv").set_index("question_id").loc[labels.question_id, "full_index"]] @ probe["w"] + probe["b"]
    assert scores[labels.behavior_label.to_numpy() == "adoption"].mean() > scores[labels.behavior_label.to_numpy() == "retention"].mean()


def test_raw_unit_weights_equal_the_oriented_decision_function():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=3.0, scale=2.0, size=(200, 5))
    y = np.where(X[:, 0] > 3.0, "adoption", "retention")
    model = ls._activation_model(1.0, 0).fit(X, y)
    w, b = ls.raw_unit_weights(model, "adoption")
    assert np.allclose(X @ w + b, ls.positive_scores(model, X, "adoption"))
    assert np.allclose(ls.positive_scores(model, X, "adoption"), -model.decision_function(X))   # classes_[1] is retention


def test_breakdowns_count_each_subgroup_and_include_the_baselines(tmp_path):
    run_dir = ls.run(ls_cfg(binary_tree(tmp_path)))
    table = pd.read_csv(run_dir / "breakdowns.csv")
    rows = pd.read_csv(run_dir / "rows_and_folds.csv").query("partition == 'validation'")
    layer = table[(table.name == "layer_17") & (table.field == "candidate_order")].set_index("value")
    assert layer["n"].to_dict() == rows["candidate_order"].value_counts().to_dict()
    assert {"text_only_baseline", "metadata_baseline"} <= set(table.name)
    assert set(table.field) == {"dataset_source", "correctness_cell", "candidate_order", "hop_group"}


def test_excluded_episodes_file_is_merged_and_conflicting_reasons_are_refused(tmp_path):
    (tmp_path / "ex.json").write_text(json.dumps({"q1": "api_error", "q2": "agreement"}))
    cfg = ls.LayerSelectionConfig(capstone_data_root=tmp_path, excluded_episodes={"q0": "manual"},
                                  excluded_episodes_file="ex.json")
    assert ls.excluded_episodes(cfg) == {"q0": "manual", "q1": "api_error", "q2": "agreement"}
    cfg.excluded_episodes = {"q1": "something else"}
    with pytest.raises(ValueError, match="different reasons"):
        ls.excluded_episodes(cfg)


# ----------------------------------------------------------------------------- raw activations as a basis

def test_activation_basis_ablation_is_exact_mean_ablation():
    rng = np.random.default_rng(2)
    X = rng.normal(loc=5.0, size=(50, 6)).astype(np.float32)
    basis = fs.ActivationBasis(X.mean(0))
    out = offline.decode(basis, offline.ablate(offline.encode(basis, X), [1, 4]))
    assert np.array_equal(out[:, [1, 4]], np.broadcast_to(X.mean(0)[[1, 4]], (50, 2)))
    assert np.allclose(out[:, [0, 2, 3, 5]], X[:, [0, 2, 3, 5]], atol=1e-5)


def test_center_file_is_written_once_and_a_mismatch_fails_loudly(tmp_path):
    X = np.random.default_rng(3).normal(size=(40, 4)).astype(np.float32)
    spec = {"kind": "activations", "path": "centers/l17.npy"}
    assert np.allclose(fs.ensure_center(spec, tmp_path, X), X.mean(0))
    assert np.allclose(fs.ensure_center(spec, tmp_path, X), X.mean(0))       # matches: accepted
    with pytest.raises(ValueError, match="not the mean of these fit rows"):
        fs.ensure_center(spec, tmp_path, X + 1.0)


def test_raw_latent_statistics_do_not_reuse_firing_frequency_or_a_positive_reference():
    X = np.random.default_rng(4).normal(scale=[1.0, 3.0, 0.5], size=(500, 3)).astype(np.float32)
    stats = fs.latent_stats(fs.ActivationBasis(X.mean(0)), X)
    assert stats["basis"] == "activations" and stats["firing_frequency"] is None and stats["ref"] is None
    assert np.allclose(stats["rms"], X.std(0), rtol=1e-3) and stats["decoder_norm"] == [1.0, 1.0, 1.0]


def _analyse(X_fit, y_fit, X_eval, y_eval, cfg):
    g_fit, g_eval = np.arange(len(y_fit)).astype(str), (np.arange(len(y_eval)) + 10_000).astype(str)
    readout, _ = offline.fit_readout(X_fit, y_fit, g_fit, cfg.c_grid, cfg.inner_folds, cfg.seed)
    return fs.analyse_sae("Raw activations", fs.ActivationBasis(X_fit.mean(0)), X_fit, y_fit, g_fit,
                          X_eval, y_eval, g_eval, readout, cfg)


def _signal_rows(seed, n=300):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    X = rng.normal(scale=np.linspace(0.5, 4.0, 12), size=(n, 12)).astype(np.float32)
    X[:, 0] += 2.0 * y
    X[:, 1] -= 2.0 * y              # promotes retention: only the signed set may hold it
    X[:, 11] = 7.0                  # a constant neuron never varies, so it is never live
    return X, y


def test_raw_basis_selects_from_fit_rows_only_and_matches_controls_on_std():
    cfg = fs.FeatureSelectionConfig(capstone_data_root=Path("."), activations_dir="", results_dir="", labels_csv="",
                                    manifest_csv="", n_targeted=2, n_random_draws=5, n_bootstrap=20,
                                    n_permutations=50, c_grid=[0.1, 1.0])
    X_fit, y_fit = _signal_rows(5)
    X_eval, y_eval = _signal_rows(6, 120)
    res = _analyse(X_fit, y_fit, X_eval, y_eval, cfg)
    s = res["summary"]
    assert s["targeted_features"][0] == 0 and 11 not in s["targeted_features"] + sum(s["random_control_draws"], [])
    assert 1 in s["targeted_features_signed"] and 1 not in s["targeted_features"]
    assert res["ranking"]["train_firing_frequency"].isna().all() and s["mean_active_features_eval"] is None
    # controls are matched on per-neuron std (every centered neuron "fires"), not on firing frequency
    spread = offline.encode(fs.ActivationBasis(X_fit.mean(0)), X_fit).std(0)
    assert s["random_control_draws"] == offline.contribution_matched_controls(
        np.ones_like(spread), spread, s["targeted_features"], cfg.n_random_draws, cfg.seed)
    assert s["selection_rules"]["random_control_draws"] == "matched on per-neuron std"
    d = res["direction"]
    assert d["w"][11] == 0 and d["w"][0] > 0 and d["w"][1] < 0
    changed = _analyse(X_fit, y_fit, X_eval * 3.0 + 1.0, y_eval, cfg)          # eval rows never steer selection
    assert changed["summary"]["targeted_features"] == s["targeted_features"]
    assert changed["summary"]["random_control_draws"] == s["random_control_draws"]
    assert np.array_equal(changed["direction"]["w"], d["w"])


def test_feature_selection_run_uses_the_contract_positive_class_and_saves_the_direction(tmp_path):
    root = binary_tree(tmp_path, d=8)
    cfg = fs.FeatureSelectionConfig(
        capstone_data_root=root, activations_dir="acts", results_dir="results", labels_csv="labels.csv",
        manifest_csv="manifest.csv", layer=17, target_labels=list(CLASSES), label_column="behavior_label",
        saes={"Raw activations": {"kind": "activations", "path": "centers/raw_layer17.npy"}},
        n_targeted=2, n_random_draws=3, n_bootstrap=20, n_permutations=50, c_grid=[0.1, 1.0], output_dir=root / "fs")
    run_dir = fs.run(cfg)
    summary = json.loads((run_dir / "feature_selection_summary.json").read_text())
    assert 0.2 < summary["readout"]["observed_accept_rate_eval"] < 0.8      # adoption = 1, not every row 0
    raw = summary["saes"]["Raw activations"]
    assert raw["targeted_features"][0] == 0 and raw["sae"]["kind"] == "activations"
    assert (run_dir / raw["probe_direction"]["file"]).is_file() and (run_dir / "ranking_Raw activations.csv").is_file()
    assert "centers/raw_layer17.npy" in summary["provenance"]["sae_sha256"]["Raw activations"]
    np.save(root / "centers" / "raw_layer17.npy", np.load(root / "centers" / "raw_layer17.npy") + 1.0)
    with pytest.raises(ValueError, match="not the mean of these fit rows"):
        fs.run(cfg)


# ----------------------------------------------------------------------------- clustered paired statistics

def test_paired_effect_is_unchanged_without_clusters_and_wider_with_family_clusters():
    rng = np.random.default_rng(7)
    d_family = rng.normal(0.1, 0.5, size=100)
    outcome, reference = np.repeat(d_family, 2), np.zeros(200)     # two identical variants per family
    questions, families = np.arange(200).astype(str), np.repeat(np.arange(100), 2).astype(str)
    plain = offline.paired_effect(outcome, reference, questions, 400, 400, 0)
    assert plain == offline.paired_effect(outcome, reference, questions, 400, 400, 0, clusters=None)
    clustered = offline.paired_effect(outcome, reference, questions, 400, 400, 0, clusters=families)
    width = lambda r: r["ci95_pp"][1] - r["ci95_pp"][0]
    assert width(clustered) > 1.2 * width(plain)
