"""Alignment, partition discipline and layer selection on a fake shared-Drive tree.

Layer 17 carries a planted signal; layer 8 is noise. The pipeline must find that, must never read
test/intervention rows, must verify the manifest index against the interactions file, and must give
NO recommendation on a tie or when nothing beats chance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mas_sae.experiments.artifacts import sha256_file
from mas_sae.probe import behavior_data as bd
from mas_sae.probe import layer_selection as ls

CLASSES = ["adopted_critic", "retained_a1", "third_answer"]   # solver_response values
CODE = {"adopted_critic": 0, "retained_a1": 1, "third_answer": 1}      # planted signal: dim 0 = adopted, dim 1 = not


def build_tree(root: Path, n_train=700, n_val=220, seed=0, labels_have_episode_id=False,
               corrupt_index=False, swap_source_split=False, old_manifest=False):
    """Fake Drive: results/{train,validation}/interactions.jsonl, acts/layer_NN/*_attempt2.pt,
    labels.csv and a four-partition manifest."""
    rng = np.random.default_rng(seed)
    q, rows = 0, []
    for split, n in (("train", n_train), ("validation", n_val)):
        for i in range(n):
            partition = ("validation" if split == "validation"
                         else str(rng.choice(["train", "test", "intervention"], p=[0.8, 0.1, 0.1])))
            rows.append(dict(question_id=f"q{q:05d}", source_split=split, row=i, partition=partition,
                             target=str(rng.choice(CLASSES, p=[0.6, 0.2, 0.2])),   # = solver_response
                             eligible=bool(rng.random() < 0.9)))
            q += 1
    truth = pd.DataFrame(rows)

    for split in ("train", "validation"):
        sub = truth[truth.source_split == split]
        d = root / "results" / split
        d.mkdir(parents=True)
        with open(d / "interactions.jsonl", "w") as f:
            for r in sub.itertuples():
                f.write(json.dumps({
                    "episode_id": f"{r.question_id}__natural", "question_id": r.question_id,
                    "attempt2_activation_index": int(r.row),
                    "solver_attempt_1": f"a1 {'alpha' if r.target == 'adopted_critic' else 'beta'} {rng.integers(1000)}",
                    "critic_advocated_answer": "x", "critic_feedback": "fb"}) + "\n")

    for layer, signal in ((8, 0.0), (17, 2.0)):
        d = root / "acts" / f"layer_{layer:02d}"
        d.mkdir(parents=True)
        for split in ("train", "validation"):
            sub = truth[truth.source_split == split]
            X = rng.normal(size=(len(sub), 16)).astype(np.float32)
            for i, t in enumerate(sub.target):
                X[i, CODE[t]] += signal
            torch.save(torch.from_numpy(X), d / f"{split}_attempt2.pt")

    # real labels.csv: primary_target is numeric (1.0 adopted / 0.0 not / blank if not eligible);
    # it has question_id but NO episode_id, and carries its own source_split/partition columns.
    lab = pd.DataFrame({"question_id": truth.question_id, "source_split": truth.source_split,
                        "partition": truth.partition, "solver_response": truth.target,
                        "rule_version": "behavior_v0.1.1_candidate",
                        "eligible_primary": truth.eligible,
                        "primary_target": np.where(truth.eligible, (truth.target == "adopted_critic").astype(float), np.nan)})
    if labels_have_episode_id:
        lab.insert(0, "episode_id", truth.question_id + "__natural")
    lab.to_csv(root / "labels.csv", index=False)

    index = truth.row.to_numpy().copy()
    if corrupt_index:
        index = np.roll(index, 5)
    man = pd.DataFrame({"question_id": truth.question_id, "source_split": truth.source_split,
                        "partition": truth.partition, "hop_group": "2hop",
                        "full_index": index, "scan_index": index})
    if swap_source_split:
        man.loc[:4, "source_split"] = np.where(man.loc[:4, "source_split"] == "train", "validation", "train")
    if old_manifest:
        man = man.rename(columns={"partition": "canonical_split"}).drop(columns=["source_split"])
    man.to_csv(root / "manifest.csv", index=False)
    return truth


def make_cfg(root: Path, **kw):
    base = dict(capstone_data_root=root, activations_dir="acts", results_dir="results",
                labels_csv="labels.csv", manifest_csv="manifest.csv", index_column="full_index",
                layers=[8, 17], n_bootstrap=100, output_dir=root / "out")
    base.update(kw)
    return ls.LayerSelectionConfig(**base)


def _summary(run_dir):
    return json.loads((run_dir / "layer_selection_summary.json").read_text())


# ------------------------------------------------------------------ partition discipline

def test_sealed_partitions_are_never_selected_and_cannot_be_requested(tmp_path):
    truth = build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    episodes = bd.build_episode_table(tmp_path / "results", ["train", "validation"], [])
    aligned, _ = bd.align_episodes(episodes, pd.read_csv(tmp_path / "labels.csv"),
                                   pd.read_csv(tmp_path / "manifest.csv"), bd.Columns(), "full_index")
    chosen, _ = bd.select_partitions(aligned, bd.Columns(), ["train", "validation"])
    assert set(chosen["partition"]) == {"train", "validation"}
    ok = truth[truth.partition.isin(["train", "validation"]) & truth.eligible]
    assert set(chosen["ep_question_id"]) == set(ok.question_id)
    for sealed in ("test", "intervention"):
        try:
            bd.select_partitions(aligned, bd.Columns(), [sealed])
        except PermissionError as e:
            assert "sealed" in str(e)
        else:
            raise AssertionError(f"{sealed} was released during development")


def test_holdout_fits_on_train_only_and_scores_on_validation_only(tmp_path):
    truth = build_tree(tmp_path)
    run_dir = ls.run(make_cfg(tmp_path))
    comp = _summary(run_dir)["comparison"]
    el = truth[truth.eligible]
    assert comp["n_rows_fit"] == int(((el.partition == "train")).sum())
    assert comp["n_rows_evaluated"] == int(((el.partition == "validation")).sum())
    used = pd.read_csv(run_dir / "rows_and_folds.csv")
    assert set(used["partition"]) == {"train", "validation"}


# ------------------------------------------------------------------------- alignment

def test_labels_stay_attached_to_the_right_activation_rows(tmp_path):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path)
    selected, _, cols = ls.load_inputs(cfg)
    X = bd.load_layer_matrix(selected, tmp_path / "acts" / "layer_17")
    d = [0 if t == "adopted_critic" else 1 for t in selected["target_name"]]
    own = np.array([X[i, k] for i, k in enumerate(d)])
    other = np.array([X[i, 1 - k] for i, k in enumerate(d)])
    assert own.mean() - other.mean() > 1.5             # planted +2 sits on exactly the labelled class


def test_labels_join_uses_question_id_as_the_real_labels_file_has_no_episode_id(tmp_path):
    build_tree(tmp_path)
    _, diag, _ = ls.load_inputs(make_cfg(tmp_path))
    assert diag["labels_join"] == "question_id"


def test_labels_join_prefers_episode_id_when_present(tmp_path):
    build_tree(tmp_path, labels_have_episode_id=True)
    _, diag, _ = ls.load_inputs(make_cfg(tmp_path))
    assert diag["labels_join"] == "episode_id"


def test_numeric_target_gets_readable_names_and_unexpected_values_are_refused():
    assert list(bd.target_names(pd.Series([1.0, 0.0, 1.0]))) == ["adopted_critic", "not_adopted", "adopted_critic"]
    try:
        bd.target_names(pd.Series([1.0, 2.0]))
    except bd.AlignmentError as e:
        assert "unexpected" in str(e)
    else:
        raise AssertionError("an unexpected target value was accepted")


def test_index_that_disagrees_with_interactions_is_rejected(tmp_path):
    build_tree(tmp_path, corrupt_index=True)
    try:
        ls.load_inputs(make_cfg(tmp_path))
    except bd.AlignmentError as e:
        assert "disagrees" in str(e)
    else:
        raise AssertionError("a misaligned manifest index was accepted")


def test_source_split_mismatch_is_rejected(tmp_path):
    build_tree(tmp_path, swap_source_split=True)
    try:
        ls.load_inputs(make_cfg(tmp_path))
    except bd.AlignmentError as e:
        assert "source_split" in str(e)
    else:
        raise AssertionError("a source_split mismatch was accepted")


def test_old_canonical_split_manifest_is_rejected_with_a_clear_message(tmp_path):
    build_tree(tmp_path, old_manifest=True)
    try:
        ls.load_inputs(make_cfg(tmp_path))
    except bd.AlignmentError as e:
        assert "four-partition" in str(e) and "partition" in str(e)
    else:
        raise AssertionError("the old manifest was accepted")


def test_misnamed_text_field_fails_loudly_instead_of_emptying_the_baseline(tmp_path):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path, text_fields=["solver_attempt_1", "no_such_field"])
    try:
        ls.load_inputs(cfg)
    except bd.AlignmentError as e:
        assert "no_such_field" in str(e)
    else:
        raise AssertionError("an all-empty text column was accepted")


# ------------------------------------------------------------------------ the protocols

def test_holdout_finds_planted_layer_and_beats_noise_layer(tmp_path):
    build_tree(tmp_path)
    comp = _summary(ls.run(make_cfg(tmp_path)))["comparison"]
    assert comp["protocol"] == "holdout" and comp["recommended_layer"] == "layer_17"
    r17, r08 = comp["results"]["layer_17"], comp["results"]["layer_08"]
    assert r17["balanced_accuracy"] > r08["balanced_accuracy"] + 0.15
    assert comp["chance_balanced_accuracy"] == 0.5 and comp["classes"] == ["adopted_critic", "not_adopted"]
    assert r17["balanced_accuracy"] > comp["chance_balanced_accuracy"] + 0.2
    assert comp["top_layer_ci_lower_above_chance"] is True
    by_resp = r17["recall_by_solver_response"]            # the 3-way breakdown stays visible
    assert set(by_resp) == set(CLASSES) and sum(v["n"] for v in by_resp.values()) == comp["n_rows_evaluated"]
    assert by_resp["retained_a1"]["recall"] > 0.5 and by_resp["third_answer"]["recall"] > 0.5
    assert "text_only_baseline" in comp["results"]


def test_cv_protocol_uses_only_development_partitions_and_finds_planted_layer(tmp_path):
    build_tree(tmp_path)
    run_dir = ls.run(make_cfg(tmp_path, protocol="cv"))
    comp = _summary(run_dir)["comparison"]
    assert comp["protocol"] == "cv" and comp["recommended_layer"] == "layer_17"
    used = pd.read_csv(run_dir / "rows_and_folds.csv")
    assert set(used["partition"]) == {"train", "validation"}      # test/intervention never touched
    assert used["question_id"].is_unique and set(used["fold"]) == set(range(5))
    assert set(used["solver_response"]) == set(CLASSES)


def _toy(seed=0, n=300, d=16, signal=0.0):
    rng = np.random.default_rng(seed)
    y = rng.choice(CLASSES, size=n, p=[0.5, 0.25, 0.25])
    X = rng.normal(size=(n, d)).astype(np.float32)
    for i, c in enumerate(y):
        X[i, CLASSES.index(c)] += signal
    return y, np.arange(n).astype(str), X


def test_exact_tie_gives_no_recommendation_cv_and_holdout():
    y, groups, X = _toy(signal=2.0)
    cfg = ls.LayerSelectionConfig(n_bootstrap=50, c_grid=[0.1, 1.0])
    two = {"layer_a": X, "layer_b": X.copy()}
    cv = ls.compare_layers(two, y, groups, None, cfg)
    is_fit = np.arange(len(y)) < 200
    ho = ls.compare_layers_holdout(two, y, groups, is_fit, None, cfg)
    for res in (cv, ho):
        assert res["recommended_layer"] is None and res["no_recommendation_reason"] == "tie"
        assert len(res["tied_with_top"]) == 1


def test_no_recommendation_when_nothing_beats_chance():
    y, groups, X1 = _toy(seed=1)
    _, _, X2 = _toy(seed=2)                       # two pure-noise layers
    cfg = ls.LayerSelectionConfig(n_bootstrap=100, c_grid=[0.1, 1.0])
    res = ls.compare_layers({"layer_a": X1, "layer_b": X2}, y, groups, None, cfg)
    assert res["recommended_layer"] is None
    assert res["no_recommendation_reason"] in {"tie", "top_layer_not_distinguishable_from_chance"}


def test_tie_tolerance_widens_what_counts_as_a_tie():
    y, groups, X_strong = _toy(signal=2.0)
    _, _, X_weak = _toy(seed=3, signal=1.2)
    base = dict(n_bootstrap=50, c_grid=[0.1, 1.0])
    clear = ls.compare_layers({"a": X_strong, "b": X_weak}, y, groups, None, ls.LayerSelectionConfig(**base))
    wide = ls.compare_layers({"a": X_strong, "b": X_weak}, y, groups, None,
                             ls.LayerSelectionConfig(tie_tolerance=1.0, **base))
    assert clear["recommended_layer"] is not None and wide["recommended_layer"] is None


def test_label_shuffle_control_lands_near_chance_for_both_protocols(tmp_path):
    build_tree(tmp_path)
    for protocol in ("holdout", "cv"):
        comp = _summary(ls.run(make_cfg(tmp_path, protocol=protocol, n_permutations=8)))["comparison"]
        ctrl = comp["label_shuffle_control"]["layer_17"]
        assert abs(ctrl["null_mean"] - 0.5) < 0.1
        assert ctrl["observed"] > ctrl["null_p95"]


def test_exported_layout_is_verified_through_the_original_row_index():
    """Exported partition files re-index rows (attempt2_activation_index) and keep the original row in
    source_activation_index; the manifest's full_index refers to the ORIGINAL row."""
    ep = pd.DataFrame({"source_split": ["train", "train"], "activation_row": [0, 1], "verify_index": [10, 11],
                       "ep_episode_id": ["a__natural", "b__natural"], "ep_question_id": ["a", "b"]})
    lab = pd.DataFrame({"episode_id": ["a__natural", "b__natural"], "primary_target": ["adopted_critic"] * 2,
                        "eligible_primary": [True, True]})
    man = pd.DataFrame({"question_id": ["a", "b"], "source_split": ["train", "train"],
                        "partition": ["train", "train"], "full_index": [10, 11]})
    df, diag = bd.align_episodes(ep, lab, man, bd.Columns(), "full_index")
    assert diag["index_verified_rows"] == 2 and list(df["activation_row"]) == [0, 1]   # tensors use the new rows
    man_bad = man.assign(full_index=[10, 99])
    try:
        bd.align_episodes(ep, lab, man_bad, bd.Columns(), "full_index")
    except bd.AlignmentError as e:
        assert "disagrees" in str(e)
    else:
        raise AssertionError("a wrong original index was accepted for exported data")


def _tiny():
    """Three exported-layout episodes (a, b, c) with matching labels and manifest rows."""
    ep = pd.DataFrame({"source_split": ["train"] * 3, "activation_row": [0, 1, 2], "verify_index": [10, 11, 12],
                       "ep_episode_id": ["a__n", "b__n", "c__n"], "ep_question_id": ["a", "b", "c"]})
    lab = pd.DataFrame({"question_id": ["a", "b", "c"], "primary_target": [1.0, 0.0, 1.0],
                        "eligible_primary": [True] * 3})
    man = pd.DataFrame({"question_id": ["a", "b", "c"], "source_split": ["train"] * 3,
                        "partition": ["train"] * 3, "full_index": [10, 11, 12]})
    return ep, lab, man


def _raises_alignment(*args, match, **kwargs):
    try:
        bd.align_episodes(*args, **kwargs)
    except bd.AlignmentError as e:
        assert match in str(e), str(e)
        return str(e)
    raise AssertionError("misaligned input was accepted")


def test_episode_missing_from_labels_stops_the_run_instead_of_dropping_out():
    ep, lab, man = _tiny()
    msg = _raises_alignment(ep, lab[lab.question_id != "b"], man, bd.Columns(), "full_index",
                            match="no row in labels.csv")
    assert "b" in msg


def test_episode_missing_from_manifest_stops_the_run_instead_of_dropping_out():
    ep, lab, man = _tiny()
    _raises_alignment(ep, lab, man[man.question_id != "c"], bd.Columns(), "full_index",
                      match="no row in the manifest")


def test_recorded_exclusion_is_dropped_and_reported():
    ep, lab, man = _tiny()
    reason = "no label: Critic output was truncated"
    df, diag = bd.align_episodes(ep, lab[lab.question_id != "b"], man, bd.Columns(), "full_index",
                                 excluded={"b": reason})
    assert list(df["ep_question_id"]) == ["a", "c"]
    assert diag["n_excluded"] == 1 and diag["excluded_episodes"] == {"b": reason}
    assert diag["n_joined"] == 2 and diag["index_verified_rows"] == 2


def test_exclusion_naming_a_question_that_is_not_in_the_run_is_refused():
    ep, lab, man = _tiny()
    _raises_alignment(ep, lab, man, bd.Columns(), "full_index", excluded={"zzz": "typo"},
                      match="not episodes of this run")


def test_partially_missing_activation_index_raises():
    ep, lab, man = _tiny()
    man = man.assign(full_index=[10, None, 12])
    msg = _raises_alignment(ep, lab, man, bd.Columns(), "full_index", match="have no full_index")
    assert msg.startswith("1 of 3")


def test_entirely_missing_activation_index_raises_instead_of_warning():
    ep, lab, man = _tiny()
    man = man.assign(full_index=[None, None, None])
    msg = _raises_alignment(ep, lab, man, bd.Columns(), "full_index", match="have no full_index")
    assert msg.startswith("3 of 3")


def test_unknown_partition_value_is_refused():
    ep, lab, man = _tiny()
    _raises_alignment(ep, lab, man.assign(partition=["train", "dev", "train"]), bd.Columns(), "full_index",
                      match="unknown partition")


def _eval_rows(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["adopted_critic", "not_adopted"] * (n // 2))
    return rng, y, np.arange(n).astype(str)


def _flip(y, idx):
    p = y.copy()
    p[idx] = np.where(p[idx] == "adopted_critic", "not_adopted", "adopted_critic")
    return p


def test_highest_observed_layer_is_not_reported_as_separated_when_a_close_layer_is_within_noise():
    rng, y, groups = _eval_rows()
    errs_a = rng.choice(len(y), 20, replace=False)
    errs_b = np.concatenate([errs_a[:16], rng.choice(np.setdiff1d(np.arange(len(y)), errs_a), 6, replace=False)])
    preds = {"layer_a": _flip(y, errs_a), "layer_b": _flip(y, errs_b)}
    res = ls._evaluate(preds, {k: {} for k in preds}, y, groups, ["adopted_critic", "not_adopted"],
                       ls.LayerSelectionConfig(n_bootstrap=300))
    assert res["highest_observed_layer"] == "layer_a"
    assert res["separated_from_all_other_layers"] is False
    assert res["not_separated_from_highest"] == ["layer_b"]
    assert res["recommendation_basis"] == "highest_observed_not_separated"
    assert "NOT statistically separated" in res["selection_statement"] and "layer_b" in res["selection_statement"]


def test_highest_observed_layer_is_reported_as_separated_when_every_other_layer_is_clearly_worse():
    rng, y, groups = _eval_rows()
    preds = {"layer_a": _flip(y, rng.choice(len(y), 10, replace=False)),
             "layer_b": rng.permutation(y)}                       # chance-level layer
    res = ls._evaluate(preds, {k: {} for k in preds}, y, groups, ["adopted_critic", "not_adopted"],
                       ls.LayerSelectionConfig(n_bootstrap=300))
    assert res["recommended_layer"] == "layer_a"
    assert res["separated_from_all_other_layers"] is True and res["not_separated_from_highest"] == []
    assert res["recommendation_basis"] == "highest_observed_and_separated"


def test_probes_are_class_weighted_and_recover_the_rare_class():
    """The target is ~10% not-adopted; a probe that ignored the imbalance would predict 'adopted' always."""
    assert ls._activation_model(1.0, 0).named_steps["clf"].class_weight == "balanced"
    assert ls._text_model(1.0, 0).named_steps["clf"].class_weight == "balanced"
    rng = np.random.default_rng(0)
    y = np.where(rng.random(600) < 0.1, "not_adopted", "adopted_critic")
    X = rng.normal(size=(600, 16)).astype(np.float32)
    X[y == "not_adopted", 0] += 1.5
    res = ls.compare_layers_holdout({"layer_x": X}, y, np.arange(600).astype(str), np.arange(600) < 400,
                                    None, ls.LayerSelectionConfig(n_bootstrap=50, c_grid=[0.01, 0.1, 1.0]))
    assert res["results"]["layer_x"]["per_class"]["not_adopted"]["recall"] > 0.5


def test_summary_records_label_hash_rule_version_and_package_versions(tmp_path):
    build_tree(tmp_path)
    summary = _summary(ls.run(make_cfg(tmp_path)))
    prov = summary["provenance"]
    assert prov["labels_sha256"] == sha256_file(tmp_path / "labels.csv")
    assert prov["manifest_sha256"] == sha256_file(tmp_path / "manifest.csv")
    assert set(prov["packages"]) == set(ls.PROVENANCE_PACKAGES) and prov["python"]
    assert summary["data_diagnostics"]["label_rule_versions"] == ["behavior_v0.1.1_candidate"]


def test_negative_activation_row_is_rejected():
    ep, lab, man = _tiny()
    _raises_alignment(ep.assign(activation_row=[0, -1, 2]), lab, man, bd.Columns(), "full_index",
                      match="negative activation_row")


def test_negative_index_is_rejected_even_when_manifest_and_interactions_agree():
    ep, lab, man = _tiny()
    _raises_alignment(ep.assign(verify_index=[10, -1, 12]), lab, man.assign(full_index=[10, -1, 12]),
                      bd.Columns(), "full_index", match="negative verify_index")


def test_non_integral_manifest_index_is_rejected_instead_of_truncated():
    ep, lab, man = _tiny()
    _raises_alignment(ep, lab, man.assign(full_index=[10.0, 11.5, 12.0]), bd.Columns(), "full_index",
                      match="not an integer row index")


def test_integral_float_manifest_index_from_csv_is_accepted():
    ep, lab, man = _tiny()
    df, _ = bd.align_episodes(ep, lab, man.assign(full_index=[10.0, 11.0, 12.0]), bd.Columns(), "full_index")
    assert list(df["m_index"]) == [10, 11, 12]


def test_two_questions_referencing_the_same_activation_row_are_rejected():
    ep, lab, man = _tiny()
    _raises_alignment(ep.assign(activation_row=[0, 1, 1]), lab, man, bd.Columns(), "full_index",
                      match="share an activation row")


def _write_interactions(path: Path, indices):
    path.mkdir(parents=True)
    with open(path / "interactions.jsonl", "w") as f:
        for i, idx in enumerate(indices):
            f.write(json.dumps({"question_id": f"q{i}", "attempt2_activation_index": idx,
                                "solver_attempt_1": "a"}) + "\n")


@pytest.mark.parametrize("bad", [1.5, "1", None, True])
def test_non_integer_index_in_interactions_is_rejected(tmp_path, bad):
    _write_interactions(tmp_path / "train", [0, bad])
    with pytest.raises(bd.AlignmentError, match="not an integer row index"):
        bd.build_episode_table(tmp_path, ["train"], ["solver_attempt_1"])


def _matrix_rows(rows):
    return pd.DataFrame({"source_split": ["train"] * len(rows), "activation_row": rows})


@pytest.mark.parametrize("rows,match", [
    ([0, 3], "has 3 rows"),          # out of range
    ([0, -1], "has 3 rows"),         # negative: NumPy would read the last row
    ([1, 1], "same row"),            # two episodes, one row
])
def test_load_layer_matrix_rejects_invalid_rows(tmp_path, rows, match):
    layer = tmp_path / "layer_33"
    layer.mkdir()
    torch.save(torch.arange(6, dtype=torch.float32).reshape(3, 2), layer / "train_attempt2.pt")
    with pytest.raises(bd.AlignmentError, match=match):
        bd.load_layer_matrix(_matrix_rows(rows), layer)


def test_load_layer_matrix_rejects_float_rows(tmp_path):
    layer = tmp_path / "layer_33"
    layer.mkdir()
    torch.save(torch.zeros(3, 2), layer / "train_attempt2.pt")
    with pytest.raises(bd.AlignmentError, match="not integer"):
        bd.load_layer_matrix(_matrix_rows([0.0, 1.0]), layer)


@pytest.mark.parametrize("values,expected", [
    ([True, False], [True, False]),
    (["true", "false"], [True, False]),
    (["TRUE", " False "], [True, False]),
    (["yes", "no"], [True, False]),
    (["1", "0"], [True, False]),
    ([1, 0], [True, False]),
    ([1.0, 0.0], [True, False]),
])
def test_as_bool_accepts_known_representations(values, expected):
    assert bd.as_bool(pd.Series(values, name="eligible_primary")).tolist() == expected


@pytest.mark.parametrize("values", [
    [True, None], ["true", np.nan], ["TRUEE", "false"], ["true", "maybe"], [1, 2], [0.5, 1.0],
])
def test_as_bool_rejects_nulls_and_malformed_values(values):
    with pytest.raises(bd.AlignmentError, match="not a recognised boolean"):
        bd.as_bool(pd.Series(values, name="eligible_primary"))


def test_malformed_eligibility_stops_selection_instead_of_dropping_rows():
    ep, lab, man = _tiny()
    df, _ = bd.align_episodes(ep, lab.assign(eligible_primary=["true", "TRUEE", "false"]), man,
                              bd.Columns(), "full_index")
    with pytest.raises(bd.AlignmentError, match="TRUEE"):
        bd.select_partitions(df, bd.Columns(), ["train"])


def test_defaults_and_entry_point_use_the_official_full9_export():
    repo = Path(__file__).resolve().parents[3]
    cfg = ls.LayerSelectionConfig.from_yaml(repo / "configs/layer_selection_full9.yaml")
    default = ls.LayerSelectionConfig()
    for c in (cfg, default):
        assert c.layers == [4, 8, 17, 25, 29, 30, 31, 32, 33]
        assert "natural_4b_full9_partitioned" in c.activations_dir
        assert "natural_4b_full9_partitioned" in c.results_dir
        assert "behavior_v012_official_20261011" in c.labels_csv
        assert "behavior_v012_official_20261011" in c.manifest_csv
        assert (c.protocol, c.fit_partition, c.eval_partition) == ("holdout", "train", "validation")
    assert "layer_selection_full9.yaml" in (repo / "scripts/probe/layer_selection.py").read_text()
    assert not list((repo / "configs").glob("layer_selection_scan*.yaml"))


def test_stale_labels_path_is_refused_until_both_files_are_set(tmp_path):
    build_tree(tmp_path)
    cfg = make_cfg(tmp_path, labels_csv="REPLACE_ME/labels.csv")
    try:
        ls.load_inputs(cfg)
    except SystemExit as e:
        assert "labels_csv" in str(e)
    else:
        raise AssertionError("ran with a placeholder labels path")
