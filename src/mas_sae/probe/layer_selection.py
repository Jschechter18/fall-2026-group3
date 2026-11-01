"""Layer selection for the Solver-Critic activation probe, under the 6 Oct split contract.

Two protocols, both restricted to the DEVELOPMENT partitions (``test`` and ``intervention`` are
sealed and can never be requested):

  holdout  fit the probe on `train`, rank layers on `validation`   (the contract; used by
           configs/layer_selection_full9.yaml, the authoritative run)
  cv       question-grouped CV over `train` + `validation`          (NOT the contract: it fits on
           validation rows; no config uses it)

Common to both: the target is `primary_target` on `eligible_primary` rows (with `target_labels` and
`label_column`, only Adoption = 1 and Retention = 0 may be eligible), the probe is class-weighted with C
chosen inside the training data only, every layer sees identical rows/folds, no fold, holdout or
bootstrap splits a question family (`group_column`), metrics are balanced accuracy + per-class
precision/recall + confusion matrix + bootstrap CIs (+ AUROC and per-subgroup breakdowns for a binary
target), a text-only baseline (what the Solver sees) and an optional metadata baseline are scored the
same way, label-shuffle controls give each layer's selectivity, and ties (or nothing beating chance)
produce NO recommendation. Holdout runs save each layer's probe in raw activation units.

Entry point: ``scripts/layer_selection.py``.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mas_sae.data.production import DEVELOPMENT_PARTITIONS
from mas_sae.experiments.artifacts import (
    get_git_commit, get_git_diff_sha256, get_package_version, get_run_command, sha256_file,
)
from mas_sae.probe import behavior_data as bd
from mas_sae.probe.cv_eval import (
    balanced_accuracy, bootstrap_ci, confusion, grouped_bootstrap_indices,
    macro_f1, make_group_folds, paired_bootstrap_diff, per_class_report,
)
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)

PATH_FIELDS = {"capstone_data_root", "output_dir"}
DEV_PARTITIONS = list(DEVELOPMENT_PARTITIONS)
# non-activation candidates: scored like the layers but never selected as a layer
BASELINES = ("text_only_baseline", "metadata_baseline")
DEFAULT_CLASSES = ("adopted_critic", "not_adopted")
# recorded with every run: results differed slightly across environments for one layer
PROVENANCE_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "pyyaml")


@dataclass
class LayerSelectionConfig:
    # Defaults match configs/layer_selection_full9.yaml: the official full nine-layer export
    # (activations and interactions from the same export) and the Behavior v0.1.2 package.
    capstone_data_root: Path = Path(".")
    # partitioned export: folders are named by partition, not by MuSiQue source split
    activations_dir: str = "data/activations/natural_4b_full9_partitioned"  # <dir>/layer_NN/{split}_attempt2.pt
    results_dir: str = "results/collection/natural_4b_full9_partitioned"    # <dir>/{split}/interactions.jsonl
    # labels.csv and split_manifest.csv come from ONE behavior package folder.
    labels_csv: str = "results/behavior/behavior_v012_official_20261011/labels.csv"
    manifest_csv: str = "results/behavior/behavior_v012_official_20261011/split_manifest.csv"
    index_column: str = "full_index"
    # question_id -> reason, for episodes deliberately left out. Any other episode missing from the
    # labels or the manifest stops the run; these are dropped and recorded in the summary.
    excluded_episodes: dict = field(default_factory=dict)
    layers: list = field(default_factory=lambda: [4, 8, 17, 25, 29, 30, 31, 32, 33])
    source_splits: list = field(default_factory=lambda: ["train", "validation"])  # folder names
    activation_file: str = "{split}_attempt2.pt"           # Solver activations before A2
    # In the lexical packages primary_target is 1 = adopted the Critic, 0 = retained A1 or third answer;
    # solver_response carries the 3-way breakdown and is reported per layer. The Claude probe packages are
    # Adoption (1) vs Retention (0) only: set target_labels and label_column to enforce that.
    target_column: str = "primary_target"
    eligible_column: str = "eligible_primary"
    # Binary contract. target_labels names target 1 and target 0 (e.g. [adoption, retention]); with
    # label_column (e.g. behavior_label) every eligible row must carry one of them with the matching
    # target, so Agreement, third answers, Out or failures can never enter as the negative class.
    target_labels: list | None = None
    label_column: str | None = None
    # JSON {question_id: reason} merged into excluded_episodes (e.g. the probe package's
    # excluded_episodes.json); a question listed in both with different reasons is refused.
    excluded_episodes_file: str | None = None
    # No fold, holdout or bootstrap splits this unit; falls back to the question id for runs that do
    # not record it. Label-shuffle controls still permute per question.
    group_column: str | None = "split_group_id"
    # per-candidate metrics within each value of these episode fields (breakdowns.csv)
    breakdown_fields: list = field(default_factory=lambda: [
        "dataset_source", "correctness_cell", "candidate_order", "hop_group"])
    # one-hot logistic baseline on these episode fields ([] = off); bounds what correctness, position
    # and source alone predict. Never list a label-derived field here.
    metadata_baseline_fields: list = field(default_factory=list)
    protocol: str = "holdout"                              # "holdout" | "cv"
    fit_partition: str = "train"
    eval_partition: str = "validation"
    # text-only baseline: everything the Solver can already SEE in the A2 prompt.
    # Never add solver_attempt_2 or any label-derived field here (that would leak the answer).
    text_fields: list = field(default_factory=lambda: [
        "solver_attempt_1", "critic_advocated_answer", "critic_feedback"])
    n_folds: int = 5                                       # cv protocol only
    seed: int = 42
    # C is chosen INSIDE the training data (inner question-grouped CV on balanced accuracy), never on
    # the evaluation rows. With thousands of features a single fixed C can collapse every layer to a
    # constant prediction, so the grid is wide.
    c_grid: list = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    inner_folds: int = 3
    n_bootstrap: int = 1000
    # Layers whose balanced accuracy differs from the top by <= this are TIED -> no recommendation.
    tie_tolerance: float = 1e-9
    n_permutations: int = 0  # >0: label-shuffle control (selectivity = observed - null mean)
    permutation_layers: str = "top"                        # "top" | "all" layers get the control
    output_dir: Path = Path("results/layer_selection_full9")

    @property
    def class_names(self) -> tuple[str, str]:
        return tuple(self.target_labels) if self.target_labels else DEFAULT_CLASSES

    @classmethod
    def from_yaml(cls, path: Path) -> "LayerSelectionConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for k in PATH_FIELDS & set(raw):
            raw[k] = Path(raw[k])
        cfg = cls(**raw)
        if cfg.protocol not in {"holdout", "cv"}:
            raise ValueError("protocol must be 'holdout' or 'cv'")
        if cfg.target_labels is not None and (len(cfg.target_labels) != 2 or len(set(cfg.target_labels)) != 2):
            raise ValueError("target_labels must name two different classes: [target 1, target 0]")
        if cfg.label_column and not cfg.target_labels:
            raise ValueError("label_column needs target_labels, the two labels allowed in the binary target")
        if cfg.permutation_layers not in {"top", "all"}:
            raise ValueError("permutation_layers must be 'top' or 'all'")
        return cfg


# ----------------------------------------------------------------------------- models

def _activation_model(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def _text_model(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def _metadata_model(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def positive_scores(model, X, positive: str) -> np.ndarray:
    """Decision scores that increase toward ``positive``. A binary ``decision_function`` points toward
    ``classes_[1]``, and string classes sort alphabetically (``adopted_critic`` < ``not_adopted``),
    so the raw scores can point the wrong way."""
    scores = model.decision_function(X)
    return scores if model.classes_[1] == positive else -scores


def raw_unit_weights(model, positive: str) -> tuple[np.ndarray, float]:
    """(w, b) of a scaler + logistic probe in raw activation units, oriented toward ``positive``:
    ``X @ w + b`` equals ``positive_scores(model, X, positive)``."""
    scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
    sign = 1.0 if clf.classes_[1] == positive else -1.0
    w = sign * clf.coef_[0] / scaler.scale_
    b = sign * (clf.intercept_[0] - float(clf.coef_[0] @ (scaler.mean_ / scaler.scale_)))
    return w, float(b)


def _auroc(y_true, scores) -> float:
    """AUROC of ``scores`` for 0/1 ``y_true``; NaN on a resample with a single class."""
    y_true = np.asarray(y_true)
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")


def _fit_with_inner_cv(make_model, X, y, groups, c_grid, inner_folds, seed):
    """Pick C by inner grouped CV on THIS training set only, then refit on all of it.
    Ties go to the smaller C (stronger regularization)."""
    n_inner = int(max(2, min(inner_folds, min(np.unique(y, return_counts=True)[1]))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inner, _ = make_group_folds(y, groups, n_inner, seed)
    best_c, best_score = c_grid[0], -1.0
    for C in c_grid:
        scores = []
        for tr, va in inner:
            m = make_model(C).fit(X[tr], y[tr])
            scores.append(balanced_accuracy(y[va], m.predict(X[va])))
        if np.mean(scores) > best_score + 1e-9:
            best_c, best_score = C, float(np.mean(scores))
    return make_model(best_c).fit(X, y), best_c


def _fit_counting_warnings(make_model, X, y, groups, cfg):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, C = _fit_with_inner_cv(make_model, X, y, groups, cfg.c_grid, cfg.inner_folds, cfg.seed)
    return model, C, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def oof_predict(make_model, X, y, groups, folds, c_grid, inner_folds, seed):
    """Out-of-fold predictions. Scaler/vectorizer AND C are refit inside every outer fold."""
    y, groups = np.asarray(y), np.asarray(groups)
    pred = np.empty(len(y), dtype=y.dtype)
    n_conv, chosen = 0, []
    for train, test in folds:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, C = _fit_with_inner_cv(make_model, X[train], y[train], groups[train],
                                          c_grid, inner_folds, seed)
        n_conv += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        chosen.append(C)
        pred[test] = model.predict(X[test])
    return pred, n_conv, chosen


# ------------------------------------------------------------------------- comparison

def _candidates(layer_matrices, texts, cfg, metadata=None):
    act = lambda C: _activation_model(C, cfg.seed)
    cand = {name: (X, act) for name, X in layer_matrices.items()}
    if texts is not None:
        cand["text_only_baseline"] = (np.asarray(texts, dtype=object),
                                      lambda C: _text_model(C, cfg.seed))
    if metadata is not None:
        cand["metadata_baseline"] = (np.asarray(metadata, dtype=object),
                                     lambda C: _metadata_model(C, cfg.seed))
    return cand


def _evaluate(preds, extras, y, groups, classes, cfg, folds=None, subgroups=None, scores=None) -> dict:
    """Metrics, CIs, paired differences, tie rule. ``y``/``groups`` are the EVALUATED rows; ``scores``
    (name -> decision scores toward the positive class) adds AUROC with a grouped-bootstrap CI."""
    boot = grouped_bootstrap_indices(groups, cfg.n_bootstrap, cfg.seed)
    positive = cfg.class_names[0]
    records = {}
    for name, pred in preds.items():
        lo, hi = bootstrap_ci(y, pred, boot)
        auroc = {}
        if scores is not None and name in scores:
            y01 = (y == positive).astype(int)
            vals = np.array([_auroc(y01[i], scores[name][i]) for i in boot])
            auroc = {"auroc": _auroc(y01, scores[name]),
                     "auroc_ci95": [float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))]}
        rec = {
            "balanced_accuracy": balanced_accuracy(y, pred), "ci95": [lo, hi], **auroc,
            "macro_f1": macro_f1(y, pred, classes),
            "per_class": per_class_report(y, pred, classes),
            "confusion_matrix": {"labels": [str(c) for c in classes],
                                 "matrix": confusion(y, pred, classes).tolist()},
            **extras[name],
        }
        if subgroups is not None:   # e.g. solver_response: did the probe get retained_a1 / third_answer rows right?
            rec["recall_by_solver_response"] = {
                str(g): {"recall": float(np.mean(pred[subgroups == g] == y[subgroups == g])),
                         "n": int((subgroups == g).sum())} for g in np.unique(subgroups)}
        if folds is not None:
            fs = [balanced_accuracy(y[test], pred[test]) for _, test in folds]
            rec.update(per_fold_balanced_accuracy=fs, per_fold_mean=float(np.mean(fs)),
                       per_fold_sd=float(np.std(fs, ddof=1)))
        records[name] = rec

    act_names = [n for n in preds if n not in BASELINES]
    best = max(act_names, key=lambda n: records[n]["balanced_accuracy"])
    for name in preds:
        d, dlo, dhi = paired_bootstrap_diff(y, preds[name], preds[best], boot)
        records[name]["diff_vs_best"] = {"diff": d, "ci95": [dlo, dhi]}
        records[name]["within_noise_of_best"] = bool(dlo <= 0 <= dhi)
    top_ba = records[best]["balanced_accuracy"]
    tied = [n for n in act_names
            if n != best and abs(records[n]["balanced_accuracy"] - top_ba) <= cfg.tie_tolerance]
    above_chance = bool(records[best]["ci95"][0] > 1.0 / len(classes))
    reason = "tie" if tied else (None if above_chance else "top_layer_not_distinguishable_from_chance")
    beats_text = best_vs_text = None
    if "text_only_baseline" in preds:
        d, dlo, dhi = paired_bootstrap_diff(y, preds[best], preds["text_only_baseline"], boot)
        beats_text, best_vs_text = bool(dlo > 0), {"diff": d, "ci95": [dlo, dhi]}
    best_vs_metadata = None
    if "metadata_baseline" in preds:
        d, dlo, dhi = paired_bootstrap_diff(y, preds[best], preds["metadata_baseline"], boot)
        best_vs_metadata = {"diff": d, "ci95": [dlo, dhi], "beats": bool(dlo > 0)}
    # Highest observed score is not the same claim as "better than every other layer": a layer is
    # separated from the top only if the paired-bootstrap CI of its difference excludes 0.
    not_separated = [n for n in act_names if n != best and records[n]["within_noise_of_best"]]
    separated = not not_separated
    if reason:
        basis = None
        statement = f"No layer recommended ({reason}). Highest observed: {best}."
    elif separated:
        basis = "highest_observed_and_separated"
        statement = (f"{best} has the highest observed balanced accuracy and is separated from every "
                     "other layer (paired-bootstrap CI of each difference excludes 0).")
    else:
        basis = "highest_observed_not_separated"
        statement = (f"{best} has the highest observed balanced accuracy but is NOT statistically "
                     f"separated from {', '.join(not_separated)} (paired-bootstrap CI of each difference "
                     "includes 0). Choosing it does not rule those layers out.")
    return {
        "classes": [str(c) for c in classes],
        "n_rows_evaluated": int(len(y)),
        "class_counts_evaluated": {str(c): int((y == c).sum()) for c in classes},
        "chance_balanced_accuracy": 1.0 / len(classes),
        "c_grid": list(cfg.c_grid),
        "highest_observed_layer": best,
        "separated_from_all_other_layers": separated,
        "not_separated_from_highest": not_separated,
        "recommendation_basis": basis,
        "selection_statement": statement,
        "recommended_layer": None if reason else best,   # None on a tie or when nothing beats chance
        "no_recommendation_reason": reason,
        "top_layer_by_balanced_accuracy": best,
        "tied_with_top": tied,
        "top_layer_ci_lower_above_chance": above_chance,
        "top_layer_beats_text_baseline": beats_text,
        "top_layer_vs_text_baseline": best_vs_text,
        "top_layer_vs_metadata_baseline": best_vs_metadata,
        "layers_indistinguishable_from_best": [n for n in act_names if records[n]["within_noise_of_best"]],
        "results": records,
    }


def _permuted_layers(out, layer_matrices, cfg) -> list[str]:
    return list(layer_matrices) if cfg.permutation_layers == "all" else [out["top_layer_by_balanced_accuracy"]]


def compare_layers(layer_matrices, y, groups, texts, cfg, subgroups=None, units=None, metadata=None) -> dict:
    """`cv` protocol: grouped CV over the given rows (all from development partitions). ``units`` are
    what label-shuffle controls permute (questions); they default to ``groups``."""
    y, groups = np.asarray(y), np.asarray(groups)
    units = groups if units is None else np.asarray(units)
    subgroups = None if subgroups is None else np.asarray(subgroups).astype(str)
    classes = sorted(np.unique(y).tolist())
    folds, fold_id = make_group_folds(y, groups, cfg.n_folds, cfg.seed)  # ONE set of folds
    preds, extras = {}, {}
    for name, (X, factory) in _candidates(layer_matrices, texts, cfg, metadata).items():
        pred, n_conv, chosen = oof_predict(factory, X, y, groups, folds, cfg.c_grid, cfg.inner_folds, cfg.seed)
        preds[name], extras[name] = pred, {"chosen_C_per_fold": chosen, "n_convergence_warnings": n_conv}
    out = _evaluate(preds, extras, y, groups, classes, cfg, folds, subgroups)
    out.update(protocol="cv", n_groups=int(len(np.unique(groups))), _fold_id=fold_id.tolist())
    if cfg.n_permutations > 0:
        out["label_shuffle_control"] = {name: _permutation_cv(
            layer_matrices[name], y, groups, units, folds, cfg, preds[name])
            for name in _permuted_layers(out, layer_matrices, cfg)}
    return out


def compare_layers_holdout(layer_matrices, y, groups, is_fit, texts, cfg, subgroups=None, units=None,
                           metadata=None) -> dict:
    """`holdout` protocol: fit on the rows where ``is_fit`` is True, score on all the others.

    ``groups`` (families) are never split by the holdout, the inner CV or the bootstrap; ``units``
    (questions, default ``groups``) are what label-shuffle controls permute. The fitted models, their
    predictions and their scores toward the positive class are returned under ``_models``, ``_preds``
    and ``_scores`` (evaluation rows ``_eval_idx``) for the caller to save and break down.
    """
    y, groups, is_fit = np.asarray(y), np.asarray(groups), np.asarray(is_fit, dtype=bool)
    units = groups if units is None else np.asarray(units)
    fit_idx, eval_idx = np.flatnonzero(is_fit), np.flatnonzero(~is_fit)
    subgroups = None if subgroups is None else np.asarray(subgroups).astype(str)[eval_idx]
    if len(fit_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("holdout needs rows in both the fit and the evaluation partition")
    shared = set(groups[fit_idx]) & set(groups[eval_idx])
    if shared:
        raise ValueError(f"{len(shared)} group(s) appear in both the fit and the evaluation partition "
                         f"(e.g. {sorted(shared)[0]}); the partitions must split whole families.")
    classes = sorted(np.unique(y).tolist())
    if len(np.unique(y[eval_idx])) < 2:
        raise ValueError("the evaluation partition contains a single class; metrics are undefined")
    positive = cfg.class_names[0]
    # AUROC, oriented scores and raw-unit probe weights need a binary target holding the positive class
    binary = len(classes) == 2 and positive in classes
    preds, scores, models, extras = {}, {}, {}, {}
    for name, (X, factory) in _candidates(layer_matrices, texts, cfg, metadata).items():
        model, C, n_conv = _fit_counting_warnings(factory, X[fit_idx], y[fit_idx], groups[fit_idx], cfg)
        preds[name] = model.predict(X[eval_idx])
        if binary:
            scores[name] = positive_scores(model, X[eval_idx], positive)
        models[name] = model
        extras[name] = {"chosen_C": C, "n_convergence_warnings": n_conv}
    scores = scores if binary else None
    out = _evaluate(preds, extras, y[eval_idx], groups[eval_idx], classes, cfg, None, subgroups, scores)
    out.update(protocol="holdout", n_rows_fit=int(len(fit_idx)),
               class_counts_fit={str(c): int((y[fit_idx] == c).sum()) for c in classes},
               _models=models, _preds=preds, _scores=scores, _eval_idx=eval_idx)
    if cfg.n_permutations > 0:
        out["label_shuffle_control"] = {name: _permutation_holdout(
            layer_matrices[name], y, groups, units, fit_idx, eval_idx, cfg, preds[name],
            scores[name] if binary else None)
            for name in _permuted_layers(out, layer_matrices, cfg)}
    return out


def _shuffle_labels(y, units, rng) -> tuple[np.ndarray, str]:
    """Labels permuted across ``units`` (questions), preserving the class counts, and the level used.

    Rows of one unit move together only if they share one label; otherwise each row is permuted on its
    own, so that a real difference between a question's episodes is not overwritten by one shared label.
    """
    y, units = np.asarray(y), np.asarray(units)
    if pd.Series(y).groupby(units).nunique().max() > 1:
        return rng.permutation(y), "episode"
    uniq, inverse = np.unique(units, return_inverse=True)
    labels = np.empty(len(uniq), dtype=y.dtype)
    labels[inverse] = y
    return rng.permutation(labels)[inverse], "question"


def _summarise_null(observed, null, observed_auroc=None, null_auroc=None, unit=None) -> dict:
    """Null distribution of a label-shuffle control. ``selectivity`` (Hewitt & Liang) is the observed
    metric minus the mean under shuffled labels, with the same tuning and held-out scoring."""
    null = np.array(null)
    out = {"observed": float(observed), "null_mean": float(null.mean()),
           "null_p95": float(np.quantile(null, 0.95)), "n_permutations": int(len(null)),
           "p_value": float((1 + (null >= observed).sum()) / (1 + len(null))),
           "selectivity": float(observed - null.mean())}
    if observed_auroc is not None:
        na = np.array(null_auroc)
        out["auroc"] = {"observed": float(observed_auroc), "null_mean": float(np.nanmean(na)),
                        "null_p95": float(np.nanquantile(na, 0.95)),
                        "p_value": float((1 + (na >= observed_auroc).sum()) / (1 + len(na))),
                        "selectivity": float(observed_auroc - np.nanmean(na))}
    if unit is not None:
        out["shuffle_unit"] = unit
    return out


def _permutation_cv(X, y, groups, units, folds, cfg, observed_pred) -> dict:
    rng = np.random.default_rng(cfg.seed + 1)
    model = lambda C: _activation_model(C, cfg.seed)
    null, unit = [], None
    for _ in range(cfg.n_permutations):
        y_perm, unit = _shuffle_labels(y, units, rng)
        pred, _, _ = oof_predict(model, X, y_perm, groups, folds, cfg.c_grid, cfg.inner_folds, cfg.seed)
        null.append(balanced_accuracy(y_perm, pred))
    return _summarise_null(balanced_accuracy(y, observed_pred), null, unit=unit)


def _permutation_holdout(X, y, groups, units, fit_idx, eval_idx, cfg, observed_pred, observed_scores=None) -> dict:
    """Shuffle the TRAINING labels per question, refit the whole tuned procedure (family-grouped inner
    CV, same C grid), and score on the real evaluation labels only."""
    rng = np.random.default_rng(cfg.seed + 1)
    model = lambda C: _activation_model(C, cfg.seed)
    positive = cfg.class_names[0]
    y01_eval = (y[eval_idx] == positive).astype(int)
    null, null_auroc, unit = [], [], None
    for _ in range(cfg.n_permutations):
        y_fit, unit = _shuffle_labels(y[fit_idx], units[fit_idx], rng)
        m, _ = _fit_with_inner_cv(model, X[fit_idx], y_fit, groups[fit_idx], cfg.c_grid, cfg.inner_folds, cfg.seed)
        null.append(balanced_accuracy(y[eval_idx], m.predict(X[eval_idx])))
        if observed_scores is not None:
            null_auroc.append(_auroc(y01_eval, positive_scores(m, X[eval_idx], positive)))
    observed_auroc = None if observed_scores is None else _auroc(y01_eval, observed_scores)
    return _summarise_null(balanced_accuracy(y[eval_idx], observed_pred), null,
                           observed_auroc, null_auroc if observed_scores is not None else None, unit)


# ------------------------------------------------------------------------------- I/O

def excluded_episodes(cfg: LayerSelectionConfig) -> dict[str, str]:
    """``excluded_episodes`` merged with ``excluded_episodes_file``; conflicting reasons are refused."""
    out = {str(k): str(v) for k, v in cfg.excluded_episodes.items()}
    if cfg.excluded_episodes_file:
        loaded = json.loads((Path(cfg.capstone_data_root) / cfg.excluded_episodes_file).read_text())
        for qid, reason in loaded.items():
            if out.get(str(qid), str(reason)) != str(reason):
                raise ValueError(f"{qid} is excluded twice with different reasons: "
                                 f"{out[str(qid)]!r} and {reason!r}")
            out[str(qid)] = str(reason)
    return out


def _distinct(labels: pd.DataFrame, column: str) -> list[str]:
    return sorted(labels[column].dropna().astype(str).unique().tolist()) if column in labels else []


def load_inputs(cfg: LayerSelectionConfig, cols: bd.Columns | None = None):
    """Read, align and filter everything except the activation tensors."""
    cols = cols or bd.Columns(target=cfg.target_column, eligible=cfg.eligible_column, label=cfg.label_column)
    if "REPLACE_ME" in str(cfg.manifest_csv) or "REPLACE_ME" in str(cfg.labels_csv):
        raise SystemExit("Set labels_csv AND manifest_csv in the config to the NEW behavior package folder "
                         "(they are written together; the manifest has a `partition` column).")
    root = Path(cfg.capstone_data_root)
    episodes = bd.build_episode_table(root / cfg.results_dir, cfg.source_splits, cfg.text_fields)
    labels = pd.read_csv(root / cfg.labels_csv)
    manifest = pd.read_csv(root / cfg.manifest_csv)
    aligned, align_diag = bd.align_episodes(episodes, labels, manifest, cols, cfg.index_column,
                                            excluded=excluded_episodes(cfg))
    parts = [cfg.fit_partition, cfg.eval_partition] if cfg.protocol == "holdout" else DEV_PARTITIONS
    selected, sel_diag = bd.select_partitions(aligned, cols, parts, class_names=cfg.class_names)
    _, group_name = bd.episode_groups(selected, cfg.group_column)
    return selected, {**align_diag, **sel_diag, "protocol": cfg.protocol,
                      "classes": {"target_1": cfg.class_names[0], "target_0": cfg.class_names[1]},
                      "binary_contract_label_column": cfg.label_column,
                      "group_column": group_name,
                      "label_rule_versions": _distinct(labels, "rule_version"),
                      "label_sources": _distinct(labels, "label_source"),
                      "label_prompt_versions": _distinct(labels, "prompt_version")}, cols


def provenance(cfg: LayerSelectionConfig) -> dict:
    """Exact inputs and environment of a run: file hashes, code state and package versions."""
    root = Path(cfg.capstone_data_root)
    try:
        uncommitted = get_git_diff_sha256()
    except RuntimeError:
        uncommitted = "unknown"
    return {
        "labels_sha256": sha256_file(root / cfg.labels_csv),
        "manifest_sha256": sha256_file(root / cfg.manifest_csv),
        "excluded_episodes_file_sha256": (sha256_file(root / cfg.excluded_episodes_file)
                                          if cfg.excluded_episodes_file else None),
        "git_commit": get_git_commit(),
        "uncommitted_changes_sha256": uncommitted,
        "command": get_run_command(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: get_package_version(p) for p in PROVENANCE_PACKAGES},
    }


def _meta(selected: pd.DataFrame, name: str) -> np.ndarray:
    column = f"meta_{name}"
    if column not in selected:
        raise ValueError(f"Unknown episode field {name!r}; available: "
                         f"{sorted(c[5:] for c in selected.columns if c.startswith('meta_'))}")
    return selected[column].fillna("unknown").astype(str).to_numpy()


def breakdowns(res: dict, y_eval: np.ndarray, fields: dict[str, np.ndarray], positive: str) -> pd.DataFrame:
    """Balanced accuracy and AUROC of every candidate within each value of each field (evaluation rows).
    Both are NaN in a subgroup missing one of the classes: balanced accuracy there would be the recall
    of the only class present, which can look perfect. The class counts are always given."""
    rows = []
    for name, pred in res["_preds"].items():
        for fname, values in fields.items():
            for v in np.unique(values):
                m = values == v
                y01 = (y_eval[m] == positive).astype(int)
                both = 0 < y01.sum() < len(y01)
                rows.append({"name": name, "field": fname, "value": v, "n": int(m.sum()),
                             "n_target_1": int(y01.sum()), "n_target_0": int((1 - y01).sum()),
                             "balanced_accuracy": balanced_accuracy(y_eval[m], pred[m]) if both else float("nan"),
                             "auroc": (_auroc(y01, res["_scores"][name][m]) if res["_scores"] is not None
                                       else float("nan"))})
    return pd.DataFrame(rows)


def save_probes(models: dict, layer_names, positive: str, out_dir: Path) -> dict[str, str]:
    """Each layer's fitted probe as raw-unit weights toward ``positive`` (``X @ w + b`` = its score)."""
    out_dir.mkdir()
    hashes = {}
    for name in layer_names:
        model = models[name]
        w, b = raw_unit_weights(model, positive)
        path = out_dir / f"{name}.npz"
        np.savez(path, w=w, b=np.float64(b), scaler_mean=model.named_steps["scale"].mean_,
                 scaler_scale=model.named_steps["scale"].scale_,
                 C=np.float64(model.named_steps["clf"].C), positive_class=np.array(positive))
        hashes[path.name] = sha256_file(path)
    return hashes


def run(cfg: LayerSelectionConfig, cols: bd.Columns | None = None) -> Path:
    selected, diag, cols = load_inputs(cfg, cols)
    y = selected["target_name"].to_numpy()
    response = selected[cols.response].astype(str).to_numpy() if cols.response in selected else None
    groups, group_name = bd.episode_groups(selected, cfg.group_column)
    questions = selected["ep_question_id"].astype(str).to_numpy()
    texts = (selected[[f"text_{f}" for f in cfg.text_fields]]
             .agg(" || ".join, axis=1).to_numpy() if cfg.text_fields else None)
    metadata = (np.column_stack([_meta(selected, f) for f in cfg.metadata_baseline_fields])
                if cfg.metadata_baseline_fields else None)
    fields_for_breakdown = {f: _meta(selected, f) for f in cfg.breakdown_fields}

    root = Path(cfg.capstone_data_root)
    matrices = {}
    for layer in cfg.layers:
        name = f"layer_{int(layer):02d}"
        matrices[name] = bd.load_layer_matrix(selected, root / cfg.activations_dir / name,
                                              cfg.activation_file)
        logger.info("loaded %s: %s", name, matrices[name].shape)

    if cfg.protocol == "holdout":
        is_fit = (selected[cols.partition].astype(str) == cfg.fit_partition).to_numpy()
        res = compare_layers_holdout(matrices, y, groups, is_fit, texts, cfg, response, questions, metadata)
    else:
        res = compare_layers(matrices, y, groups, texts, cfg, response, questions, metadata)

    run_dir = create_versioned_run_dir(cfg.output_dir)
    fold_id = res.pop("_fold_id", None)
    if "_models" in res:   # holdout: per-subgroup metrics and the fitted probes
        eval_idx = res.pop("_eval_idx")
        breakdowns(res, y[eval_idx], {f: v[eval_idx] for f, v in fields_for_breakdown.items()},
                   cfg.class_names[0]).to_csv(run_dir / "breakdowns.csv", index=False)
        if res["_scores"] is not None:   # raw-unit weights are defined for the binary target only
            res["probe_files_sha256"] = save_probes(res["_models"], list(matrices), cfg.class_names[0],
                                                    run_dir / "probes")
        for key in ("_models", "_preds", "_scores"):
            res.pop(key)
    rows = pd.DataFrame({"question_id": questions, group_name: groups,
                         "partition": selected[cols.partition].to_numpy(), "target": y,
                         **fields_for_breakdown})
    if response is not None:
        rows["solver_response"] = response
    if fold_id is not None:
        rows["fold"] = fold_id
    rows.to_csv(run_dir / "rows_and_folds.csv", index=False)   # the exact definition of what was used
    _write_table(res, run_dir / "layer_comparison.csv")
    (run_dir / "layer_selection_summary.json").write_text(json.dumps(
        {"data_diagnostics": diag, "config": json.loads(json.dumps(asdict(cfg), default=str)),
         "provenance": provenance(cfg), "comparison": res}, indent=2, default=str))
    logger.info("wrote %s", run_dir)
    return run_dir


def _write_table(res: dict, path: Path) -> None:
    rows = []
    for name, r in res["results"].items():
        row = {"name": name, "balanced_accuracy": r["balanced_accuracy"],
               "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1],
               "auroc": r.get("auroc"), "auroc_ci95_lo": r.get("auroc_ci95", [None, None])[0],
               "auroc_ci95_hi": r.get("auroc_ci95", [None, None])[1], "macro_f1": r["macro_f1"],
               "diff_vs_best": r["diff_vs_best"]["diff"],
               "diff_ci95_lo": r["diff_vs_best"]["ci95"][0], "diff_ci95_hi": r["diff_vs_best"]["ci95"][1],
               "within_noise_of_best": r["within_noise_of_best"],
               "median_chosen_C": float(np.median(r.get("chosen_C_per_fold", [r.get("chosen_C", np.nan)])))}
        if "per_fold_mean" in r:
            row["per_fold_mean"], row["per_fold_sd"] = r["per_fold_mean"], r["per_fold_sd"]
        for c, m in r["per_class"].items():
            row[f"recall_{c}"], row[f"precision_{c}"], row[f"n_{c}"] = m["recall"], m["precision"], m["support"]
        for g, m in r.get("recall_by_solver_response", {}).items():
            row[f"resp_recall_{g}"], row[f"resp_n_{g}"] = m["recall"], m["n"]
        rows.append(row)
    pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).to_csv(path, index=False)


def inspect_inputs(cfg: LayerSelectionConfig, cols: bd.Columns | None = None) -> None:
    """Print what the files actually contain, and whether alignment verifies."""
    cols = cols or bd.Columns(target=cfg.target_column, eligible=cfg.eligible_column)
    root = Path(cfg.capstone_data_root)
    if "REPLACE_ME" in str(cfg.manifest_csv) or "REPLACE_ME" in str(cfg.labels_csv):
        print("labels_csv / manifest_csv are still REPLACE_ME: point both at the NEW behavior package folder.")
        return
    print("== labels.csv ==")
    labels = pd.read_csv(root / cfg.labels_csv)
    print(labels.dtypes.to_string(), "\n", labels.head(3).to_string(), "\n")
    print(f"labels.csv sha256: {sha256_file(root / cfg.labels_csv)}")
    if "rule_version" in labels:
        print(f"label rule versions: {sorted(labels['rule_version'].dropna().astype(str).unique())}\n")
    print("== manifest ==")
    manifest = pd.read_csv(root / cfg.manifest_csv)
    print(manifest.dtypes.to_string(), "\n", manifest.head(3).to_string(), "\n")
    first = bd.read_jsonl(root / cfg.results_dir / cfg.source_splits[0] / "interactions.jsonl")[0]
    print("== interactions.jsonl: keys of first record ==\n", sorted(first), "\n")
    print("== activation tensors (every configured layer must exist) ==")
    missing = []
    for layer in cfg.layers:
        layer_dir = root / cfg.activations_dir / f"layer_{int(layer):02d}"
        for split in cfg.source_splits:
            path = layer_dir / cfg.activation_file.format(split=split)
            if not path.is_file():
                missing.append(str(path))
                continue
            print(f"{layer_dir.name}/{path.name}: {tuple(bd.load_tensor(path).shape)}")
    if missing:
        raise FileNotFoundError(f"Missing activation tensors: {missing}")
    print("\n== alignment check + eligible rows per partition and class ==")
    _, diag, _ = load_inputs(cfg, cols)
    print(json.dumps(diag, indent=2, default=str))
