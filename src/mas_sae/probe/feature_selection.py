"""SAE feature selection with linear probes (issue #89), and the offline causal-proxy table.

Uses the same aligned rows as layer selection (official full9 export, Behavior v0.1.2 labels,
eligible_primary rows; test and intervention are never read):

  1. Encode the real layer activations with each SAE (custom TopK/L1 and pretrained Gemma Scope).
  2. Fit a class-weighted logistic probe on TRAIN SAE features (C by inner grouped CV on train).
  3. Rank features on TRAIN by mean |SHAP| (exact for a linear model on standardized inputs:
     |coef| * mean |standardized value|). Ranking on train keeps feature choice independent of
     the validation rows the effects are measured on.
  4. Score the probe on VALIDATION (balanced accuracy, bootstrap CI): do SAE features carry the signal?
  5. Offline causal proxy on VALIDATION (see mas_sae.causal.offline): no ablation / random /
     targeted ablation of the top acceptance-promoting features, read by a raw-activation probe.

Entry point: ``scripts/feature_selection.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mas_sae.causal import offline
from mas_sae.experiments.artifacts import get_untracked_files, get_untracked_sha256, sha256_file
from mas_sae.probe import behavior_data as bd
from mas_sae.probe import layer_selection as ls
from mas_sae.probe.cv_eval import balanced_accuracy, bootstrap_ci, grouped_bootstrap_indices
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)

ACCEPT = "adopted_critic"
CODE_DIRS = ("src", "scripts", "configs")
PATH_KEYS = ("capstone_data_root", "activations_dir", "results_dir", "labels_csv", "manifest_csv")


@dataclass
class FeatureSelectionConfig:
    # Data paths have no defaults: they come from the config passed to scripts/feature_selection.py.
    capstone_data_root: Path
    activations_dir: str
    results_dir: str
    labels_csv: str
    manifest_csv: str
    index_column: str = "full_index"
    layer: int = 33
    activation_file: str = "{split}_attempt2.pt"
    fit_partition: str = "train"
    eval_partition: str = "validation"
    # name -> {kind: custom | gemma_scope, path: <run dir> | <Gemma Scope SAE folder>}
    saes: dict = field(default_factory=dict)
    n_targeted: int = 10                 # top acceptance-promoting features ablated together
    n_random_draws: int = 50             # frequency-matched random control sets
    n_top_report: int = 25               # features per direction written to the ranking CSV
    seed: int = 42
    c_grid: list = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0, 10.0])
    inner_folds: int = 3
    n_bootstrap: int = 1000
    n_permutations: int = 10000
    output_dir: Path = Path("results/feature_selection")

    @classmethod
    def from_yaml(cls, path: Path) -> "FeatureSelectionConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        missing = [k for k in PATH_KEYS if not raw.get(k)]
        if missing:
            raise ValueError(f"Config {path} must set {missing}.")
        for k in ("capstone_data_root", "output_dir"):
            if k in raw:
                raw[k] = Path(raw[k])
        cfg = cls(**raw)
        for name, spec in cfg.saes.items():
            if spec.get("kind") not in {"custom", "gemma_scope"} or "path" not in spec:
                raise ValueError(f"SAE {name!r} needs kind: custom|gemma_scope and a path.")
        return cfg

    def layer_selection_config(self) -> ls.LayerSelectionConfig:
        return ls.LayerSelectionConfig(
            capstone_data_root=self.capstone_data_root, activations_dir=self.activations_dir,
            results_dir=self.results_dir, labels_csv=self.labels_csv, manifest_csv=self.manifest_csv,
            index_column=self.index_column, layers=[self.layer], activation_file=self.activation_file,
            protocol="holdout", fit_partition=self.fit_partition, eval_partition=self.eval_partition,
            text_fields=[], seed=self.seed)


def load_sae(spec: dict, root: Path, layer: int):
    """(sae, description) for a custom checkpoint run directory or a Gemma Scope folder."""
    path = root / spec["path"]
    if spec["kind"] == "custom":
        from mas_sae.sae.callbacks.checkpointing import load_sae_checkpoint
        sae = load_sae_checkpoint(path / "checkpoints" / "best_checkpoint.pt")
        return sae, {"kind": "custom", "path": str(spec["path"]), "sparsity_mode": sae.sparsity_mode,
                     "top_k": sae.top_k, "latent_dim": sae.latent_dim}
    from mas_sae.sae.pretrained import load_gemma_scope_sae
    sae, config = load_gemma_scope_sae(path, expected_layer=layer)
    return sae, {"kind": "gemma_scope", "path": str(spec["path"]), **config}


def sae_files(spec: dict, root: Path) -> list[Path]:
    path = root / spec["path"]
    if spec["kind"] == "custom":
        return [path / "checkpoints" / "best_checkpoint.pt"]
    return [path / "config.json", path / "params.safetensors"]


def provenance(cfg: FeatureSelectionConfig, config_path: Path | None) -> dict:
    """Layer-selection provenance plus hashes of every input this run reads, and the code state.

    ``git diff`` cannot see new files, so untracked files under src/, scripts/ and configs/ are
    hashed and listed too: a run made from uncommitted code says so.
    """
    root = Path(cfg.capstone_data_root)
    layer_dir = root / cfg.activations_dir / f"layer_{cfg.layer:02d}"
    inputs = {str(p.relative_to(root)): sha256_file(p) for p in
              [layer_dir / cfg.activation_file.format(split=s) for s in (cfg.fit_partition, cfg.eval_partition)]
              + [root / cfg.results_dir / s / "interactions.jsonl" for s in (cfg.fit_partition, cfg.eval_partition)]}
    saes = {name: {str(p.relative_to(root)): sha256_file(p) for p in sae_files(spec, root)}
            for name, spec in cfg.saes.items()}
    untracked = get_untracked_files(*CODE_DIRS)
    if untracked:
        logger.warning("running with %d untracked code files; commit them for a reproducible run: %s",
                       len(untracked), untracked)
    return {
        **ls.provenance(cfg.layer_selection_config()),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": sha256_file(config_path) if config_path else None,
        "input_sha256": inputs,
        "sae_sha256": saes,
        "untracked_code_files": untracked,
        "untracked_code_sha256": get_untracked_sha256(*CODE_DIRS),
    }


def rank_features(probe, Z_fit: np.ndarray) -> pd.DataFrame:
    """Per-feature coefficient (toward ACCEPT, standardized units) and mean |SHAP| on ``Z_fit``."""
    scaler, clf = probe.named_steps["scale"], probe.named_steps["clf"]
    sign = 1.0 if clf.classes_[1] == 1 else -1.0           # binary coef_ points toward classes_[1]
    coef = sign * clf.coef_[0]
    std = (Z_fit - scaler.mean_) / scaler.scale_
    return pd.DataFrame({
        "feature": np.arange(Z_fit.shape[1]),
        "coef_toward_accept": coef,
        "mean_abs_shap": np.abs(coef) * np.abs(std).mean(axis=0),
        "train_firing_frequency": (Z_fit > 0).mean(axis=0),
    })


def analyse_sae(name, sae, X_fit, y_fit, g_fit, X_eval, y_eval, g_eval, readout, cfg) -> dict:
    Z_fit, Z_eval = offline.encode(sae, X_fit), offline.encode(sae, X_eval)
    freq = (Z_fit > 0).mean(axis=0)
    live = np.flatnonzero(freq > 0)
    logger.info("%s: %d of %d features fire on train", name, len(live), Z_fit.shape[1])

    # ---- feature-selection probe (fit and rank on train, score on validation)
    model, C, n_conv = ls._fit_counting_warnings(
        lambda c: ls._activation_model(c, cfg.seed), Z_fit[:, live], y_fit, g_fit, cfg)
    pred = model.predict(Z_eval[:, live])
    boot = grouped_bootstrap_indices(g_eval, cfg.n_bootstrap, cfg.seed)
    ranking = rank_features(model, Z_fit[:, live])
    ranking["feature"] = live
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    promote = ranking[ranking.coef_toward_accept > 0]
    targeted = promote["feature"].head(cfg.n_targeted).astype(int).tolist()

    # ---- offline causal proxy on validation
    recon = offline.decode(sae, Z_eval)
    p_control = offline.p_accept(readout, recon)
    p_targeted = offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, targeted)))
    draws = offline.frequency_matched_controls(freq, targeted, cfg.n_random_draws, cfg.seed)
    p_random_draws = np.stack([offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, d)))
                               for d in draws])
    p_random = p_random_draws.mean(axis=0)
    stats = lambda p: offline.paired_effect(p, p_control, g_eval, cfg.n_bootstrap, cfg.n_permutations, cfg.seed)

    # Sensitivity: keep the SAE's reconstruction error (x + decode(z_ablated) - decode(z)), so the
    # control is the original activation and only the ablated features change.
    err = X_eval - recon
    p_orig = offline.p_accept(readout, X_eval)
    p_t_err = offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, targeted)) + err)
    p_r_err = np.stack([offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, d)) + err)
                        for d in draws]).mean(axis=0)
    sens = lambda p: offline.paired_effect(p, p_orig, g_eval, cfg.n_bootstrap, cfg.n_permutations, cfg.seed)

    fve = 1 - float(((X_eval - recon) ** 2).sum() / ((X_eval - X_eval.mean(0)) ** 2).sum())
    return {
        "ranking": ranking,
        "per_row": {f"{name}__no_ablation": p_control, f"{name}__random_ablation": p_random,
                    f"{name}__targeted_ablation": p_targeted},
        "summary": {
            "live_features_on_train": int(len(live)), "latent_dim": int(Z_fit.shape[1]),
            "mean_active_features_eval": float((Z_eval > 0).sum(1).mean()),
            "variance_explained_eval": fve,
            "probe": {"chosen_C": C, "n_convergence_warnings": n_conv,
                      "balanced_accuracy_eval": balanced_accuracy(y_eval, pred),
                      "ci95": list(bootstrap_ci(y_eval, pred, boot))},
            "targeted_features": targeted,
            "random_control_draws": draws,
            "conditions": {
                "no_ablation": {"acceptance_pct": 100 * float(p_control.mean())},
                "random_ablation": {"acceptance_pct": 100 * float(p_random.mean()), **stats(p_random),
                                    "acceptance_pct_across_draws_5_95": [
                                        100 * float(np.quantile(p_random_draws.mean(1), q)) for q in (0.05, 0.95)]},
                "targeted_ablation": {"acceptance_pct": 100 * float(p_targeted.mean()), **stats(p_targeted)},
            },
            "sensitivity_error_preserving": {
                "random_ablation": stats_dict(p_r_err, sens), "targeted_ablation": stats_dict(p_t_err, sens)},
        },
    }


def stats_dict(p, fn) -> dict:
    return {"acceptance_pct": 100 * float(p.mean()), **fn(p)}


def results_table(observed_pct: float, readout_pct: float, sae_summaries: dict) -> pd.DataFrame:
    """The presentation table: one block per SAE, Δ against that SAE's no-ablation row."""
    rows = [{"Experiment": "Baseline", "Condition": "No SAE (observed Solver behavior)",
             "Outcome": "observed", "Acceptance Rate (%)": observed_pct,
             "Δ vs. SAE Control (pp)": None, "95% CI for Δ": None, "p-value": None},
            {"Experiment": "Baseline", "Condition": "No SAE (readout on original activations)",
             "Outcome": "probe-predicted", "Acceptance Rate (%)": readout_pct,
             "Δ vs. SAE Control (pp)": None, "95% CI for Δ": None, "p-value": None}]
    for name, s in sae_summaries.items():
        for key, label in (("no_ablation", "No ablation"), ("random_ablation", "Random ablation"),
                           ("targeted_ablation", "Targeted ablation")):
            c = s["conditions"][key]
            rows.append({"Experiment": name, "Condition": label, "Outcome": "probe-predicted",
                         "Acceptance Rate (%)": c["acceptance_pct"],
                         "Δ vs. SAE Control (pp)": c.get("delta_pp", "Reference"),
                         "95% CI for Δ": (f"[{c['ci95_pp'][0]:.2f}, {c['ci95_pp'][1]:.2f}]"
                                          if "ci95_pp" in c else None),
                         "p-value": c.get("p_value")})
    return pd.DataFrame(rows)


def run(cfg: FeatureSelectionConfig, config_path: Path | None = None) -> Path:
    if not cfg.saes:
        raise ValueError("Configure at least one SAE under `saes`.")
    prov = provenance(cfg, config_path)                    # before any run artifacts are written
    selected, diag, cols = ls.load_inputs(cfg.layer_selection_config())
    root = Path(cfg.capstone_data_root)
    X = bd.load_layer_matrix(selected, root / cfg.activations_dir / f"layer_{cfg.layer:02d}", cfg.activation_file)
    y = (selected["target_name"] == ACCEPT).astype(int).to_numpy()
    groups = selected["ep_question_id"].astype(str).to_numpy()
    part = selected[cols.partition].astype(str).to_numpy()
    fit, ev = part == cfg.fit_partition, part == cfg.eval_partition
    logger.info("rows: %d fit / %d eval; layer %d activations %s", fit.sum(), ev.sum(), cfg.layer, X.shape)

    readout, readout_c = offline.fit_readout(X[fit], y[fit], groups[fit], cfg.c_grid, cfg.inner_folds, cfg.seed)
    p_orig = offline.p_accept(readout, X[ev])
    readout_pred = (p_orig >= 0.5).astype(int)

    run_dir = create_versioned_run_dir(cfg.output_dir)
    summaries, per_row = {}, {"question_id": groups[ev], "accepted_observed": y[ev], "readout_original": p_orig}
    for name, spec in cfg.saes.items():
        sae, info = load_sae(spec, root, cfg.layer)
        res = analyse_sae(name, sae, X[fit], y[fit], groups[fit], X[ev], y[ev], groups[ev], readout, cfg)
        res["ranking"].assign(rank=np.arange(1, len(res["ranking"]) + 1)).pipe(
            lambda r: pd.concat([r[r.coef_toward_accept > 0].head(cfg.n_top_report).assign(direction="promotes_accept"),
                                 r[r.coef_toward_accept < 0].head(cfg.n_top_report).assign(direction="promotes_not_accept")])
        ).to_csv(run_dir / f"top_features_{name}.csv", index=False)
        summaries[name] = {"sae": info, **res["summary"]}
        per_row.update(res["per_row"])
        logger.info("%s done: probe BA %.3f", name, res["summary"]["probe"]["balanced_accuracy_eval"])

    table = results_table(100 * float(y[ev].mean()), 100 * float(p_orig.mean()), summaries)
    table.to_csv(run_dir / "causal_proxy_table.csv", index=False)
    pd.DataFrame(per_row).to_csv(run_dir / "per_row_outcomes.csv", index=False)
    (run_dir / "feature_selection_summary.json").write_text(json.dumps({
        "data_diagnostics": diag,
        "config": json.loads(json.dumps(asdict(cfg), default=str)),
        "provenance": prov,
        "readout": {"chosen_C": readout_c, "n_fit": int(fit.sum()), "n_eval": int(ev.sum()),
                    "mean_p_accept_eval": float(p_orig.mean()),
                    "observed_accept_rate_eval": float(y[ev].mean()),
                    "balanced_accuracy_eval_at_0.5": balanced_accuracy(y[ev], readout_pred)},
        "saes": summaries,
        "outcome_note": ("Acceptance rates for SAE conditions are a raw-activation probe's mean predicted "
                         "P(accept) on validation activations, not regenerated Solver answers."),
    }, indent=2, default=str))
    _plot(table, run_dir / "causal_proxy_effects.svg")
    logger.info("wrote %s", run_dir)
    return run_dir


def _plot(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table[table["Condition"].isin(["Random ablation", "Targeted ablation"])].reset_index(drop=True)
    lo = [float(s.strip("[]").split(",")[0]) for s in rows["95% CI for Δ"]]
    hi = [float(s.strip("[]").split(",")[1]) for s in rows["95% CI for Δ"]]
    d = rows["Δ vs. SAE Control (pp)"].astype(float).to_numpy()
    labels = [f"{e} · {c.split()[0].lower()}" for e, c in zip(rows["Experiment"], rows["Condition"])]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.2))
    ypos = np.arange(len(rows))[::-1]
    ax.errorbar(d, ypos, xerr=[d - np.array(lo), np.array(hi) - d], fmt="o", capsize=3, color="#3266ad")
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("Δ predicted acceptance vs. that SAE's no-ablation control (pp, 95% CI)")
    ax.set_title("Offline SAE ablation on real layer-33 activations (validation)")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
    # matplotlib leaves trailing spaces in SVG path data; strip them so the file passes git diff --check
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
