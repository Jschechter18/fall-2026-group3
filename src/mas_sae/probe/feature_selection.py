"""SAE feature selection with linear probes (issue #89), and the offline causal-proxy table.

Uses the same aligned rows as layer selection (official full9 export, Behavior v0.1.2 labels,
eligible_primary rows; test and intervention are never read):

  1. Encode the real layer activations with each SAE (custom TopK/L1 and pretrained Gemma Scope), and
     optionally with the raw activations themselves (kind `activations`, an ActivationBasis): the
     control the SAEs are compared against, selected by the same rule, mean-ablated, with std-matched
     random controls and the probe's full direction saved for the causal replay.
  2. Fit a class-weighted logistic probe on TRAIN SAE features (C by inner grouped CV on train).
  3. Rank features on TRAIN by mean |SHAP| (exact for a linear model on standardized inputs:
     |coef| * mean |standardized value|). Ranking on train keeps feature choice independent of
     the validation rows the effects are measured on.
  4. Score the probe on VALIDATION (balanced accuracy, bootstrap CI): do SAE features carry the signal?
  5. Offline causal proxy on VALIDATION (see mas_sae.causal.offline): no ablation / random /
     targeted ablation of the top acceptance-promoting features, read by a raw-activation probe.

Entry point: ``scripts/probe/feature_selection.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from mas_sae.causal import offline
from mas_sae.experiments.artifacts import get_untracked_files, get_untracked_sha256, sha256_file
from mas_sae.probe import behavior_data as bd
from mas_sae.probe import layer_selection as ls
from mas_sae.probe.cv_eval import balanced_accuracy, bootstrap_ci, grouped_bootstrap_indices
from mas_sae.utils.versioning import create_versioned_run_dir

logger = logging.getLogger(__name__)

CODE_DIRS = ("src", "scripts", "configs")
PATH_KEYS = ("capstone_data_root", "activations_dir", "results_dir", "labels_csv", "manifest_csv")
# custom = our TopK/L1 checkpoint run; gemma_scope = pretrained JumpReLU; activations = the raw layer
# activations as an identity basis (ActivationBasis), the control the SAEs are compared against
SAE_KINDS = ("custom", "gemma_scope", "activations")
TARGETED_RULE = "top n_targeted by mean |SHAP| on the fit partition with coefficient toward target 1 > 0"
SIGNED_RULE = "top n_targeted by mean |SHAP| on the fit partition, either sign (ablation-only sensitivity)"


class _Shift(torch.nn.Module):
    def __init__(self, shift: torch.Tensor):
        super().__init__()
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.shift


class ActivationBasis(torch.nn.Module):
    """Raw activations as a pseudo-SAE: latent i is neuron i minus its mean on the fit rows.

    ``encoder(x) = x - mean`` and ``decoder(z) = z + mean``, so zeroing a latent sets that neuron to
    its mean (mean ablation, not zero ablation) and leaves every other neuron as it was. Latents are
    signed and dense, so they have no firing frequency or positive reference activation: random
    controls are matched on per-neuron std instead.
    """
    kind = "activations"

    def __init__(self, mean):
        super().__init__()
        mean = torch.as_tensor(np.asarray(mean, dtype=np.float32))
        self.encoder, self.decoder = _Shift(-mean), _Shift(mean)
        self.input_dim = self.latent_dim = int(mean.numel())


def is_activation_basis(sae) -> bool:
    return getattr(sae, "kind", None) == ActivationBasis.kind


def ensure_center(spec: dict, root: Path, X_fit: np.ndarray) -> np.ndarray:
    """The fit-row mean of an ``activations`` basis, written to ``spec['path']`` if it is missing.

    An existing file must equal the mean of these rows: a center from other rows or another layer
    would silently change what mean ablation means, so a mismatch raises."""
    path = root / spec["path"]
    mean = X_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    if path.exists():
        saved = np.load(path)
        if saved.shape != mean.shape or not np.allclose(saved, mean, rtol=0, atol=1e-5):
            raise ValueError(f"{path} is not the mean of these fit rows (max |difference| "
                             f"{float(np.max(np.abs(saved - mean))) if saved.shape == mean.shape else 'shape'}); "
                             "it was computed on other data. Remove it to recompute.")
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mean)
    return mean


@dataclass
class FeatureSelectionConfig:
    # Data paths have no defaults: they come from the config passed to scripts/probe/feature_selection.py.
    capstone_data_root: Path
    activations_dir: str
    results_dir: str
    labels_csv: str
    manifest_csv: str
    index_column: str = "full_index"
    # question_id -> reason; collected episodes deliberately left out of a label package
    excluded_episodes: dict = field(default_factory=dict)
    excluded_episodes_file: str | None = None
    target_column: str = "primary_target"
    eligible_column: str = "eligible_primary"
    # binary contract and grouping, as in LayerSelectionConfig
    target_labels: list | None = None
    label_column: str | None = None
    group_column: str | None = "split_group_id"
    layer: int = 33
    activation_file: str = "{split}_attempt2.pt"
    fit_partition: str = "train"
    eval_partition: str = "validation"
    # name -> {kind: custom | gemma_scope | activations,
    #          path: <run dir> | <Gemma Scope SAE folder> | <center .npy, written on the first run>}
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
            if spec.get("kind") not in SAE_KINDS or "path" not in spec:
                raise ValueError(f"SAE {name!r} needs kind: {'|'.join(SAE_KINDS)} and a path.")
        return cfg

    def layer_selection_config(self) -> ls.LayerSelectionConfig:
        return ls.LayerSelectionConfig(
            capstone_data_root=self.capstone_data_root, activations_dir=self.activations_dir,
            results_dir=self.results_dir, labels_csv=self.labels_csv, manifest_csv=self.manifest_csv,
            index_column=self.index_column, excluded_episodes=self.excluded_episodes,
            excluded_episodes_file=self.excluded_episodes_file,
            layers=[self.layer], activation_file=self.activation_file,
            target_column=self.target_column, eligible_column=self.eligible_column,
            target_labels=self.target_labels, label_column=self.label_column, group_column=self.group_column,
            protocol="holdout", fit_partition=self.fit_partition, eval_partition=self.eval_partition,
            text_fields=[], seed=self.seed)


def load_sae(spec: dict, root: Path, layer: int):
    """(sae, description) for a custom checkpoint run directory or a Gemma Scope folder."""
    path = root / spec["path"]
    if spec["kind"] == "activations":
        sae = ActivationBasis(np.load(path))
        return sae, {"kind": "activations", "path": str(spec["path"]), "latent_dim": sae.latent_dim,
                     "ablation": "mean (latent = neuron - fit-row mean)"}
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
    if spec["kind"] == "activations":
        return [path]
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


def reference_activations(cfg: FeatureSelectionConfig, summary: dict, batch_size: int = 512) -> dict:
    """Typical activation of every targeted and random-control feature of a feature-selection run.

    For each SAE of ``summary`` (which must be the run made with ``cfg``'s layer and SAE files), the
    fit-partition rows are encoded and each feature in the targeted set or any random draw gets
    ``ref`` = its mean activation where it fires, plus its firing frequency and the 95th percentile
    of its nonzero values. Causal-replay steering clamps a feature to ``strength * ref``, so ``ref``
    is the unit strengths are expressed in. A feature that never fires has no ``ref`` and is refused.

    ``latents`` has the same statistics for every latent of the SAE (``ref`` is None where it never
    fires), plus ``rms`` (root mean square activation, zeros included) and ``decoder_norm`` (norm of
    what one unit of the latent adds to the reconstruction): the replay draws random controls matched
    on firing frequency and decoded contribution (rms * decoder_norm) from them.
    """
    if summary["config"]["layer"] != cfg.layer:
        raise ValueError(f"Feature selection was run on layer {summary['config']['layer']}, not {cfg.layer}.")
    root = Path(cfg.capstone_data_root)
    selected, _, cols = ls.load_inputs(cfg.layer_selection_config())
    X = bd.load_layer_matrix(selected, root / cfg.activations_dir / f"layer_{cfg.layer:02d}", cfg.activation_file)
    X = X[selected[cols.partition].astype(str).to_numpy() == cfg.fit_partition]
    out = {}
    for name, s in summary["saes"].items():
        if name not in cfg.saes:
            raise ValueError(f"SAE {name!r} of the summary is not in the feature-selection config.")
        hashes = {str(p.relative_to(root)): sha256_file(p) for p in sae_files(cfg.saes[name], root)}
        if hashes != summary["provenance"]["sae_sha256"].get(name):
            raise ValueError(f"SAE {name!r} files differ from the ones feature selection used.")
        ids = sorted({int(f) for f in s["targeted_features"]} | {int(f) for d in s["random_control_draws"] for f in d}
                     | {int(f) for f in s.get("targeted_features_signed", [])})
        sae = load_sae(cfg.saes[name], root, cfg.layer)[0].eval()
        Z = np.concatenate([offline.encode(sae, X[i:i + batch_size])[:, ids] for i in range(0, len(X), batch_size)])
        feats = {}
        for j, f in enumerate(ids):
            if is_activation_basis(sae):   # signed, dense: no firing frequency or positive ref
                feats[str(f)] = {"mean": float(sae.decoder.shift[f]), "std": float(Z[:, j].std()),
                                 "p95_abs": float(np.quantile(np.abs(Z[:, j]), 0.95))}
                continue
            on = Z[:, j][Z[:, j] > 0]
            if not len(on):
                raise ValueError(f"{name} feature {f} never fires on {cfg.fit_partition}: it has no reference activation.")
            feats[str(f)] = {"ref": float(on.mean()), "firing_frequency": float(len(on) / len(Z)),
                             "p95": float(np.quantile(on, 0.95))}
        out[name] = {"sae_sha256": hashes, "features": feats, "latents": latent_stats(sae, X, batch_size)}
        logger.info("%s: reference activations for %d features on %d %s rows", name, len(ids), len(Z), cfg.fit_partition)
    return out


@torch.no_grad()
def latent_stats(sae, X: np.ndarray, batch_size: int = 512) -> dict:
    """Per-latent statistics of ``sae`` on the rows ``X``, as lists indexed by latent id (see ``reference_activations``).

    For an ``ActivationBasis`` the latents are centered neurons: ``firing_frequency`` and ``ref`` are
    None (undefined for signed, dense values), ``rms`` is the per-neuron std on these rows and
    ``decoder_norm`` is 1, so a contribution match is a match on std.
    """
    r = lambda v: float(f"{v:.6g}")
    if is_activation_basis(sae):
        s = s2 = 0
        for i in range(0, len(X), batch_size):
            Z = offline.encode(sae, X[i:i + batch_size]).astype(np.float64)
            s, s2 = s + Z.sum(0), s2 + (Z ** 2).sum(0)
        return {"n_rows": len(X), "basis": ActivationBasis.kind, "firing_frequency": None, "ref": None,
                "mean": [r(v) for v in sae.decoder.shift.numpy()],
                "rms": [r(v) for v in np.sqrt(s2 / len(X))],
                "std": [r(v) for v in np.sqrt(np.maximum(s2 / len(X) - (s / len(X)) ** 2, 0))],
                "decoder_norm": [1.0] * sae.latent_dim}
    n_fire = s = s2 = 0
    for i in range(0, len(X), batch_size):
        Z = offline.encode(sae, X[i:i + batch_size]).astype(np.float64)
        n_fire, s, s2 = n_fire + (Z > 0).sum(0), s + Z.sum(0), s2 + (Z ** 2).sum(0)
    # decoders are affine, so decode(e_i) - decode(0) is latent i's decoder column in activation units;
    # one batch of unit vectors at a time (a full identity is latent_dim^2 floats: 1 GiB at 16k latents)
    bias = sae.decoder(torch.zeros(1, sae.latent_dim))
    dec_norm = np.empty(sae.latent_dim)
    for j in range(0, sae.latent_dim, batch_size):
        k = min(batch_size, sae.latent_dim - j)
        unit = torch.zeros(k, sae.latent_dim)
        unit[torch.arange(k), torch.arange(j, j + k)] = 1.0
        dec_norm[j:j + k] = (sae.decoder(unit) - bias).norm(dim=1).numpy()
    return {"n_rows": len(X),
            "firing_frequency": [r(v) for v in n_fire / len(X)],
            "ref": [r(t / n) if n else None for t, n in zip(s, n_fire)],
            "rms": [r(v) for v in np.sqrt(s2 / len(X))],
            "decoder_norm": [r(v) for v in dec_norm]}


def rank_features(probe, Z_fit: np.ndarray, signed: bool = False) -> pd.DataFrame:
    """Per-feature coefficient (toward target 1, standardized units) and mean |SHAP| on ``Z_fit``.

    ``signed`` latents (an ActivationBasis) have no firing frequency: that column is NaN and the
    per-latent std on the fit rows is given instead."""
    scaler, clf = probe.named_steps["scale"], probe.named_steps["clf"]
    sign = 1.0 if clf.classes_[1] == 1 else -1.0           # binary coef_ points toward classes_[1]
    coef = sign * clf.coef_[0]
    std = (Z_fit - scaler.mean_) / scaler.scale_
    return pd.DataFrame({
        "feature": np.arange(Z_fit.shape[1]),
        "coef_toward_accept": coef,
        "mean_abs_shap": np.abs(coef) * np.abs(std).mean(axis=0),
        "train_firing_frequency": np.full(Z_fit.shape[1], np.nan) if signed else (Z_fit > 0).mean(axis=0),
        **({"train_std": Z_fit.std(axis=0)} if signed else {}),
    })


def probe_direction(model, live: np.ndarray, sae: ActivationBasis, X_fit, y_fit) -> dict:
    """The raw-activation probe as one direction in activation space, for the causal replay: ``w``,
    ``b`` with ``X @ w + b`` = the probe's score toward target 1 (zero weight on constant neurons),
    and the difference in class means on the fit rows."""
    w_live, b = ls.raw_unit_weights(model, 1)              # in centered-latent units
    center = sae.decoder.shift.numpy().astype(np.float64)
    w = np.zeros(sae.latent_dim)
    w[live] = w_live
    return {"w": w, "b": np.float64(b - w_live @ center[live]), "center": center,
            "diff_means": X_fit[y_fit == 1].mean(0, dtype=np.float64) - X_fit[y_fit == 0].mean(0, dtype=np.float64)}


def analyse_sae(name, sae, X_fit, y_fit, g_fit, X_eval, y_eval, g_eval, readout, cfg, q_eval=None) -> dict:
    """``g_fit``/``g_eval`` are the groups no fold or bootstrap splits (families); ``q_eval`` are the
    evaluation question ids for the paired statistics (one row each), clustered by ``g_eval`` when they differ."""
    raw = is_activation_basis(sae)
    Z_fit, Z_eval = offline.encode(sae, X_fit), offline.encode(sae, X_eval)
    freq = (Z_fit > 0).mean(axis=0)
    spread = Z_fit.std(axis=0)
    live = np.flatnonzero(spread > 0) if raw else np.flatnonzero(freq > 0)
    logger.info("%s: %d of %d features %s on train", name, len(live), Z_fit.shape[1], "vary" if raw else "fire")
    q_eval = g_eval if q_eval is None else np.asarray(q_eval)
    clusters = None if np.array_equal(q_eval, g_eval) else g_eval

    # ---- feature-selection probe (fit and rank on train, score on validation)
    model, C, n_conv = ls._fit_counting_warnings(
        lambda c: ls._activation_model(c, cfg.seed), Z_fit[:, live], y_fit, g_fit, cfg)
    pred = model.predict(Z_eval[:, live])
    boot = grouped_bootstrap_indices(g_eval, cfg.n_bootstrap, cfg.seed)
    ranking = rank_features(model, Z_fit[:, live], signed=raw)
    ranking["feature"] = live
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    promote = ranking[ranking.coef_toward_accept > 0]
    targeted = promote["feature"].head(cfg.n_targeted).astype(int).tolist()
    signed = ranking.head(cfg.n_targeted)
    targeted_signed = signed["feature"].astype(int).tolist()

    # ---- offline causal proxy on validation
    recon = offline.decode(sae, Z_eval)
    p_control = offline.p_accept(readout, recon)
    p_targeted = offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, targeted)))
    # raw neurons all "fire" (signed, dense), so their controls are matched on std alone
    draws = (offline.contribution_matched_controls(np.ones_like(spread), spread, targeted, cfg.n_random_draws, cfg.seed)
             if raw else offline.frequency_matched_controls(freq, targeted, cfg.n_random_draws, cfg.seed))
    p_random_draws = np.stack([offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, d)))
                               for d in draws])
    p_random = p_random_draws.mean(axis=0)
    stats = lambda p: offline.paired_effect(p, p_control, q_eval, cfg.n_bootstrap, cfg.n_permutations, cfg.seed,
                                            clusters)

    # Sensitivity: keep the SAE's reconstruction error (x + decode(z_ablated) - decode(z)), so the
    # control is the original activation and only the ablated features change.
    err = X_eval - recon
    p_orig = offline.p_accept(readout, X_eval)
    p_t_err = offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, targeted)) + err)
    p_r_err = np.stack([offline.p_accept(readout, offline.decode(sae, offline.ablate(Z_eval, d)) + err)
                        for d in draws]).mean(axis=0)
    sens = lambda p: offline.paired_effect(p, p_orig, q_eval, cfg.n_bootstrap, cfg.n_permutations, cfg.seed,
                                           clusters)

    fve = 1 - float(((X_eval - recon) ** 2).sum() / ((X_eval - X_eval.mean(0)) ** 2).sum())
    return {
        "ranking": ranking,
        "direction": probe_direction(model, live, sae, X_fit, y_fit) if raw else None,
        "per_row": {f"{name}__no_ablation": p_control, f"{name}__random_ablation": p_random,
                    f"{name}__targeted_ablation": p_targeted},
        "summary": {
            "live_features_on_train": int(len(live)), "latent_dim": int(Z_fit.shape[1]),
            "mean_active_features_eval": None if raw else float((Z_eval > 0).sum(1).mean()),
            "variance_explained_eval": fve,
            "probe": {"chosen_C": C, "n_convergence_warnings": n_conv,
                      "balanced_accuracy_eval": balanced_accuracy(y_eval, pred),
                      "ci95": list(bootstrap_ci(y_eval, pred, boot))},
            "selection_rules": {"targeted_features": TARGETED_RULE, "targeted_features_signed": SIGNED_RULE,
                                "random_control_draws": "matched on per-neuron std" if raw
                                else "matched on training firing frequency"},
            "targeted_features": targeted,
            "targeted_features_signed": targeted_signed,
            "targeted_signed_coefficient_signs": [int(np.sign(c)) for c in signed["coef_toward_accept"]],
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
    ls_cfg = cfg.layer_selection_config()
    selected, diag, cols = ls.load_inputs(ls_cfg)
    root = Path(cfg.capstone_data_root)
    X = bd.load_layer_matrix(selected, root / cfg.activations_dir / f"layer_{cfg.layer:02d}", cfg.activation_file)
    y = (selected["target_name"] == ls_cfg.class_names[0]).astype(int).to_numpy()
    groups, _ = bd.episode_groups(selected, cfg.group_column)
    questions = selected["ep_question_id"].astype(str).to_numpy()
    part = selected[cols.partition].astype(str).to_numpy()
    fit, ev = part == cfg.fit_partition, part == cfg.eval_partition
    logger.info("rows: %d fit / %d eval; layer %d activations %s", fit.sum(), ev.sum(), cfg.layer, X.shape)
    for spec in cfg.saes.values():                         # fit-row centers exist before they are hashed
        if spec["kind"] == "activations":
            ensure_center(spec, root, X[fit])
    prov = provenance(cfg, config_path)                    # before any run artifacts are written

    readout, readout_c = offline.fit_readout(X[fit], y[fit], groups[fit], cfg.c_grid, cfg.inner_folds, cfg.seed)
    p_orig = offline.p_accept(readout, X[ev])
    readout_pred = (p_orig >= 0.5).astype(int)

    run_dir = create_versioned_run_dir(cfg.output_dir)
    summaries, per_row = {}, {"question_id": questions[ev], "accepted_observed": y[ev], "readout_original": p_orig}
    for name, spec in cfg.saes.items():
        sae, info = load_sae(spec, root, cfg.layer)
        res = analyse_sae(name, sae, X[fit], y[fit], groups[fit], X[ev], y[ev], groups[ev], readout, cfg,
                          q_eval=questions[ev])
        res["ranking"].assign(rank=np.arange(1, len(res["ranking"]) + 1)).pipe(
            lambda r: pd.concat([r[r.coef_toward_accept > 0].head(cfg.n_top_report).assign(direction="promotes_accept"),
                                 r[r.coef_toward_accept < 0].head(cfg.n_top_report).assign(direction="promotes_not_accept")])
        ).to_csv(run_dir / f"top_features_{name}.csv", index=False)
        res["ranking"].to_csv(run_dir / f"ranking_{name}.csv", index=False)   # the full signed ranking
        summaries[name] = {"sae": info, **res["summary"]}
        if res["direction"] is not None:
            path = run_dir / f"probe_direction_layer{cfg.layer:02d}_{name.replace(' ', '_')}.npz"
            np.savez(path, layer=np.int64(cfg.layer), **res["direction"])
            summaries[name]["probe_direction"] = {"file": path.name, "sha256": sha256_file(path)}
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
    _plot(table, run_dir / "causal_proxy_effects.svg",
          title=f"Offline SAE ablation on real layer-{cfg.layer} activations ({cfg.eval_partition})",
          xlabel="Δ predicted acceptance vs. that SAE's no-ablation control (pp, 95% CI)")
    logger.info("wrote %s", run_dir)
    return run_dir


def _plot(table: pd.DataFrame, path: Path, title: str, xlabel: str,
          conditions=("Random ablation", "Targeted ablation")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table[table["Condition"].isin(list(conditions))].reset_index(drop=True)
    lo = [float(s.strip("[]").split(",")[0]) for s in rows["95% CI for Δ"]]
    hi = [float(s.strip("[]").split(",")[1]) for s in rows["95% CI for Δ"]]
    d = rows["Δ vs. SAE Control (pp)"].astype(float).to_numpy()
    labels = [f"{e} · {c.split()[0].lower() if c.endswith(' ablation') else c.lower()}"
              for e, c in zip(rows["Experiment"], rows["Condition"])]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.2))
    ypos = np.arange(len(rows))[::-1]
    ax.errorbar(d, ypos, xerr=[d - np.array(lo), np.array(hi) - d], fmt="o", capsize=3, color="#3266ad")
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)
    # matplotlib leaves trailing spaces in SVG path data; strip them so the file passes git diff --check
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
