"""Causal replay (issue #90): regenerate the Solver's A2 with an SAE spliced into the residual stream.

For each eligible episode of one frozen partition, the stored A1 and Critic feedback are replayed
through the Solver's revise prompt with the same model, revision, prompt and greedy settings as
collection. During the A2 prefill, the layer's output at the final prompt token (exactly where the
collected activations, probes and SAE features were measured) is replaced by the SAE
reconstruction, with or without chosen features zeroed. Later decode steps are not modified.
The regenerated A2 is labeled with the same Behavior v0.1.2 rules as the stored data, and
acceptance means ``solver_response == "adopted_critic"``.

Conditions per episode:
  Baseline   no SAE: plain replay; also compared with the stored A2 as a faithfulness check
  per SAE    no ablation (reconstruction only) / targeted ablation / random ablation (draws)

Targeted and random feature sets come from the #89 feature-selection summary, so they are the
features reported there; each SAE's checkpoint hashes must match that run's.

Generations are appended to ``per_row.jsonl`` as they finish, so an interrupted run resumes
where it stopped and ``analyse`` can be run on a partial run at any time.
Entry point: ``scripts/causal_replay.py``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from mas_sae.causal import offline
from mas_sae.evaluation import behavior_labels as bl
from mas_sae.experiments.artifacts import (
    get_git_commit, get_git_diff_sha256, get_package_version, get_run_command,
    get_untracked_files, get_untracked_sha256, sha256_file,
)

logger = logging.getLogger(__name__)

POLICY = bl.V012_QC
ACCEPT = "adopted_critic"
BASELINE = "Baseline"
CODE_DIRS = ("src", "scripts", "configs")
PATH_KEYS = ("capstone_data_root", "interactions_jsonl", "labels_csv", "feature_selection_summary")
PACKAGES = ("torch", "transformers", "datasets", "numpy", "pandas")
CONDITION_LABELS = {"no_sae": "No SAE (replayed A2)", "no_ablation": "No ablation",
                    "random_ablation": "Random ablation", "targeted_ablation": "Targeted ablation"}
TABLE_COLUMNS = ["Experiment", "Condition", "Outcome", "N", "Acceptance Rate (%)",
                 "Δ vs. SAE Control (pp)", "95% CI for Δ", "p-value"]


@dataclass
class ReplayConfig:
    # Data paths have no defaults: they come from the config passed to scripts/causal_replay.py.
    capstone_data_root: Path
    interactions_jsonl: str
    labels_csv: str
    feature_selection_summary: str
    solver: dict = field(default_factory=dict)       # same role spec as the collection config
    musique_revision: str | None = None
    partition: str = "intervention"
    layer: int = 33
    # name -> {kind: custom | gemma_scope, path: ...}; must be SAEs of the feature-selection run
    saes: dict = field(default_factory=dict)
    n_random_draws: int = 5
    limit: int | None = None                         # first N episodes of the seeded order
    seed: int = 42
    n_bootstrap: int = 1000
    n_permutations: int = 10000
    output_dir: Path = Path("results/causal_replay")

    @classmethod
    def from_yaml(cls, path: Path) -> "ReplayConfig":
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
            if name == BASELINE or spec.get("kind") not in {"custom", "gemma_scope"} or "path" not in spec:
                raise ValueError(f"SAE {name!r} needs kind: custom|gemma_scope and a path.")
        if cfg.limit is not None and (type(cfg.limit) is not int or cfg.limit <= 0):
            raise ValueError("limit must be a positive integer or null.")
        if type(cfg.n_random_draws) is not int or cfg.n_random_draws <= 0:
            raise ValueError("n_random_draws must be a positive integer.")
        return cfg


# ---------------------------------------------------------------------------------------------
# The intervention


class SAESplice:
    """Replace a module's output at the final prompt token of the prefill with an SAE reconstruction.

    ``generate`` calls the module once for the prompt (prefill) and once per new token. Only the
    prefill's last position is changed, which is where the collected activations were taken.
    Decoder layers return a tensor or a tuple whose first item is the hidden state; both work.
    """

    def __init__(self, model: torch.nn.Module, module_name: str, sae, ablate=()):
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise ValueError(f"Module {module_name!r} was not found in the model.")
        self.module, self.sae, self.ablate = modules[module_name], sae, sorted(int(f) for f in ablate)
        self.calls, self.stats, self._handle = 0, {}, None

    def _hook(self, module, inputs, output):
        self.calls += 1
        if self.calls > 1:
            return None
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 2:
            raise ValueError(f"Expected a single-prompt prefill (1, seq>1, d), got {tuple(hidden.shape)}.")
        x = hidden[:, -1, :].float()
        with torch.no_grad():
            z = self.sae.encoder(x)
            recon_full = self.sae.decoder(z)
            self.stats = {"n_active": int((z > 0).sum()),
                          "recon_rel_error": float((x - recon_full).norm() / x.norm().clamp_min(1e-12))}
            recon = recon_full
            if self.ablate:
                # how many of the ablated features were actually firing here: 0 means a no-op ablation
                self.stats["n_ablated_active"] = int((z[0, self.ablate] > 0).sum())
                z = z.clone()
                z[:, self.ablate] = 0.0
                recon = self.sae.decoder(z)
        hidden = hidden.clone()
        hidden[:, -1, :] = recon.to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, (tuple, list)) else hidden

    def __enter__(self) -> "SAESplice":
        self.calls, self.stats = 0, {}
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc[0] is None and self.calls == 0:
            raise RuntimeError("The SAE splice hook never ran: the module is not on the generation path.")


# ---------------------------------------------------------------------------------------------
# Inputs


def load_episodes(cfg: ReplayConfig) -> pd.DataFrame:
    """Eligible episodes of ``cfg.partition`` in a seeded order, with their stored labels.

    Every eligible label must have exactly one stored Natural episode, and relabeling the stored
    A2 with the v0.1.2 rules must reproduce labels.csv, so the regenerated A2s are labeled by
    exactly the rules that produced the stored labels.
    """
    from mas_sae.probe.behavior_data import read_jsonl

    root = Path(cfg.capstone_data_root)
    labels = bl.read_labels(root / cfg.labels_csv)
    versions = {lab["rule_version"] for lab in labels.values()}
    if versions != {POLICY.version}:
        raise ValueError(f"labels.csv rule_version {sorted(versions)} is not {POLICY.version!r}.")
    eligible = {q: lab for q, lab in labels.items()
                if lab["partition"] == cfg.partition and lab["eligible_primary"]}
    if not eligible:
        raise ValueError(f"No eligible_primary labels in partition {cfg.partition!r}.")

    records = read_jsonl(root / cfg.interactions_jsonl)
    seen: dict[str, dict] = {}
    for r in records:
        qid = str(r["question_id"])
        if r.get("critic_condition") != "natural":
            raise ValueError(f"{qid}: expected Natural-condition episodes, got {r.get('critic_condition')!r}.")
        if qid in seen:
            raise ValueError(f"{qid}: more than one episode in {cfg.interactions_jsonl}.")
        if r.get("partition") not in (None, cfg.partition):
            raise ValueError(f"{qid}: episode partition {r.get('partition')!r} is not {cfg.partition!r}.")
        seen[qid] = r
    missing = sorted(set(eligible) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} eligible questions have no stored episode, e.g. {missing[:3]}.")

    rows = []
    for qid, lab in eligible.items():
        r = seen[qid]
        relabel = bl.classify_candidate(r, POLICY)
        if (relabel["solver_response"], relabel["primary_target"]) != (lab["solver_response"], lab["primary_target"]):
            raise ValueError(f"{qid}: relabeling the stored A2 gives {relabel['solver_response']!r}, "
                             f"labels.csv says {lab['solver_response']!r}.")
        rows.append({"question_id": qid, "source_split": r["source_split"], "question": r["question"],
                     "solver_attempt_1": r["solver_attempt_1"], "critic_feedback": r["critic_feedback"],
                     "critic_advocated_answer": r["critic_advocated_answer"],
                     "critic_noncommittal": r["critic_noncommittal"],
                     "stored_solver_attempt_2": r["solver_attempt_2"],
                     "stored_solver_response": lab["solver_response"],
                     "stored_adopted": int(lab["primary_target"])})
    df = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    df = df.iloc[np.random.default_rng(cfg.seed).permutation(len(df))].reset_index(drop=True)
    return df.head(cfg.limit) if cfg.limit else df


def load_paragraphs(episodes: pd.DataFrame, revision: str | None) -> dict[str, list]:
    """MuSiQue paragraphs for each episode, from its source split at the pinned revision."""
    from mas_sae.data.musique import load_musique_examples_by_id

    out = {}
    for split, ids in episodes.groupby("source_split")["question_id"]:
        for ex in load_musique_examples_by_id(split, list(ids), revision=revision):
            out[str(ex["id"])] = ex["paragraphs"]
    return out


def feature_sets(cfg: ReplayConfig, sae_hashes: dict) -> dict:
    """Targeted and random feature sets per SAE, from the #89 summary, after checking it used the same SAEs."""
    summary = json.loads((Path(cfg.capstone_data_root) / cfg.feature_selection_summary).read_text())
    if summary["config"]["layer"] != cfg.layer:
        raise ValueError(f"Feature selection was run on layer {summary['config']['layer']}, not {cfg.layer}.")
    out = {}
    for name in cfg.saes:
        if name not in summary["saes"]:
            raise ValueError(f"SAE {name!r} is not in {cfg.feature_selection_summary}.")
        expected = summary["provenance"].get("sae_sha256", {}).get(name)
        if expected != sae_hashes[name]:
            raise ValueError(f"SAE {name!r} files differ from the ones feature selection used "
                             f"({sae_hashes[name]} vs {expected}).")
        s = summary["saes"][name]
        draws = s["random_control_draws"]
        if len(draws) < cfg.n_random_draws:
            raise ValueError(f"{name}: only {len(draws)} random draws available.")
        out[name] = {"targeted": [int(f) for f in s["targeted_features"]],
                     "random": [[int(f) for f in d] for d in draws[:cfg.n_random_draws]]}
    return out


def conditions(features: dict) -> list[dict]:
    """Every (experiment, condition, draw) generated per episode, with the features it ablates."""
    out = [{"experiment": BASELINE, "condition": "no_sae", "draw": None, "ablate": None}]
    for name, f in features.items():
        out.append({"experiment": name, "condition": "no_ablation", "draw": None, "ablate": []})
        out.append({"experiment": name, "condition": "targeted_ablation", "draw": None, "ablate": f["targeted"]})
        out += [{"experiment": name, "condition": "random_ablation", "draw": i, "ablate": d}
                for i, d in enumerate(f["random"])]
    return out


def key(c: dict) -> tuple:
    return (c["question_id"], c["experiment"], c["condition"], c["draw"])


# ---------------------------------------------------------------------------------------------
# Running


@dataclass
class Components:
    """What a replay needs besides the episodes: a configured Solver, its model, the hooked module and SAEs."""
    solver: Any
    model: Any
    site: str
    saes: dict
    features: dict
    paragraphs: dict
    info: dict = field(default_factory=dict)


def load_components(cfg: ReplayConfig, episodes: pd.DataFrame) -> Components:
    from mas_sae.activations.sites import resolve_solver_sites
    from mas_sae.agents.solver import Solver
    from mas_sae.models.roles import configure_agent, load_spec, resolve_role_spec
    from mas_sae.probe.feature_selection import load_sae, sae_files

    root = Path(cfg.capstone_data_root)
    hashes = {name: {str(p.relative_to(root)): sha256_file(p) for p in sae_files(spec, root)}
              for name, spec in cfg.saes.items()}
    features = feature_sets(cfg, hashes)
    paragraphs = load_paragraphs(episodes, cfg.musique_revision)

    spec = resolve_role_spec("solver", cfg.solver, default_max_new_tokens=64)
    model, processor = load_spec(spec)
    solver = Solver(model, processor)
    configure_agent(solver, spec)
    site = resolve_solver_sites(model, spec["loader"], [cfg.layer])[0]
    saes = {}
    for name, sae_spec in cfg.saes.items():
        sae, _ = load_sae(sae_spec, root, cfg.layer)
        saes[name] = sae.to(model.device).eval()
    return Components(solver, model, site, saes, features, paragraphs,
                      {"solver_spec": spec, "sae_sha256": hashes,
                       "device": str(model.device),
                       "device_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None})


def generate(comp: Components, ep: dict, cond: dict) -> dict:
    """One regenerated A2 under ``cond``, labeled, with telemetry."""
    sae = comp.saes.get(cond["experiment"])
    splice = (SAESplice(comp.model, comp.site, sae, cond["ablate"]) if cond["ablate"] is not None
              else contextlib.nullcontext())
    t0 = time.perf_counter()
    with splice:
        a2 = comp.solver.revise(ep["question"], comp.paragraphs[ep["question_id"]],
                                ep["solver_attempt_1"], ep["critic_feedback"])
    seconds = time.perf_counter() - t0
    label = bl.classify_candidate({"solver_attempt_1": ep["solver_attempt_1"], "solver_attempt_2": a2,
                                   "critic_advocated_answer": ep["critic_advocated_answer"],
                                   "critic_noncommittal": ep["critic_noncommittal"]}, POLICY)
    gen = comp.solver.last_generation or {}
    prompt = comp.solver.last_prompt or {}
    stats = splice.stats if isinstance(splice, SAESplice) else {}
    return {"question_id": ep["question_id"], "experiment": cond["experiment"],
            "condition": cond["condition"], "draw": cond["draw"],
            "solver_attempt_2": a2, "stored_solver_attempt_2": ep["stored_solver_attempt_2"],
            "a2_matches_stored": a2 == ep["stored_solver_attempt_2"],
            "solver_response": label["solver_response"], "adopted": int(label["solver_response"] == ACCEPT),
            "stored_adopted": ep["stored_adopted"],
            "generated_tokens": gen.get("generated_tokens"), "reached_token_budget": gen.get("reached_token_budget"),
            "prompt_tokens": prompt.get("prompt_tokens"), "input_ids_sha256": prompt.get("input_ids_sha256"),
            "n_active_features": stats.get("n_active"), "n_ablated_active": stats.get("n_ablated_active"),
            "recon_rel_error": stats.get("recon_rel_error"), "seconds": round(seconds, 4)}


def provenance(cfg: ReplayConfig, config_path: Path | None, comp: Components) -> dict:
    root = Path(cfg.capstone_data_root)
    try:
        uncommitted = get_git_diff_sha256()
    except RuntimeError:
        uncommitted = "unknown"
    untracked = get_untracked_files(*CODE_DIRS)
    if untracked:
        logger.warning("running with %d untracked code files; commit them for a reproducible run: %s",
                       len(untracked), untracked)
    return {
        "git_commit": get_git_commit(), "uncommitted_changes_sha256": uncommitted,
        "untracked_code_files": untracked, "untracked_code_sha256": get_untracked_sha256(*CODE_DIRS),
        "command": get_run_command(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": sha256_file(config_path) if config_path else None,
        "input_sha256": {p: sha256_file(root / p) for p in
                         (cfg.interactions_jsonl, cfg.labels_csv, cfg.feature_selection_summary)},
        "label_policy": POLICY.version, "label_policy_sha256": POLICY.definition_sha256(),
        "packages": {p: get_package_version(p) for p in PACKAGES},
        **comp.info,
    }


def _config_dict(cfg: ReplayConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=str))


def replay(cfg: ReplayConfig, episodes: pd.DataFrame, comp: Components, run_dir: Path,
           config_path: Path | None = None) -> Path:
    """Generate every missing (episode, condition) into ``run_dir/per_row.jsonl``; resumable."""
    run_dir.mkdir(parents=True, exist_ok=True)
    conds = conditions(comp.features)
    meta_path = run_dir / "replay_config.json"
    meta = {"config": _config_dict(cfg), "conditions": conds, "features": comp.features}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        # what was generated must not change; limit may (to extend a smoke run into the full run)
        same = lambda m: ({k: v for k, v in m["config"].items() if k != "limit"}, m["conditions"], m["features"])
        if same(old) != same(json.loads(json.dumps(meta))):
            raise ValueError(f"{run_dir} was started with a different config or feature sets; use a new run.")
        logger.info("resuming %s", run_dir)
    else:
        meta_path.write_text(json.dumps({**meta, "provenance": provenance(cfg, config_path, comp)},
                                        indent=2, default=str))
    episodes.drop(columns=["question", "critic_feedback"]).to_csv(run_dir / "episodes.csv", index=False)

    out = run_dir / "per_row.jsonl"
    done = {key(r) for r in _read_rows(out)}
    todo = [(ep, c) for ep in episodes.to_dict("records") for c in conds
            if (ep["question_id"], c["experiment"], c["condition"], c["draw"]) not in done]
    logger.info("%d episodes x %d conditions; %d generations already done, %d to go",
                len(episodes), len(conds), len(done), len(todo))
    t0 = time.perf_counter()
    with out.open("a") as f:
        for i, (ep, c) in enumerate(todo, 1):
            f.write(json.dumps(generate(comp, ep, c)) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                rate = (time.perf_counter() - t0) / i
                logger.info("%d/%d generations, %.2fs each, ~%.1f min left",
                            i, len(todo), rate, rate * (len(todo) - i) / 60)
    return run_dir


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping a truncated line in %s (interrupted write)", path)
    return rows


# ---------------------------------------------------------------------------------------------
# Analysis


def analyse(run_dir: Path, n_bootstrap: int | None = None, n_permutations: int | None = None) -> pd.DataFrame:
    """Write causal_table.csv, per_row.csv, causal_replay_summary.json and a plot for ``run_dir``.

    Uses only episodes with every condition generated, so a partial run gives a valid (smaller) table.
    """
    meta = json.loads((run_dir / "replay_config.json").read_text())
    cfg = meta["config"]
    n_bootstrap = n_bootstrap or cfg["n_bootstrap"]
    n_permutations = n_permutations or cfg["n_permutations"]
    expected = {(c["experiment"], c["condition"], c["draw"]) for c in meta["conditions"]}
    rows = pd.DataFrame(_read_rows(run_dir / "per_row.jsonl"))
    if rows.empty:
        raise ValueError(f"No generations in {run_dir}.")
    rows = rows.drop_duplicates(["question_id", "experiment", "condition", "draw"], keep="last")
    have = rows.assign(k=list(zip(rows.experiment, rows.condition, rows.draw))).groupby("question_id")["k"].agg(set)
    complete = sorted(q for q, ks in have.items() if expected <= {(e, c, None if pd.isna(d) else int(d))
                                                                  for e, c, d in ks})
    rows = rows[rows.question_id.isin(complete)].sort_values(["question_id", "experiment", "condition", "draw"])
    rows.to_csv(run_dir / "per_row.csv", index=False)
    if not complete:
        raise ValueError(f"No episode in {run_dir} has every condition yet.")

    def outcome(exp, cond) -> np.ndarray:
        r = rows[(rows.experiment == exp) & (rows.condition == cond)]
        return r.groupby("question_id")["adopted"].mean().reindex(complete).to_numpy(float)

    n = len(complete)
    groups = np.array(complete)
    stored = rows[rows.experiment == BASELINE].set_index("question_id")["stored_adopted"].reindex(complete)
    table = [{"Experiment": BASELINE, "Condition": "No SAE (stored A2)", "Outcome": "observed", "N": n,
              "Acceptance Rate (%)": 100 * float(stored.mean())},
             {"Experiment": BASELINE, "Condition": CONDITION_LABELS["no_sae"], "Outcome": "regenerated", "N": n,
              "Acceptance Rate (%)": 100 * float(outcome(BASELINE, "no_sae").mean())}]
    effects = {}
    for exp in [e for e in dict.fromkeys(c["experiment"] for c in meta["conditions"]) if e != BASELINE]:
        ref = outcome(exp, "no_ablation")
        table.append({"Experiment": exp, "Condition": "No ablation", "Outcome": "regenerated", "N": n,
                      "Acceptance Rate (%)": 100 * float(ref.mean()), "Δ vs. SAE Control (pp)": "Reference"})
        for cond in ("random_ablation", "targeted_ablation"):
            y = outcome(exp, cond)
            e = offline.paired_effect(y, ref, groups, n_bootstrap, n_permutations, cfg["seed"])
            effects[f"{exp}__{cond}"] = e
            table.append({"Experiment": exp, "Condition": CONDITION_LABELS[cond], "Outcome": "regenerated",
                          "N": n, "Acceptance Rate (%)": 100 * float(y.mean()),
                          "Δ vs. SAE Control (pp)": e["delta_pp"],
                          "95% CI for Δ": f"[{e['ci95_pp'][0]:.2f}, {e['ci95_pp'][1]:.2f}]",
                          "p-value": e["p_value"]})
    table = pd.DataFrame(table).reindex(columns=TABLE_COLUMNS)
    table.to_csv(run_dir / "causal_table.csv", index=False)

    base = rows[rows.experiment == BASELINE]
    by_cond = rows.assign(c=rows.experiment + " · " + rows.condition)
    summary = {
        "n_episodes_complete": n,
        "n_episodes_planned": int(len(pd.read_csv(run_dir / "episodes.csv"))),
        "n_generations": int(len(rows)),
        "replay_check": {  # plain replay against the stored A2 (issue #116)
            "a2_exact_match_rate": float(base.a2_matches_stored.mean()),
            "solver_response_match_rate": float((base.solver_response.to_numpy()
                                                 == _stored_responses(run_dir, base)).mean()),
        },
        "effects": effects,
        "solver_response_counts": {c: g.solver_response.value_counts().to_dict() for c, g in by_cond.groupby("c")},
        "mean_ablated_features_active": {c: float(g.n_ablated_active.mean()) for c, g in by_cond.groupby("c")
                                         if g.n_ablated_active.notna().any()},
        "mean_recon_rel_error": {e: float(g.recon_rel_error.mean()) for e, g in rows.groupby("experiment")
                                 if g.recon_rel_error.notna().any()},
        "reached_token_budget_rate": float(rows.reached_token_budget.fillna(False).astype(bool).mean()),
        "seconds_per_generation": float(rows.seconds.mean()),
        "features": meta["features"],
        "provenance": meta.get("provenance"),
        "outcome_note": ("Acceptance = regenerated A2 labeled adopted_critic under Behavior v0.1.2 rules. "
                         "SAE conditions replace the layer output at the final prompt token of the A2 "
                         "prefill with the SAE reconstruction; Δ is against the same SAE's no-ablation run, "
                         "paired by question. Random ablation is the per-question mean over the draws."),
    }
    (run_dir / "causal_replay_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    from mas_sae.probe.feature_selection import _plot
    _plot(table, run_dir / "causal_effects.svg",
          title=f"SAE ablation during Solver replay (layer {cfg['layer']}, {cfg['partition']}, N={n})",
          xlabel="Δ acceptance vs. that SAE's no-ablation run (pp, 95% CI)")
    return table


def _stored_responses(run_dir: Path, base: pd.DataFrame) -> np.ndarray:
    eps = pd.read_csv(run_dir / "episodes.csv").set_index("question_id")
    return eps.loc[base.question_id, "stored_solver_response"].to_numpy()


def run(cfg: ReplayConfig, config_path: Path | None = None, resume: Path | None = None) -> Path:
    from mas_sae.utils.versioning import create_versioned_run_dir

    episodes = load_episodes(cfg)
    logger.info("%d eligible %s episodes", len(episodes), cfg.partition)
    comp = load_components(cfg, episodes)
    run_dir = resume or create_versioned_run_dir(cfg.output_dir)
    replay(cfg, episodes, comp, run_dir, config_path)
    analyse(run_dir)
    logger.info("wrote %s", run_dir)
    return run_dir
