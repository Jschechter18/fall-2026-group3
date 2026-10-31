"""Causal replay (issue #90): regenerate the Solver's A2 with an SAE spliced into the residual stream.

For each eligible episode of one frozen partition, the stored A1 and Critic feedback are replayed
through the Solver's revise prompt with the same model, revision, prompt and greedy settings as
collection. The intervention is a final-prompt-token, prefill-only intervention: during the A2
prefill, the configured layer's output at the final prompt token (exactly where the collected
activations, probes and SAE features were measured) is replaced, and nothing else is: earlier
positions and decode-step computations at that layer are not modified. At the final layer the change
reaches only the logits of the first generated token, so any later effect comes through a different
generated token. At an earlier layer it also changes that position's keys and values in the layers
above, which every decode step attends to, so the answer can change even when the first token does
not. Either way, a null result says the features do not matter at this position, not that they are
unimportant throughout answer generation. Each row records the first token generation actually
produced and its log-probability, and the log-probability under the condition of the token its
control produced, as a continuous measure beside answer-level acceptance.

Each intervention sets a set of features of z = encode(x) at that position to fixed values, giving z':
  ablation           chosen features set to 0
  steering (s)       chosen feature f set to s * ref_f, where ref_f is its mean activation where it fires on
                     the feature-selection fit partition (``scripts/feature_reference_activations.py``):
                     s > 0 pushes the features up, s < 0 pushes them below zero (ablation is s = 0)

Two splices per SAE:
  reconstruction     x' = decode(z')                   (estimand: inside the SAE-reconstructed model)
  error-preserving   x' = x + decode(z') - decode(z)   (keeps the SAE error; no edit gives x)

Conditions per episode:
  Baseline       no SAE: plain replay; checked against the stored A2 before any SAE run (replay check)
  per SAE        no ablation (reconstruction only), then for ablation (if ``ablation``) and each steering
                 strength (``steer_strengths``): the targeted set and the random sets (draws)
  per SAE, EP    the same targeted / random edits with the error-preserving splice (control: plain replay)

The regenerated A2 is labeled with the same configured behavior-label rules as the stored data, and
acceptance means ``solver_response == "adopted_critic"``. Conclusions are about critic adoption on
eligible disagreement episodes only; claims about correctness need correctness-aware analysis.

Targeted and random feature sets come from the #89 feature-selection summary, so they are the
features reported there; each SAE's checkpoint hashes must match that run's.

The plain replays run first, for every episode, and are checked against collection before any SAE
condition is generated: a run whose replay does not reproduce collection stops (or, if configured,
continues marked unvalidated). Generations are appended to ``per_row.jsonl`` as they finish, so an
interrupted run resumes where it stopped and ``analyse`` can be run on a partial run at any time. A
resume must use the same inputs, code, environment and episode order as the run it continues (it
may only add episodes), and an unfinished last line from an interrupted write is cut off before
appending. Entry point: ``scripts/causal_replay.py``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import re
import subprocess
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

ACCEPT = "adopted_critic"
BASELINE = "Baseline"
INTERVENTION = "final_prompt_token_prefill_only"
MODES = ("reconstruction", "error_preserving")
EP_SUFFIX = " · error-preserving"
CODE_DIRS = ("src", "scripts", "configs")
PATH_KEYS = ("capstone_data_root", "interactions_jsonl", "labels_csv", "feature_selection_summary")
REFERENCE_KEYS = ("layer", "feature_selection_summary_sha256", "saes")
REQUIRED_KEYS = (*PATH_KEYS, "layer")
# provenance that must be unchanged for a resume to append to a run
IDENTITY_KEYS = ("code_tree_sha", "uncommitted_code_sha256", "untracked_code_sha256", "input_sha256",
                 "label_policy", "label_policy_sha256", "packages", "solver_spec", "site", "sae_sha256",
                 "device_name", "intervention")
PACKAGES = ("torch", "transformers", "datasets", "numpy", "pandas")
CONDITION_LABELS = {"no_sae": "No SAE (replayed A2)", "no_ablation": "No ablation",
                    "random_ablation": "Random ablation", "targeted_ablation": "Targeted ablation"}
ABLATION = "ablation"
TABLE_COLUMNS = ["Experiment", "Condition", "Outcome", "N", "Acceptance Rate (%)",
                 "Δ vs. SAE Control (pp)", "95% CI for Δ", "p-value"]


@dataclass
class ReplayConfig:
    # Data paths have no defaults: they come from the config passed to scripts/causal_replay.py.
    capstone_data_root: Path
    interactions_jsonl: str
    labels_csv: str
    feature_selection_summary: str
    layer: int                                       # Solver layer the SAEs are spliced into
    solver: dict = field(default_factory=dict)       # same role spec as the collection config
    musique_revision: str | None = None
    contexts_jsonl: str | None = None              # exact collected payloads for mixed datasets
    label_policy: str = "v0.1.2-qc"               # legacy default; candidates use v0.2.0-candidate
    partition: str = "intervention"
    # name -> {kind: custom | gemma_scope, path: ...}; must be SAEs of the feature-selection run
    saes: dict = field(default_factory=dict)
    n_random_draws: int = 5
    error_preserving: bool = True                    # also run the error-preserving splice
    ablation: bool = True                            # targeted / random ablation conditions
    # steering: clamp the targeted / random features to s * ref for each s here (empty: no steering)
    steer_strengths: list = field(default_factory=list)
    feature_reference: str | None = None             # reference activations (needed for steering)
    max_hours: float | None = None                   # warn when the SAE conditions are projected to take longer
    # replay check: the plain replay must reproduce collection before any SAE condition runs
    min_replay_label_match: float = 0.98             # share of episodes whose regenerated A2 keeps its label
    on_replay_check_fail: str = "stop"               # stop | mark_unvalidated
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
        missing = [k for k in REQUIRED_KEYS if raw.get(k) in (None, "")]
        if missing:
            raise ValueError(f"Config {path} must set {missing}.")
        for k in ("capstone_data_root", "output_dir"):
            if k in raw:
                raw[k] = Path(raw[k])
        cfg = cls(**raw)
        if cfg.label_policy not in bl.POLICIES:
            raise ValueError(f"Unknown label_policy {cfg.label_policy!r}")
        for name, spec in cfg.saes.items():
            if name == BASELINE or spec.get("kind") not in {"custom", "gemma_scope"} or "path" not in spec:
                raise ValueError(f"SAE {name!r} needs kind: custom|gemma_scope and a path.")
        if type(cfg.layer) is not int or cfg.layer < 0:
            raise ValueError("layer must be a nonnegative integer.")
        if cfg.limit is not None and (type(cfg.limit) is not int or cfg.limit <= 0):
            raise ValueError("limit must be a positive integer or null.")
        if type(cfg.n_random_draws) is not int or cfg.n_random_draws <= 0:
            raise ValueError("n_random_draws must be a positive integer.")
        if type(cfg.error_preserving) is not bool:
            raise ValueError("error_preserving must be true or false.")
        if type(cfg.ablation) is not bool:
            raise ValueError("ablation must be true or false.")
        s = cfg.steer_strengths
        if (not isinstance(s, list) or any(type(v) not in (int, float) or not np.isfinite(v) or v == 0 for v in s)
                or len(set(s)) != len(s)):
            raise ValueError("steer_strengths must be a list of distinct nonzero numbers (0 is ablation).")
        if not cfg.ablation and not s:
            raise ValueError("Nothing to run: set ablation: true or some steer_strengths.")
        if s and not cfg.feature_reference:
            raise ValueError("steer_strengths needs feature_reference (scripts/feature_reference_activations.py).")
        if cfg.max_hours is not None and (type(cfg.max_hours) not in (int, float) or cfg.max_hours <= 0):
            raise ValueError("max_hours must be a positive number or null.")
        if not 0 < cfg.min_replay_label_match <= 1:
            raise ValueError("min_replay_label_match must be in (0, 1].")
        if cfg.on_replay_check_fail not in ("stop", "mark_unvalidated"):
            raise ValueError("on_replay_check_fail must be stop or mark_unvalidated.")
        return cfg


# ---------------------------------------------------------------------------------------------
# The intervention


class SAESplice:
    """Replace a module's output at the final prompt token of the prefill with an SAE splice.

    ``generate`` calls the module once for the prompt (prefill) and once per new token. Only the
    prefill's last position is changed, which is where the collected activations were taken.
    ``features`` are set to ``values`` (zero when ``values`` is None: ablation) before decoding.
    ``mode`` is ``reconstruction`` (decode the edited features) or ``error_preserving`` (add the
    change the edit makes to the reconstruction to the original activation).
    Decoder layers return a tensor or a tuple whose first item is the hidden state; both work.
    """

    def __init__(self, model: torch.nn.Module, module_name: str, sae, features=(), mode: str = "reconstruction",
                 values=None):
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise ValueError(f"Module {module_name!r} was not found in the model.")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
        features = [int(f) for f in features]
        values = [0.0] * len(features) if values is None else [float(v) for v in values]
        if len(values) != len(features):
            raise ValueError(f"{len(features)} features but {len(values)} values.")
        self.module, self.sae = modules[module_name], sae
        self.features, self.values = features, values
        self.mode = mode
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
            x_norm = x.norm().clamp_min(1e-12)
            self.stats = {"n_active": int((z > 0).sum()),
                          "recon_rel_error": float((x - recon_full).norm() / x_norm)}
            recon = recon_full
            if self.features:
                # how many of the edited features were firing before the edit: 0 makes an ablation a no-op
                self.stats["n_ablated_active"] = int((z[0, self.features] > 0).sum())
                z = z.clone()
                z[:, self.features] = torch.tensor(self.values, dtype=z.dtype, device=z.device)
                recon = self.sae.decoder(z)
                # size of the change the edit makes to the reconstruction, relative to the activation
                self.stats["removed_rel_norm"] = float((recon_full - recon).norm() / x_norm)
            new = recon if self.mode == "reconstruction" else x + (recon - recon_full)
        hidden = hidden.clone()
        hidden[:, -1, :] = new.to(hidden.dtype)
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


class FirstTokenLogits:
    """The model's log-probabilities for the first generated token (its output at the last prefill position).

    These are the raw model outputs, before any logits processing ``generate`` applies; the token
    generation chose is read from the Solver, not from their argmax.
    """

    def __init__(self, model: torch.nn.Module):
        self.model, self.logprobs, self._handle = model, None, None

    def _hook(self, module, inputs, output):
        if self.logprobs is None:
            logits = output.logits if hasattr(output, "logits") else output[0]
            self.logprobs = logits[0, -1].float().log_softmax(-1)

    def __enter__(self) -> "FirstTokenLogits":
        self.logprobs = None
        self._handle = self.model.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        self._handle.remove()
        if exc[0] is None and self.logprobs is None:
            raise RuntimeError("The model's forward pass never ran during generation.")


# ---------------------------------------------------------------------------------------------
# Inputs


def load_episodes(cfg: ReplayConfig) -> pd.DataFrame:
    """Eligible episodes of ``cfg.partition`` in a seeded order, with their stored labels.

    Every eligible label must have exactly one stored Natural episode, and relabeling the stored
    A2 with the configured rules must reproduce labels.csv, so the regenerated A2s are labeled by
    exactly the rules that produced the stored labels.
    """
    from mas_sae.probe.behavior_data import read_jsonl

    root = Path(cfg.capstone_data_root)
    labels = bl.read_labels(root / cfg.labels_csv)
    policy = bl.POLICIES[cfg.label_policy]
    eligibility, target = (("eligible_strict", "strict_target") if policy.candidate_rules else ("eligible_primary", "primary_target"))
    versions = {lab["rule_version"] for lab in labels.values()}
    if versions != {policy.version}:
        raise ValueError(f"labels.csv rule_version {sorted(versions)} is not {policy.version!r}.")
    eligible = {q: lab for q, lab in labels.items()
                if lab["partition"] == cfg.partition and lab[eligibility]}
    if not eligible:
        raise ValueError(f"No {eligibility} labels in partition {cfg.partition!r}.")

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
        relabel = bl.classify_candidate(r, policy)
        if lab.get("episode_id") and lab["episode_id"] != r.get("episode_id"):
            raise ValueError(f"{qid}: label episode_id differs from the collected episode")
        if (relabel["solver_response"], relabel[target]) != (lab["solver_response"], lab[target]):
            raise ValueError(f"{qid}: relabeling the stored A2 gives {relabel['solver_response']!r}, "
                             f"labels.csv says {lab['solver_response']!r}.")
        rows.append({**{k: r[k] for k in policy.input_fields if k in r},
                     "a2_presentation": r.get("a2_presentation", "feedback"),
                     "candidate_order": r.get("candidate_order"), "question_id": qid, "source_split": r["source_split"], "question": r["question"],
                     "solver_attempt_1": r["solver_attempt_1"], "critic_feedback": r["critic_feedback"],
                     "critic_advocated_answer": r["critic_advocated_answer"],
                     "critic_noncommittal": r["critic_noncommittal"],
                     "stored_solver_attempt_2": r["solver_attempt_2"],
                     "stored_solver_response": lab["solver_response"],
                     "stored_adopted": int(lab[target]),
                     # exact A2 prompt token ids at collection, when the collection recorded them
                     "stored_a2_input_ids_sha256": ((r.get("generation_telemetry") or {}).get("solver_a2_prompt")
                                                    or {}).get("input_ids_sha256")})
    df = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    df = df.iloc[np.random.default_rng(cfg.seed).permutation(len(df))].reset_index(drop=True)
    return df.head(cfg.limit) if cfg.limit else df


def load_paragraphs(episodes: pd.DataFrame, revision: str | None, contexts_path: Path | None = None) -> dict[str, list]:
    """Exact exported contexts, or the original MuSiQue lookup for legacy runs."""
    if contexts_path is not None:
        from mas_sae.experiments.records import read_jsonl
        contexts = read_jsonl(contexts_path)
        by_id = {r["question_id"]: r["payload"] for r in contexts}
        if len(by_id) != len(contexts):
            raise ValueError("Duplicate context question IDs")
        out = {}
        for ep in episodes.to_dict("records"):
            payload = by_id.get(ep["question_id"])
            if payload is None or payload.get("question") != ep["question"] or not isinstance(payload.get("paragraphs"), list):
                raise ValueError(f"{ep['question_id']}: missing or mismatched collected context")
            out[ep["question_id"]] = payload["paragraphs"]
        return out
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
        # ids are kept exactly as stored: validate_features rejects anything that is not an integer
        out[name] = {"targeted": list(s["targeted_features"]),
                     "random": [list(d) for d in draws[:cfg.n_random_draws]]}
    if cfg.steer_strengths:
        refs = reference_activations(cfg, sae_hashes)
        for name, f in out.items():
            ids = {str(i) for i in [*f["targeted"], *(i for d in f["random"] for i in d)]}
            f["ref"] = {i: r for i, r in refs[name].items() if i in ids}
    return out


def reference_activations(cfg: ReplayConfig, sae_hashes: dict) -> dict:
    """``{sae: {feature id (str): ref}}`` from ``cfg.feature_reference``, after checking it was computed for
    this layer, this feature-selection summary and these SAE files."""
    root = Path(cfg.capstone_data_root)
    ref = json.loads((root / cfg.feature_reference).read_text())
    missing = [k for k in REFERENCE_KEYS if k not in ref]
    if missing:
        raise ValueError(f"{cfg.feature_reference} is missing {missing}.")
    if ref["layer"] != cfg.layer:
        raise ValueError(f"Reference activations are for layer {ref['layer']}, not {cfg.layer}.")
    if ref["feature_selection_summary_sha256"] != sha256_file(root / cfg.feature_selection_summary):
        raise ValueError(f"{cfg.feature_reference} was computed from a different feature-selection summary.")
    out = {}
    for name in cfg.saes:
        if name not in ref["saes"]:
            raise ValueError(f"SAE {name!r} is not in {cfg.feature_reference}.")
        if ref["saes"][name]["sae_sha256"] != sae_hashes[name]:
            raise ValueError(f"SAE {name!r} files differ from the ones the reference activations used.")
        out[name] = {i: v["ref"] for i, v in ref["saes"][name]["features"].items()}
    return out


def validate_features(features: dict, saes: dict) -> None:
    """Refuse feature sets that cannot be a fair targeted-vs-random comparison on these SAEs.

    Each SAE needs a non-empty targeted set and random sets of the same size, all distinct integer
    latents of that SAE, with no targeted feature in any random set.
    """
    if set(features) != set(saes):
        raise ValueError(f"Feature sets are for {sorted(features)}, SAEs are {sorted(saes)}.")
    for name, f in features.items():
        width = saes[name].latent_dim
        sets = {"targeted": f["targeted"], **{f"random draw {i}": d for i, d in enumerate(f["random"])}}
        ref = f.get("ref")
        for label, ids in sets.items():
            if not ids or any(type(i) is not int for i in ids):
                raise ValueError(f"{name} {label}: feature ids must be a non-empty list of integers.")
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} {label}: feature ids repeat.")
            if min(ids) < 0 or max(ids) >= width:
                raise ValueError(f"{name} {label}: feature ids must be in [0, {width}).")
            if len(ids) != len(f["targeted"]):
                raise ValueError(f"{name} {label}: {len(ids)} features, targeted set has {len(f['targeted'])}.")
            if label != "targeted" and set(ids) & set(f["targeted"]):
                raise ValueError(f"{name} {label}: includes targeted features.")
            bad = [i for i in ids if ref is not None and not (np.isfinite(ref.get(str(i), np.nan)) and ref[str(i)] > 0)]
            if bad:
                raise ValueError(f"{name} {label}: no positive reference activation for features {bad}.")


def steer_arm(strength: float) -> str:
    return f"steer_{strength:+g}"


def split_condition(condition: str) -> tuple[str, str] | None:
    """``("targeted" | "random", arm)`` for an edit condition (arm: ``ablation`` or ``steer_<s>``), else None."""
    kind, _, arm = condition.partition("_")
    return (kind, arm) if kind in ("targeted", "random") else None


def condition_label(condition: str) -> str:
    if condition in CONDITION_LABELS:
        return CONDITION_LABELS[condition]
    kind, arm = split_condition(condition)
    return f"{kind.capitalize()} steer ({arm.removeprefix('steer_')}× ref)"


def conditions(features: dict, error_preserving: bool = False, ablation: bool = True,
               steer_strengths=()) -> list[dict]:
    """Every (experiment, condition, draw) generated per episode, with its SAE, splice mode, edited
    features and the values they are set to (None: zero, i.e. ablation).

    Order matters: each condition comes after its control (see ``control_key``) for the same episode.
    """
    out = [{"experiment": BASELINE, "condition": "no_sae", "draw": None, "features": None, "values": None,
            "strength": None, "sae": None, "mode": None}]
    for name, f in features.items():
        sets = [("targeted", None, f["targeted"]), *(("random", i, d) for i, d in enumerate(f["random"]))]
        arms = ([(ABLATION, None)] if ablation else []) + [(steer_arm(s), s) for s in steer_strengths]
        for mode in MODES if error_preserving else MODES[:1]:
            exp = name if mode == "reconstruction" else name + EP_SUFFIX
            c = {"experiment": exp, "sae": name, "mode": mode}
            if mode == "reconstruction":       # error-preserving without an edit is the plain replay
                out.append({**c, "condition": "no_ablation", "draw": None, "features": [], "values": None,
                            "strength": None})
            for arm, s in arms:
                out += [{**c, "condition": f"{kind}_{arm}", "draw": draw, "features": ids, "strength": s,
                         "values": None if s is None else [s * f["ref"][str(i)] for i in ids]}
                        for kind, draw, ids in sets]
    return out


def control_key(c: dict) -> tuple | None:
    """(experiment, condition, draw) that ``c`` is compared with on the same episode.

    Reconstruction ablations are compared with the same SAE's reconstruction-only run; that run and
    error-preserving ablations are compared with the plain replay (an error-preserving splice
    without ablation leaves the activation unchanged).
    """
    if c["experiment"] == BASELINE:
        return None
    if c.get("mode", "reconstruction") == "reconstruction" and c["condition"] != "no_ablation":
        return (c["experiment"], "no_ablation", None)
    return (BASELINE, "no_sae", None)


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
    saes = {name: load_sae(sae_spec, root, cfg.layer)[0].eval() for name, sae_spec in cfg.saes.items()}
    validate_features(features, saes)                    # before the (slow) model load
    paragraphs = load_paragraphs(episodes, cfg.musique_revision, root / cfg.contexts_jsonl if cfg.contexts_jsonl else None)

    spec = resolve_role_spec("solver", cfg.solver, default_max_new_tokens=64)
    model, processor = load_spec(spec)
    solver = Solver(model, processor)
    configure_agent(solver, spec)
    site = resolve_solver_sites(model, spec["loader"], [cfg.layer])[0]
    saes = {name: sae.to(model.device) for name, sae in saes.items()}
    return Components(solver, model, site, saes, features, paragraphs,
                      {"solver_spec": spec, "sae_sha256": hashes, "num_layers": num_layers(model),
                       "device": str(model.device),
                       "device_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None})


def num_layers(model) -> int | None:
    """Number of decoder layers of a (possibly multimodal) Hugging Face model, if its config says."""
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None) or config
    return getattr(text, "num_hidden_layers", None)


def intervention_description(layer: int, n_layers: int | None) -> str:
    """What a final-prompt-token, prefill-only splice at ``layer`` can affect, for the summary."""
    base = ("The layer output at the final prompt token of the A2 prefill is replaced; earlier positions "
            "and decode-step computations at this layer are not. ")
    if n_layers is not None and layer == n_layers - 1:
        return base + ("This is the final layer, so the change reaches only the first generated token's logits: "
                       "any effect on the answer comes through a different first token.")
    return base + ("This is not the final layer (or the depth is unknown), so the change also alters that "
                   "position's keys and values in later layers, which every decode step attends to: the answer "
                   "can change even when the first token does not.")


def generate(comp: Components, ep: dict, cond: dict, control_token: int | None = None) -> dict:
    """One regenerated A2 under ``cond``, labeled, with telemetry.

    ``control_token`` is the first token its control generated; the row records its log-probability here.
    """
    splice = (SAESplice(comp.model, comp.site, comp.saes[cond.get("sae", cond["experiment"])], cond["features"],
                        cond.get("mode") or "reconstruction", cond.get("values"))
              if cond["features"] is not None else contextlib.nullcontext())
    t0 = time.perf_counter()
    with splice, FirstTokenLogits(comp.model) as first:
        if ep.get("a2_presentation", "feedback") == "candidates":
            a2 = comp.solver.choose(ep["question"], comp.paragraphs[ep["question_id"]],
                                    ep["solver_attempt_1"], ep["critic_advocated_answer"], ep["candidate_order"])
        else:
            a2 = comp.solver.revise(ep["question"], comp.paragraphs[ep["question_id"]],
                                    ep["solver_attempt_1"], ep["critic_feedback"])
    seconds = time.perf_counter() - t0
    policy = bl.POLICIES[comp.info.get("label_policy_name", "v0.1.2-qc")]
    generated_ids = comp.solver.last_generated_token_ids or []
    label = bl.classify_candidate({**ep, "solver_attempt_2": a2,
        "solver_a2_first_token_id": generated_ids[0] if generated_ids else None}, policy)
    gen = comp.solver.last_generation or {}
    prompt = comp.solver.last_prompt or {}
    stats = splice.stats if isinstance(splice, SAESplice) else {}
    # the token generation actually chose (after any logits processing), scored by the model's raw log-probs
    token = generated_ids[0] if generated_ids else None
    stored_prompt = ep.get("stored_a2_input_ids_sha256")
    stored_prompt = stored_prompt if isinstance(stored_prompt, str) else None
    return {"question_id": ep["question_id"], "experiment": cond["experiment"],
            "condition": cond["condition"], "draw": cond["draw"], "mode": cond.get("mode"),
            "strength": cond.get("strength"),
            "solver_attempt_2": a2, "stored_solver_attempt_2": ep["stored_solver_attempt_2"],
            "a2_matches_stored": a2 == ep["stored_solver_attempt_2"],
            "solver_response": label["solver_response"], "adopted": int(label["solver_response"] == ACCEPT),
            "stored_solver_response": ep["stored_solver_response"], "stored_adopted": ep["stored_adopted"],
            "generated_tokens": gen.get("generated_tokens"), "reached_token_budget": gen.get("reached_token_budget"),
            "prompt_tokens": prompt.get("prompt_tokens"), "input_ids_sha256": prompt.get("input_ids_sha256"),
            "prompt_matches_stored": None if stored_prompt is None else prompt.get("input_ids_sha256") == stored_prompt,
            "first_token_id": token,
            "first_token_logprob": None if token is None else float(first.logprobs[token]),
            "control_first_token_logprob": None if control_token is None else float(first.logprobs[control_token]),
            "n_active_features": stats.get("n_active"), "n_ablated_active": stats.get("n_ablated_active"),
            "removed_rel_norm": stats.get("removed_rel_norm"),
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
        "git_commit": get_git_commit(), "code_tree_sha": _code_tree_sha(),
        "uncommitted_changes_sha256": uncommitted, "uncommitted_code_sha256": _code_diff_sha256(),
        "untracked_code_files": untracked, "untracked_code_sha256": get_untracked_sha256(*CODE_DIRS),
        "command": get_run_command(),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": sha256_file(config_path) if config_path else None,
        "input_sha256": {p: sha256_file(root / p) for p in
                         (cfg.interactions_jsonl, cfg.labels_csv, cfg.feature_selection_summary, cfg.feature_reference, cfg.contexts_jsonl)
                         if p},
        "label_policy": bl.POLICIES[cfg.label_policy].version, "label_policy_sha256": bl.POLICIES[cfg.label_policy].definition_sha256(),
        "packages": {p: get_package_version(p) for p in PACKAGES},
        "site": comp.site, "intervention": INTERVENTION,
        **comp.info,
    }


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def _code_tree_sha() -> dict:
    """Git tree id of each code directory at HEAD: changes when code is committed, not when results are."""
    try:
        return {d: _git("rev-parse", f"HEAD:{d}").decode().strip() for d in CODE_DIRS}
    except (OSError, subprocess.CalledProcessError):
        return {d: "unknown" for d in CODE_DIRS}


def _code_diff_sha256() -> str:
    """Hash of uncommitted changes to tracked code only (edits to notebooks or docs don't count)."""
    try:
        return hashlib.sha256(_git("diff", "--no-ext-diff", "--binary", "HEAD", "--", *CODE_DIRS)).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_dict(cfg: ReplayConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=str))


def replay(cfg: ReplayConfig, episodes: pd.DataFrame, comp: Components, run_dir: Path,
           config_path: Path | None = None, preflight_only: bool = False) -> Path:
    """Generate every missing (episode, condition) into ``run_dir/per_row.jsonl``; resumable.

    The plain replays of all episodes run first and must pass the replay check (``replay_check``)
    before any SAE condition is generated; ``preflight_only`` stops after that check.
    """
    validate_features(comp.features, comp.saes)
    if cfg.steer_strengths and any("ref" not in f for f in comp.features.values()):
        raise ValueError("Steering needs reference activations for every SAE's features.")
    run_dir.mkdir(parents=True, exist_ok=True)
    conds = conditions(comp.features, cfg.error_preserving, cfg.ablation, cfg.steer_strengths)
    comp.info["label_policy_name"] = cfg.label_policy
    meta_path = run_dir / "replay_config.json"
    meta = {"config": _config_dict(cfg), "conditions": conds, "features": comp.features}
    prov = provenance(cfg, config_path, comp)
    manifest = episodes.drop(columns=["question", "critic_feedback"])
    if meta_path.exists():
        _check_resume(run_dir, json.loads(meta_path.read_text()), meta, prov, manifest)
        logger.info("resuming %s", run_dir)
    else:
        meta_path.write_text(json.dumps({**meta, "provenance": prov}, indent=2, default=str))
    manifest_path = run_dir / "episodes.csv"
    if not manifest_path.exists() or len(manifest) > len(pd.read_csv(manifest_path)):
        manifest.to_csv(manifest_path, index=False)          # first start, or a checked extension

    out = run_dir / "per_row.jsonl"
    _repair_tail(out)
    rows = _read_rows(out)
    unknown = {r["question_id"] for r in rows} - set(episodes.question_id)
    if unknown:
        raise ValueError(f"{out} has generations for {len(unknown)} episodes outside this run's episodes.")
    first = {key(r): r.get("first_token_id") for r in rows}
    todo = [(ep, c) for ep in episodes.to_dict("records") for c in conds
            if (ep["question_id"], c["experiment"], c["condition"], c["draw"]) not in first]
    logger.info("%d episodes x %d conditions; %d generations already done, %d to go",
                len(episodes), len(conds), len(first), len(todo))
    _generate_all([t for t in todo if t[1]["features"] is None], comp, out, first, "plain replay")

    generated = pd.DataFrame(_read_rows(out))
    check = replay_check(generated, manifest, cfg.min_replay_label_match)
    (run_dir / "replay_check.json").write_text(json.dumps(check, indent=2))
    _mismatches(generated, manifest).to_csv(run_dir / "replay_check_mismatches.csv", index=False)
    if preflight_only:
        logger.info("replay check %s; stopping after the preflight", "passed" if check["passed"] else "FAILED")
        return run_dir
    if not check["passed"]:
        msg = (f"Plain replay does not reproduce collection ({check['failures']}); see "
               f"{run_dir / 'replay_check_mismatches.csv'}.")
        if cfg.on_replay_check_fail == "stop":
            raise RuntimeError(msg + " Stopping before the SAE conditions.")
        logger.warning("%s Continuing; the run is marked unvalidated.", msg)
    _generate_all([t for t in todo if t[1]["features"] is not None], comp, out, first, "SAE condition", cfg.max_hours)
    return run_dir


def _generate_all(todo: list, comp: Components, out: Path, first: dict, what: str,
                  max_hours: float | None = None) -> None:
    """Generate and append ``todo`` in order; ``first`` maps done keys to their first generated token.

    With ``max_hours``, warns once (after 200 generations) if ``todo`` is projected to take longer.
    """
    t0 = time.perf_counter()
    with out.open("a") as f:
        for i, (ep, c) in enumerate(todo, 1):
            ctl = control_key(c)
            # conditions() orders each control before what is compared with it, so its row exists
            row = generate(comp, ep, c, None if ctl is None else first[(ep["question_id"], *ctl)])
            first[key(row)] = row["first_token_id"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            if max_hours and i == min(200, len(todo)) and (hours := (time.perf_counter() - t0) / i * len(todo) / 3600) > max_hours:
                logger.warning("%d %s generations are projected to take %.1f h, more than max_hours=%g; "
                               "the run can be stopped and analysed on the episodes it completed",
                               len(todo), what, hours, max_hours)
            if i % 50 == 0 or i == len(todo):
                rate = (time.perf_counter() - t0) / i
                logger.info("%d/%d %s generations, %.2fs each, ~%.1f min left",
                            i, len(todo), what, rate, rate * (len(todo) - i) / 60)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def _mismatch_kind(r) -> str:
    if r.prompt_matches_stored is False:
        return "prompt_differs"
    if r.solver_attempt_2 == r.stored_solver_attempt_2:
        return "exact_match"
    if r.solver_response != r.stored_solver_response:
        return "label_changed"
    if _normalize(r.solver_attempt_2) == _normalize(r.stored_solver_attempt_2):
        return "formatting_only"
    return "text_differs_same_label"


def _plain(rows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Plain-replay rows of the manifest's episodes, with the stored response and a mismatch kind."""
    base = rows[rows.experiment == BASELINE] if len(rows) else rows
    if base.empty:
        return pd.DataFrame(columns=["question_id", "kind"])
    stored = manifest.assign(question_id=manifest.question_id.astype(str)).set_index("question_id")
    base = base.assign(question_id=base.question_id.astype(str))
    base = base[base.question_id.isin(stored.index)].copy()
    base["stored_solver_response"] = stored.loc[base.question_id, "stored_solver_response"].to_numpy()
    if "prompt_matches_stored" not in base:
        base["prompt_matches_stored"] = None
    base["prompt_matches_stored"] = base.prompt_matches_stored.astype(object).where(
        base.prompt_matches_stored.notna(), None)
    base["kind"] = [_mismatch_kind(r) for r in base.itertuples()]
    return base


def replay_check(rows: pd.DataFrame, manifest: pd.DataFrame, min_label_match: float) -> dict:
    """Does the plain replay reproduce collection? Counts each kind of mismatch and applies the criterion.

    Match rates compare answers and labels directly. Each episode also gets one kind, in order of
    precedence (so a prompt difference that also changed the label counts as ``prompt_differs``):
    ``prompt_differs`` (prompt token ids differ from collection's,
    where collection recorded them), ``exact_match``, ``label_changed`` (the regenerated A2 gets a
    different Behavior label), ``formatting_only`` (same text up to case, punctuation and spacing)
    and ``text_differs_same_label`` (a different answer with the same label, e.g. numerical noise).
    Passing needs no prompt differences and a label match rate of at least ``min_label_match``.
    """
    base = _plain(rows, manifest)
    n = len(base)
    counts = base.kind.value_counts().to_dict() if n else {}
    # match rates come from direct comparisons; the kinds (one per episode) only explain mismatches
    label_match = float((base.solver_response == base.stored_solver_response).mean()) if n else float("nan")
    text_match = float((base.solver_attempt_2 == base.stored_solver_attempt_2).mean()) if n else float("nan")
    failures = []
    if n < len(manifest):
        failures.append(f"{len(manifest) - n} episodes have no plain replay yet")
    if counts.get("prompt_differs"):
        failures.append(f"{counts['prompt_differs']} prompts differ from collection")
    if n and label_match < min_label_match:
        failures.append(f"label match {label_match:.3f} < {min_label_match}")
    return {"n_episodes": n, "n_planned": int(len(manifest)), "mismatch_counts": counts,
            "a2_exact_match_rate": text_match,
            "solver_response_match_rate": label_match,
            "n_prompt_hashes_compared": int(base.prompt_matches_stored.notna().sum()) if n else 0,
            "stored_acceptance_pct": 100 * float((base.stored_solver_response == ACCEPT).mean()) if n else None,
            "replayed_acceptance_pct": 100 * float(base.adopted.mean()) if n else None,
            "criterion": {"min_label_match": min_label_match, "max_prompt_differences": 0},
            "passed": not failures, "failures": failures}


def _mismatches(rows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Every plain replay that is not an exact match, with both answers and labels, for inspection."""
    base = _plain(rows, manifest)
    cols = ["question_id", "kind", "stored_solver_attempt_2", "solver_attempt_2", "stored_solver_response",
            "solver_response", "prompt_matches_stored", "first_token_logprob"]
    return base[base.kind != "exact_match"].reindex(columns=cols)


def _check_resume(run_dir: Path, old: dict, meta: dict, prov: dict, manifest: pd.DataFrame) -> None:
    """Refuse to append to ``run_dir`` unless the new generations would come from the same experiment."""
    # what was generated must not change; limit may grow (to extend a smoke run into the full run)
    same = lambda m: ({k: v for k, v in m["config"].items() if k != "limit"}, m["conditions"], m["features"])
    if same(old) != same(json.loads(json.dumps(meta))):
        raise ValueError(f"{run_dir} was started with a different config or feature sets; use a new run.")
    old_prov, new_prov = old.get("provenance") or {}, json.loads(json.dumps(prov, default=str))
    changed = [k for k in IDENTITY_KEYS if old_prov.get(k) != new_prov.get(k)]
    if changed:
        raise ValueError(f"{run_dir} was generated with different {changed}; use a new run.")
    # the saved manifest must be a prefix of this run's episodes: same episodes, same order, only more
    saved = pd.read_csv(run_dir / "episodes.csv", dtype={"question_id": str})
    if len(manifest) < len(saved):
        raise ValueError(f"{run_dir} has {len(saved)} episodes; resuming with {len(manifest)} would shrink it.")
    head = manifest.head(len(saved)).reset_index(drop=True)
    fresh = pd.read_csv(io.StringIO(head.to_csv(index=False)), dtype={"question_id": str})
    if list(fresh.columns) != list(saved.columns) or not fresh.equals(saved):
        raise ValueError(f"{run_dir}/episodes.csv does not match the first {len(saved)} episodes of this run.")


def _repair_tail(path: Path) -> None:
    """Cut off an unfinished last record (an interrupted write) so the next append starts on a new line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        logger.warning("removing an unfinished last record from %s (%d bytes, interrupted write)",
                       path, len(data) - keep)
        with path.open("r+b") as f:
            f.truncate(keep)


def _read_rows(path: Path) -> list[dict]:
    """Complete records of ``path``; an unfinished last line (a write in progress) is ignored.

    Any other unreadable line means the file is corrupt, not interrupted, and is an error.
    """
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.split("\n")
    if lines[-1]:
        logger.warning("ignoring an unfinished last record in %s (interrupted or in-progress write)", path)
    rows = []
    for i, line in enumerate(lines[:-1], 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {i} is not valid JSON: the file is corrupt.") from e
    return rows


# ---------------------------------------------------------------------------------------------
# Analysis


# per-row fields that older runs may not have
ROW_FIELDS = ("mode", "strength", "first_token_id", "first_token_logprob", "control_first_token_logprob",
              "removed_rel_norm", "n_ablated_active", "recon_rel_error", "prompt_matches_stored",
              "reached_token_budget")
TVR_COLUMNS = ["Experiment", "Splice", "Arm", "N", "Random draws", "Targeted − random (pp)", "95% CI (questions)",
               "95% CI (questions + draws)", "p-value", "p-value (Holm)", "Draw tail fraction (exploratory)"]
DOSE_COLUMNS = ["Experiment", "Splice", "Arm", "Strength (× ref)", "N", "Targeted acceptance (%)",
                "Targeted Δ vs. control (pp)", "Random acceptance (%)", "Random Δ vs. control (pp)",
                "Targeted − random (pp)"]
BIDIRECTIONAL_COLUMNS = ["Experiment", "Splice", "Strength (× ref)", "N", "Targeted Δ(+s) − Δ(−s) (pp)",
                         "Random Δ(+s) − Δ(−s) (pp)", "Targeted − random (pp)", "95% CI", "p-value"]


def arms_of(conds: list[dict]) -> list[str]:
    """Edit arms of a run in condition order: ``ablation`` and/or ``steer_<s>``."""
    return list(dict.fromkeys(split_condition(c["condition"])[1] for c in conds if split_condition(c["condition"])))


def arm_strength(arm: str) -> float:
    """The arm's clamp in units of ref: ablation sets the features to 0."""
    return 0.0 if arm == ABLATION else float(arm.removeprefix("steer_"))


def analyse(run_dir: Path, n_bootstrap: int | None = None, n_permutations: int | None = None) -> pd.DataFrame:
    """Write the results of ``run_dir`` and return the causal table.

      causal_table.csv           acceptance per condition; Δ vs. the condition's control, paired by question
      targeted_vs_random.csv     targeted minus mean random edit per SAE, splice and arm (ablation, each
                                 steering strength); Holm across SAEs within the ablation family and within
                                 the steering family (reconstruction splice)
      dose_response.csv          per SAE and splice: targeted and random acceptance at each strength (ablation = 0)
      steering_bidirectional.csv targeted vs. random difference between pushing up (+s) and down (−s)
      random_draws.csv           each random draw's effect, and targeted minus that draw
      reconstruction_damage.csv  reconstruction only vs. plain replay: acceptance, answers, first token
      diagnostics.csv            per condition: answer, label and first-token changes vs. its control, the
                                 control token's log-prob shift, empty and truncated answers, edited features
                                 inactive before the edit and the size of the edit
      transitions.csv            control response -> condition response counts (random draws pooled)
      changed_answers.csv        every generation whose answer differs from its control's
      per_row.csv, causal_replay_summary.json, causal_effects.svg

    Uses only episodes with every condition generated, so a partial run gives a valid (smaller) table.
    """
    meta = json.loads((run_dir / "replay_config.json").read_text())
    cfg, conds = meta["config"], meta["conditions"]
    n_bootstrap = n_bootstrap or cfg["n_bootstrap"]
    n_permutations = n_permutations or cfg["n_permutations"]
    seed = cfg["seed"]
    expected = {(c["experiment"], c["condition"], c["draw"]) for c in conds}
    all_rows = pd.DataFrame(_read_rows(run_dir / "per_row.jsonl"))
    if all_rows.empty:
        raise ValueError(f"No generations in {run_dir}.")
    rows = all_rows.drop_duplicates(["question_id", "experiment", "condition", "draw"], keep="last")
    have = rows.assign(k=list(zip(rows.experiment, rows.condition, rows.draw))).groupby("question_id")["k"].agg(set)
    complete = sorted(q for q, ks in have.items() if expected <= {(e, c, None if pd.isna(d) else int(d))
                                                                  for e, c, d in ks})
    rows = rows[rows.question_id.isin(complete)].sort_values(["question_id", "experiment", "condition", "draw"])
    rows = rows.reindex(columns=list(dict.fromkeys([*rows.columns, *ROW_FIELDS])))
    rows.to_csv(run_dir / "per_row.csv", index=False)
    if not complete:
        raise ValueError(f"No episode in {run_dir} has every condition yet.")
    rows["d"] = rows.draw.fillna(-1).astype(int)
    n, groups = len(complete), np.array(complete)

    def outcome(exp, cond) -> np.ndarray:            # per question; random draws averaged
        r = rows[(rows.experiment == exp) & (rows.condition == cond)]
        return r.groupby("question_id")["adopted"].mean().reindex(complete).to_numpy(float)

    def by_draw(exp, arm) -> np.ndarray:             # questions x draws
        r = rows[(rows.experiment == exp) & (rows.condition == f"random_{arm}")]
        return r.pivot(index="question_id", columns="d", values="adopted").reindex(complete).to_numpy(float)

    experiments = [e for e in dict.fromkeys(c["experiment"] for c in conds) if e != BASELINE]
    arms = arms_of(conds)
    edited = {c["experiment"]: c for c in reversed(conds) if split_condition(c["condition"])}
    control = {e: control_key(edited[e])[:2] for e in experiments}
    splice = {e: edited[e].get("mode") or "reconstruction" for e in experiments}
    plain = outcome(BASELINE, "no_sae")
    stored = rows[rows.experiment == BASELINE].set_index("question_id")["stored_adopted"].reindex(complete)

    # ---- acceptance per condition against its control
    table = [{"Experiment": BASELINE, "Condition": "No SAE (stored A2)", "Outcome": "observed", "N": n,
              "Acceptance Rate (%)": 100 * float(stored.mean())},
             {"Experiment": BASELINE, "Condition": CONDITION_LABELS["no_sae"], "Outcome": "regenerated", "N": n,
              "Acceptance Rate (%)": 100 * float(plain.mean())}]
    effects = {}
    for exp in experiments:
        ref = outcome(*control[exp])
        table.append({"Experiment": exp, "Outcome": "regenerated", "N": n, "Acceptance Rate (%)": 100 * float(ref.mean()),
                      "Condition": "No ablation" if control[exp][0] == exp else "No ablation (= plain replay)",
                      "Δ vs. SAE Control (pp)": "Reference"})
        for arm in arms:
            for cond in (f"random_{arm}", f"targeted_{arm}"):
                y = outcome(exp, cond)
                e = offline.paired_effect(y, ref, groups, n_bootstrap, n_permutations, seed)
                effects[f"{exp}__{cond}"] = e
                table.append({"Experiment": exp, "Condition": condition_label(cond), "Outcome": "regenerated",
                              "N": n, "Acceptance Rate (%)": 100 * float(y.mean()),
                              "Δ vs. SAE Control (pp)": e["delta_pp"], "95% CI for Δ": _ci(e["ci95_pp"]),
                              "p-value": e["p_value"]})
    table = pd.DataFrame(table).reindex(columns=TABLE_COLUMNS)
    table.to_csv(run_dir / "causal_table.csv", index=False)

    # ---- targeted vs. random, directly; Holm across the reconstruction-splice tests of each family
    tvr, draws, dose = [], [], []
    for exp in experiments:
        ref = outcome(*control[exp])
        for arm in arms:
            t, r = outcome(exp, f"targeted_{arm}"), by_draw(exp, arm)
            res = offline.targeted_vs_random(t, r, ref, groups, n_bootstrap, n_permutations, seed)
            tvr.append({"experiment": exp, "splice": splice[exp], "arm": arm, **res})
            dose.append({"Experiment": exp, "Splice": splice[exp], "Arm": arm, "Strength (× ref)": arm_strength(arm),
                         "N": n, "Targeted acceptance (%)": 100 * float(t.mean()),
                         "Targeted Δ vs. control (pp)": 100 * float((t - ref).mean()),
                         "Random acceptance (%)": 100 * float(r.mean()),
                         "Random Δ vs. control (pp)": 100 * float((r.mean(1) - ref).mean()),
                         "Targeted − random (pp)": res["delta_pp"]})
            for d in range(r.shape[1]):
                e = offline.paired_effect(r[:, d], ref, groups, n_bootstrap, n_permutations, seed)
                draws.append({"Experiment": exp, "Arm": arm, "Draw": d,
                              "Acceptance Rate (%)": 100 * float(np.mean(r[:, d])),
                              "Δ vs. control (pp)": e["delta_pp"], "95% CI for Δ": _ci(e["ci95_pp"]),
                              "p-value": e["p_value"], "Targeted − this draw (pp)": 100 * float(np.mean(t - r[:, d]))})
    for family in (lambda a: a == ABLATION, lambda a: a != ABLATION):
        confirmatory = [x for x in tvr if x["splice"] == "reconstruction" and family(x["arm"])]
        for x, adj in zip(confirmatory, offline.holm([x["p_value"] for x in confirmatory])):
            x["p_value_holm"] = adj
    pd.DataFrame([{"Experiment": x["experiment"], "Splice": x["splice"], "Arm": x["arm"], "N": n,
                   "Random draws": x["n_random_draws"],
                   "Targeted − random (pp)": x["delta_pp"], "95% CI (questions)": _ci(x["ci95_pp"]),
                   "95% CI (questions + draws)": _ci(x["ci95_pp_questions_and_draws"]), "p-value": x["p_value"],
                   "p-value (Holm)": x.get("p_value_holm"),
                   "Draw tail fraction (exploratory)": x["draw_tail_fraction"]} for x in tvr],
                 columns=TVR_COLUMNS).to_csv(run_dir / "targeted_vs_random.csv", index=False)
    pd.DataFrame(draws).to_csv(run_dir / "random_draws.csv", index=False)
    dose.sort(key=lambda x: (experiments.index(x["Experiment"]), x["Strength (× ref)"]))
    pd.DataFrame(dose, columns=DOSE_COLUMNS).to_csv(run_dir / "dose_response.csv", index=False)

    # ---- pushing up vs. pushing down: does the targeted set move acceptance in opposite directions, more than random?
    bidirectional = []
    strengths = {arm_strength(a) for a in arms if a != ABLATION}
    for exp in experiments:
        for s in sorted(v for v in strengths if v > 0 and -v in strengths):
            up, down = steer_arm(s), steer_arm(-s)
            t = outcome(exp, f"targeted_{up}") - outcome(exp, f"targeted_{down}")
            r = by_draw(exp, up).mean(1) - by_draw(exp, down).mean(1)
            e = offline.paired_effect(t, r, groups, n_bootstrap, n_permutations, seed)
            bidirectional.append({"Experiment": exp, "Splice": splice[exp], "Strength (× ref)": s, "N": n,
                                  "Targeted Δ(+s) − Δ(−s) (pp)": 100 * float(t.mean()),
                                  "Random Δ(+s) − Δ(−s) (pp)": 100 * float(r.mean()),
                                  "Targeted − random (pp)": e["delta_pp"], "95% CI": _ci(e["ci95_pp"]),
                                  "p-value": e["p_value"]})
    pd.DataFrame(bidirectional, columns=BIDIRECTIONAL_COLUMNS).to_csv(run_dir / "steering_bidirectional.csv",
                                                                      index=False)

    # ---- what changed, against each condition's control
    pairs = _pairs(rows, conds)
    diagnostics = [_diagnostics(rows[rows.experiment == BASELINE], None, BASELINE, "no_sae", complete, groups,
                                n_bootstrap, seed)]
    diagnostics += [_diagnostics(g, g, exp, cond, complete, groups, n_bootstrap, seed)
                    for (exp, cond), g in pairs.groupby(["experiment", "condition"], sort=False)]
    pd.DataFrame(diagnostics).to_csv(run_dir / "diagnostics.csv", index=False)
    damage = []
    for exp in [e for e in experiments if splice[e] == "reconstruction"]:
        g = pairs[(pairs.experiment == exp) & (pairs.condition == "no_ablation")]
        recon = outcome(exp, "no_ablation")
        e = offline.paired_effect(recon, plain, groups, n_bootstrap, n_permutations, seed)
        diag = _diagnostics(g, g, exp, "no_ablation", complete, groups, n_bootstrap, seed)
        damage.append({"Experiment": exp, "N": n, "Plain replay acceptance (%)": 100 * float(plain.mean()),
                       "Reconstruction-only acceptance (%)": 100 * float(recon.mean()),
                       "Δ (pp)": e["delta_pp"], "95% CI for Δ": _ci(e["ci95_pp"]), "p-value": e["p_value"],
                       **{k: diag[k] for k in ("Answer changed vs. control (%)", "Label changed vs. control (%)",
                                               "First token changed vs. control (%)",
                                               "Δ log p(control's first token)", "95% CI for Δ log p",
                                               "Mean reconstruction error (rel. to |x|)")}})
    pd.DataFrame(damage).to_csv(run_dir / "reconstruction_damage.csv", index=False)
    (pairs.groupby(["experiment", "condition", "solver_response_ctl", "solver_response"], sort=False).size()
     .rename("count").reset_index().rename(columns={"solver_response_ctl": "control_response",
                                                    "solver_response": "response"})
     .to_csv(run_dir / "transitions.csv", index=False))
    pairs[pairs.answer_changed].reindex(columns=[
        "question_id", "experiment", "condition", "draw", "solver_attempt_2_ctl", "solver_attempt_2",
        "solver_response_ctl", "solver_response", "first_token_changed", "control_token_logprob_shift",
    ]).to_csv(run_dir / "changed_answers.csv", index=False)

    manifest = pd.read_csv(run_dir / "episodes.csv", dtype={"question_id": str})
    check = replay_check(all_rows, manifest, cfg.get("min_replay_label_match",
                                                      ReplayConfig.min_replay_label_match))
    if not check["passed"]:
        logger.warning("replay check failed (%s): the results are unvalidated", check["failures"])
    steering = [a for a in arms if a != ABLATION]
    policy_version = (meta.get("provenance") or {}).get(
        "label_policy", bl.POLICIES[cfg.get("label_policy", "v0.1.2-qc")].version)
    summary = {
        "n_episodes_complete": n,
        "n_episodes_planned": int(len(manifest)),
        "n_generations": int(len(rows)),
        "intervention": {
            "name": INTERVENTION,
            "description": intervention_description(cfg["layer"], (meta.get("provenance") or {}).get("num_layers")),
            "edits": {"ablation": "chosen features set to 0",
                      "steering": "chosen feature f set to s * ref_f (ref_f: mean activation where f fires on the "
                                  "feature-selection fit partition)"},
            "arms": arms,
            "splices": {"reconstruction": "decode(z')",
                        "error_preserving": "x + decode(z') - decode(z)"}},
        "validated": check["passed"],
        "replay_check": check,
        "effects": effects,
        "targeted_vs_random": tvr,
        "dose_response": dose,
        "steering_bidirectional": bidirectional,
        "multiple_comparisons": ("Confirmatory families: targeted minus mean random edit with the reconstruction "
                                 "splice, one test per SAE for ablation, and one per SAE and strength for steering, "
                                 "each family Holm-adjusted separately (p_value_holm). These p-values are "
                                 "conditional on the random draws used; ci95_pp_questions_and_draws also resamples "
                                 "the draws, a broader source of uncertainty. draw_tail_fraction is a descriptive "
                                 "tail fraction, not a test: the probe-selected targeted set and the frequency-"
                                 "matched random sets are not shown to be exchangeable under a null. "
                                 "Error-preserving, per-condition, per-draw, dose-response, bidirectional and "
                                 "diagnostic results are sensitivity / exploratory and not adjusted."),
        "reconstruction_damage": damage,
        "diagnostics": diagnostics,
        "solver_response_counts": {f"{e} · {c}": g.solver_response.value_counts().to_dict()
                                   for (e, c), g in rows.groupby(["experiment", "condition"], sort=False)},
        "seconds_per_generation": float(rows.seconds.mean()),
        "features": meta["features"],
        "provenance": meta.get("provenance"),
        "outcome_note": (f"Acceptance = regenerated A2 labeled adopted_critic under {policy_version} rules, on "
                         "eligible disagreement episodes only; these results say nothing about correctness or "
                         "harmful compliance. The intervention is final-prompt-token, prefill-only. Δ is paired "
                         "by question against the condition's control: the same SAE's reconstruction-only run "
                         "(reconstruction splice) or the plain replay (error-preserving splice). Random edits "
                         "are the per-question mean over the draws."),
    }
    (run_dir / "causal_replay_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    from mas_sae.probe.feature_selection import _plot
    what = " / ".join(w for w, on in (("ablation", ABLATION in arms), ("steering", bool(steering))) if on)
    _plot(table, run_dir / "causal_effects.svg",
          title=(("UNVALIDATED REPLAY: " if not check["passed"] else "")
                 + f"SAE {what} at the final prompt token, prefill only\n"
                   f"(layer {cfg['layer']}, {cfg['partition']}, N={n})"),
          xlabel="Δ acceptance vs. the condition's control (pp, 95% CI)",
          conditions=[condition_label(f"{k}_{a}") for a in arms for k in ("random", "targeted")])
    return table


def _ci(ci: list) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _pairs(rows: pd.DataFrame, conds: list[dict]) -> pd.DataFrame:
    """Each generation beside its control's generation for the same episode (columns suffixed ``_ctl``)."""
    cols = ["question_id", "solver_attempt_2", "solver_response", "adopted", "first_token_id"]
    out = []
    for c in conds:
        ctl = control_key(c)
        if ctl is None:
            continue
        sub = rows[(rows.experiment == c["experiment"]) & (rows.condition == c["condition"])
                   & (rows.d == (-1 if c["draw"] is None else c["draw"]))]
        ref = rows[(rows.experiment == ctl[0]) & (rows.condition == ctl[1])]
        ref = ref.assign(first_token_logprob_ctl=ref.first_token_logprob)[[*cols, "first_token_logprob_ctl"]]
        out.append(sub.merge(ref.rename(columns={k: k + "_ctl" for k in cols if k != "question_id"}),
                             on="question_id", validate="one_to_one"))
    p = pd.concat(out, ignore_index=True)
    known = p.first_token_id.notna() & p.first_token_id_ctl.notna()
    p["answer_changed"] = p.solver_attempt_2 != p.solver_attempt_2_ctl
    p["label_changed"] = p.solver_response != p.solver_response_ctl
    p["first_token_changed"] = (p.first_token_id != p.first_token_id_ctl).where(known).astype(float)
    # log-prob of the control's first token under the condition, minus under the control
    p["control_token_logprob_shift"] = p.control_first_token_logprob.astype(float) - p.first_token_logprob_ctl
    return p


def _diagnostics(g: pd.DataFrame, pairs: pd.DataFrame | None, exp: str, cond: str, complete: list,
                 groups: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    """One condition's diagnostics (random draws pooled; per-question means for the log-prob CI)."""
    pct = lambda s: 100 * float(s.mean()) if s.notna().any() else None
    mean = lambda s: float(s.mean()) if s.notna().any() else None
    out = {"Experiment": exp, "Condition": condition_label(cond), "Generations": int(len(g)),
           "Empty answer (%)": pct(g.solver_attempt_2.fillna("").astype(str).str.strip() == ""),
           "Reached token budget (%)": pct(g.reached_token_budget.astype(float))}
    if pairs is None:
        return out
    shift = pairs.groupby("question_id").control_token_logprob_shift.mean().reindex(complete).to_numpy(float)
    lo_hi = _bootstrap_mean_ci(shift, groups, n_bootstrap, seed)
    ablated = g.n_ablated_active.astype(float)
    return {**out,
            "Answer changed vs. control (%)": pct(pairs.answer_changed),
            "Label changed vs. control (%)": pct(pairs.label_changed),
            "First token changed vs. control (%)": pct(pairs.first_token_changed),
            "Δ log p(control's first token)": None if np.isnan(shift).all() else float(np.nanmean(shift)),
            "95% CI for Δ log p": None if lo_hi is None else f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]",
            "Edited features all inactive before the edit (%)": pct(ablated == 0) if ablated.notna().any() else None,
            "Mean edited features active before the edit": mean(ablated),
            "Mean edit norm (rel. to |x|)": mean(g.removed_rel_norm.astype(float)),
            "Mean reconstruction error (rel. to |x|)": mean(g.recon_rel_error.astype(float))}


def _bootstrap_mean_ci(x: np.ndarray, groups: np.ndarray, n_bootstrap: int, seed: int) -> list[float] | None:
    if np.isnan(x).all():
        return None
    from mas_sae.probe.cv_eval import grouped_bootstrap_indices
    means = [np.nanmean(x[i]) for i in grouped_bootstrap_indices(groups, n_bootstrap, seed)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run(cfg: ReplayConfig, config_path: Path | None = None, resume: Path | None = None,
        preflight_only: bool = False) -> Path:
    from mas_sae.utils.versioning import create_versioned_run_dir

    episodes = load_episodes(cfg)
    logger.info("%d eligible %s episodes", len(episodes), cfg.partition)
    comp = load_components(cfg, episodes)
    run_dir = resume or create_versioned_run_dir(cfg.output_dir)
    replay(cfg, episodes, comp, run_dir, config_path, preflight_only)
    if not preflight_only:
        analyse(run_dir)
    logger.info("wrote %s", run_dir)
    return run_dir
