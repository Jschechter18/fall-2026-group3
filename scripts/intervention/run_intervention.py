"""
THE INTERVENTION EXPERIMENT
===========================

QUESTION
--------
Are an SAE's adoption and retention features causal for whether the Solver adopts the Critic's
answer? We steer the Solver along those features while it writes its second answer (A2), and ask
whether the stored label flips more often than when we steer along matched random features.


EPISODES
--------
Stored intervention episodes whose stored Claude label is adoption or retention (the stratum),
one per question, Natural Critic, candidates presentation. Sorted by question id and shuffled with
seed 42: the study is positions 1..1000, choose-r positions 1001..1050. Never the same episode.


WHERE THE EDIT HAPPENS
----------------------
Solver layer 25, at the final prompt token of the A2 prefill, and nowhere else: earlier positions
and decode steps are not edited. Below the last layer the edited token's keys and values still reach
every decode step, so the answer can change even when the first token does not. A null result means
"these features don't matter at this position", not "they don't matter".


THE EDIT (mas_sae.intervention.generation.edits)
-------------------------------------
x is the layer-25 activation at that token. Every edit is x' = x + Δ with |Δ| = r·|x|, one size r for
every arm and every feature set, chosen by choose-r. So a difference between targeted and random,
or between arms, comes from direction alone.

    d_i      direction of feature i: its unit decoder row (SAE), or e_i (raw neuron), signed so that
             +d_i means "more of the set's label". Random raw neurons use +e_i.
    u_S      = normalize(Σ_{i∈S} d_i), one direction per feature set, computed once per run

    stratum                 arm         Δ
    adoption → retention    decrease    −r·|x|·u_adoption
                            increase    +r·|x|·u_retention
                            both        r·|x|·normalize(u_retention − u_adoption)
    retention → adoption    decrease    −r·|x|·u_retention
                            increase    +r·|x|·u_adoption
                            both        r·|x|·normalize(u_adoption − u_retention)

"Decrease" is a steering edit, not an ablation, so it is never a no-op. "Both" is one edit of size r,
not two. A random draw uses one pool index for both labels, so "both" pairs adoption draw i with
retention draw i.


CONDITIONS PER EPISODE (65 generations)
---------------------------------------
      1   baseline: no SAE, no edit (the plain replay)
    + 4   reconstruction splice, x' = decode(encode(x)), one per SAE (no edit)
    + 60  5 representations × 3 arms × (targeted + 3 random draws)

    representations: Raw activations (control), L1 SAE, Pretrained SAE, Fine-tuned decoder,
                     Fully fine-tuned † (exploratory: its own feature sets)
    feature sets:    feature_sets.json from the feature-selection rerun (read-only here)
    random draws:    each question gets its own 3 distinct draws from the pool of the first
                     random_draw_pool_size (100) matched draws, fixed by the seed and the question id
                     and frozen by prepare, so across questions the random baseline samples the pool


OUTCOME
-------
"Flip": Claude labels the regenerated A2 as the opposite of the episode's stored label. Agreement and
out count as not flipped (and are reported). Labels come from the exact labeler of the stored labels
(Claude Sonnet 4.5 on Bedrock, prompt claude_binary_v3_candidates), one A2 per request, blind to the
condition. A labeling failure is never retention: retried, then its episode is dropped everywhere.


ANALYSIS PLAN (fixed before any outcome is seen; also in the config's `analysis` section)
-----------------------------------------------------------------------------------------
Every comparison: Δ = targeted flip − mean flip over the question's 3 random draws, paired by
question; 95% bootstrap CI over questions (each carries its own random draws, so this also resamples
them); permutation p (swap targeted and random within each question).

PRIMARY (Table P): "some SAE's features are causal"
    L1 SAE, Pretrained SAE, Fine-tuned decoder × decrease, increase, both = 9 tests, strata pooled
    (an arm's two strata are mirror images), one-sided (targeted > random), Holm across the 9.
    The claim holds if at least one test survives. Not in this family: Raw activations (control) and
    Fully fine-tuned (exploratory). Which SAE and direction the write-up highlights may be chosen
    after the run; this family may not.

SECONDARY (each its own Holm family, two-sided)
    Table 1   per representation × stratum × arm; Holm within each representation (6 rows)
    Table 2   between-SAE contrasts Δ_A − Δ_B, strata pooled, one row per arm:
              pre-specified: L1 vs Pretrained, Fine-tuned decoder vs Pretrained, L1 vs Fine-tuned decoder
              exploratory:   Fully fine-tuned † vs Pretrained
    Table D   which direction works best, within each representation: both − decrease,
              both − increase, increase − decrease; Holm across the 3. One arm significant and
              another not is not evidence that they differ: only these contrasts are.
    Reference rows (baseline and reconstruction flips) and Table 3 (diagnostics): descriptive.


HOW TO RUN
----------
    python scripts/intervention/run_intervention.py choose-r --config configs/intervention/intervention.yaml
    python scripts/intervention/run_intervention.py prepare --config configs/intervention/intervention.yaml
    python scripts/intervention/run_intervention.py run     --config configs/intervention/intervention.yaml --shard K
    python scripts/intervention/run_intervention.py merge   --config configs/intervention/intervention.yaml

Every run gets its own folder, <output_root>/<run_id>, named <UTC time>-<12-character git commit> like
the repo's SAE runs, so a new run never overwrites an old one. choose-r with run_id: null creates it and
prints its name; set run_id to that name in the config (as you set edit_size) before any other stage
or rerun. Every other stage refuses to start without an existing run folder.

Every stage first runs the preflight, which lists everything the machine is missing in one error
before any GPU work; `--check-only` runs it and exits. Every stage can be killed and rerun with the
same command: finished generations and Claude responses are never redone, and a worker refuses to
continue a run frozen with different code, inputs or packages. The GPU model is recorded, not
enforced; a worker whose baseline of an episode differs from prepare's excludes that episode, and
merge refuses if more than (1 − min_baseline_label_match) of the episodes were excluded that way.

    choose-r choose r (and, with run_id: null, create the run): the choose-r episodes, the 3 primary SAEs, each size in candidate_edit_sizes.
             Writes choose_r/choose_r_report.md. Then stop, choose r, and set experiment.edit_size.
    prepare  freeze the run, generate and check the baseline of every study episode, write shards.
    run      one GPU worker, one shard. Needs frozen_run.json, shards.json, baseline.jsonl.
    merge    check every shard is complete, label every answer, write the tables and summary.json.

Speed: within an episode, layers 0–25 of the prefill are computed once and replayed in every
other condition (the edit is at layer 25), and x is encoded once per SAE encoder. Conditions are
generated one at a time (not batched), so each row is exactly the plain generation's.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from mas_sae.intervention.run_files import layout
from mas_sae.intervention.analysis import tables
from mas_sae.intervention.checks.baseline import check_baseline_reproduces_stored
from mas_sae.intervention.experiment.conditions import BASELINE, choose_r_conditions, random_draws_for, study_conditions
from mas_sae.intervention.experiment.config import load_config
from mas_sae.intervention.experiment.episodes import study_and_choose_r_episodes
from mas_sae.intervention.generation.episode import BaselineMismatch, generate_episode, make_row
from mas_sae.intervention.analysis.labeling import StubClaudeClient, create_client, label_rows, read_protocol
from mas_sae.intervention.analysis.choose_r import write_choose_r_report
from mas_sae.intervention.checks.preflight import run_preflight
from mas_sae.intervention.run_files.provenance import (
    device_name, frozen_content, provenance, require_same_experiment, uncommitted_code_sha256, write_frozen_run,
)
from mas_sae.intervention.experiment.representations import frozen_feature_sets, load_representations
from mas_sae.intervention.run_files.results_file import ResultsFile, read_rows
from mas_sae.intervention.run_files.shards import (
    mark_shard_complete, read_complete_shards, read_excluded, record_excluded, shard_rows_path, split_into_shards,
)
from mas_sae.intervention.generation.solver import generate_baseline, load_solver
from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file

if TYPE_CHECKING:
    from mas_sae.intervention.experiment.conditions import Condition
    from mas_sae.intervention.experiment.config import InterventionConfig
    from mas_sae.intervention.experiment.episodes import Episode
    from mas_sae.intervention.experiment.representations import Representation
    from mas_sae.intervention.generation.solver import SolverModel

logger = logging.getLogger("intervention")


# =====================================================================
# CHOOSE-R: CHOOSE THE EDIT SIZE r
# =====================================================================

def start_run(config: InterventionConfig) -> InterventionConfig:
    """Create a new run folder <output_root>/<run_id> and record where it came from. Never reuses one."""
    run_id = layout.new_run_id()
    run_dir = Path(config.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        uncommitted = uncommitted_code_sha256()
    except Exception:  # noqa: BLE001 - not a git checkout; recorded, not enforced (prepare enforces code identity)
        uncommitted = "unknown"
    write_json(layout.run_record(run_dir), {
        "run_id": run_id, "created_utc": run_id.split("-")[0], "git_commit": get_git_commit(),
        "uncommitted_code_sha256": uncommitted, "command": get_run_command()})
    return replace(config, run_id=run_id)


def choose_r(config: InterventionConfig, labels_stubbed: bool) -> None:
    """Choose-r stage: generate the choose-r episodes at every candidate edit size for the primary
    SAEs, label them, and write the choose-r report. Refuses a rerun with other episodes or conditions."""
    run_dir = config.run_dir
    e = config.experiment
    _, choose_r_episodes = study_and_choose_r_episodes(config)

    solver_model = load_solver(config)
    representations = on_device(load_representations(config), solver_model, roles=("primary",))
    conditions = choose_r_conditions(list(representations.values()), e.candidate_edit_sizes, e.number_of_random_draws)
    record_choose_r_run(run_dir, choose_r_episodes, conditions)

    results = ResultsFile(layout.choose_r_rows(run_dir))
    draws = random_draws(config, choose_r_episodes)
    for episode in tqdm(choose_r_episodes, desc="choose-r episodes"):
        generate_episode(solver_model, episode, conditions, representations, results, draws[episode.episode_id])

    labeled, excluded = label(results.rows, choose_r_episodes, config, layout.choose_r_dir(run_dir), labels_stubbed)
    write_choose_r_report(labeled, excluded, list(representations), e.candidate_edit_sizes, e.number_of_random_draws,
                       config.analysis.n_bootstrap, layout.choose_r_report_csv(run_dir),
                       layout.choose_r_report_md(run_dir), labels_stubbed)
    print(f"Choose-r report: {layout.choose_r_report_md(run_dir)}\n"
          f"Choose r, write it into the config as experiment.edit_size (with run_id: {config.run_id}), "
          "then run prepare.")


# =====================================================================
# PREPARE: FREEZE THE RUN, CHECK THE BASELINE, WRITE THE SHARDS
# =====================================================================

def prepare(config: InterventionConfig) -> None:
    """Prepare stage: refuse if a choose-r episode is in the study, freeze ``frozen_run.json`` (a rerun
    must freeze the same content), generate every study episode's baseline and check it against the
    stored A2s, then write ``shards.json``. Raises RuntimeError, before any edit, if the check fails."""
    run_dir = config.run_dir
    study_episodes, choose_r_episodes = study_and_choose_r_episodes(config)
    require_no_choose_r_episode_in_study(run_dir, study_episodes)

    representations = load_representations(config)
    conditions = study_conditions(representations, config.experiment.edit_size,
                                  config.experiment.number_of_random_draws)
    solver_model = load_solver(config)

    # Freeze everything a worker must reproduce. A rerun of prepare must freeze the same run.
    content = frozen_content(config, ids(study_episodes), ids(choose_r_episodes),
                             frozen_feature_sets(representations), [c.key for c in conditions],
                             random_draws(config, study_episodes + choose_r_episodes))
    frozen_sha = write_frozen_run(layout.frozen_run(run_dir), content,
                                  provenance(config, device_name(solver_model)),
                                  recorded={"edit_size": config.experiment.edit_size,
                                            "choose_r_report_sha256": sha256_file(layout.choose_r_report_csv(run_dir)),
                                            "set_cosines": {r.name: r.set_cosine() for r in representations}})

    # Baseline: does the plain replay reproduce the stored A2s? If not, stop.
    baseline = ResultsFile(layout.baseline_rows(run_dir))
    for episode in tqdm(study_episodes, desc="baseline"):
        if not baseline.has(episode.episode_id, BASELINE.key):
            generation, _ = generate_baseline(solver_model, episode)
            baseline.append(make_row(episode, BASELINE, generation, generation, None, None))
    check, mismatches = check_baseline_reproduces_stored(baseline.rows, study_episodes,
                                                         config.experiment.min_baseline_label_match)
    write_json(layout.baseline_check(run_dir), check)
    mismatches.to_csv(layout.baseline_mismatches(run_dir), index=False)
    if not check["passed"]:
        raise RuntimeError(f"The baseline does not reproduce the stored answers ({check['failures']}); "
                           f"see {layout.baseline_mismatches(run_dir)}. Stopping before any edit.")

    shards = split_into_shards(ids(study_episodes), config.experiment.number_of_shards)
    write_json(layout.shards(run_dir), {"frozen_run_sha256": frozen_sha, "shards": shards})
    print(f"Baseline check passed (label match {check['label_match_rate']:.3f}). "
          f"{len(shards)} shards in {layout.shards(run_dir)}.")


# =====================================================================
# RUN: ONE SHARD OF EPISODES ON ONE GPU WORKER
# =====================================================================

def run(config: InterventionConfig, shard: int) -> None:
    """Run stage, one GPU worker: generate every condition of the shard's episodes into
    ``shard_K.jsonl`` and write its completion marker. Refuses if ``frozen_run.json`` changed after
    sharding, the baseline check failed, or this machine's experiment, code, packages or inputs differ
    from the frozen run. Episodes whose regenerated baseline differs from prepare's are excluded."""
    run_dir = config.run_dir
    frozen, shards = read_json(layout.frozen_run(run_dir)), read_json(layout.shards(run_dir))
    if sha256_file(layout.frozen_run(run_dir)) != shards["frozen_run_sha256"]:
        raise RuntimeError("frozen_run.json changed after the shards were written.")
    if not read_json(layout.baseline_check(run_dir))["passed"]:
        raise RuntimeError("The baseline check did not pass; no edit is run.")

    study_episodes, choose_r_episodes = study_and_choose_r_episodes(config)
    representations = load_representations(config) 
    conditions = study_conditions(representations, config.experiment.edit_size,
                                  config.experiment.number_of_random_draws)
    solver_model = load_solver(config)

    # This worker must be running the frozen experiment, with the same code, packages and inputs.
    # The GPU model is recorded, not enforced: each episode's baseline check below is what guarantees it.
    content = frozen_content(config, ids(study_episodes), ids(choose_r_episodes),
                             frozen_feature_sets(representations), [c.key for c in conditions],
                             random_draws(config, study_episodes + choose_r_episodes))
    require_same_experiment(frozen, content, provenance(config, device_name(solver_model)),
                            where=str(layout.frozen_run(run_dir)))

    representations = on_device(representations, solver_model)
    by_id = {episode.episode_id: episode for episode in study_episodes}
    shard_episodes = [by_id[episode_id] for episode_id in shards["shards"][shard]]
    prepared_baseline = {r["episode_id"]: r for r in read_rows(layout.baseline_rows(run_dir))}

    # Each episode's baseline is regenerated first and must equal prepare's, before any of its edits.
    # An episode whose baseline differs is excluded (recorded, never retried) and the shard goes on.
    results = ResultsFile(shard_rows_path(run_dir, shard))
    excluded = read_excluded(run_dir, shard)
    for episode in tqdm(shard_episodes, desc=f"shard {shard}"):
        if episode.episode_id in excluded:
            continue
        try:
            generate_episode(solver_model, episode, conditions, representations, results,
                             frozen["random_draws"][episode.episode_id],
                             prepared_baseline=prepared_baseline[episode.episode_id])
        except BaselineMismatch as mismatch:
            logger.warning("Excluding %s", mismatch)
            excluded = record_excluded(run_dir, shard, episode.episode_id, str(mismatch))

    marker = mark_shard_complete(run_dir, shard, shards["shards"][shard], [c.key for c in conditions],
                                 shards["frozen_run_sha256"], excluded)
    seconds = [r["seconds"] for r in results.rows]
    per_generation = sum(seconds) / len(seconds) if seconds else float("nan")
    print(f"Shard {shard} complete: {marker['row_count']} rows, {len(excluded)} of {len(shard_episodes)} episodes "
          f"excluded (baseline differs from prepare's), {per_generation:.2f} s per generation "
          f"(~{per_generation * len(conditions) * 1000 / 3600:.1f} GPU-hours per 1000 episodes).")


# =====================================================================
# MERGE: COMBINE THE SHARDS, LABEL EVERY ANSWER, COMPUTE THE TABLES
# =====================================================================

def merge(config: InterventionConfig, labels_stubbed: bool) -> None:
    """Merge stage: check the config still defines the frozen experiment and every shard is complete
    and made for this frozen run, label every row (a stub for smoke runs), and write
    ``labeled_rows.jsonl``, the tables and ``summary.json`` to ``merge/``. Raises RuntimeError if
    more episodes were excluded for a baseline mismatch than (1 − min_baseline_label_match) of the
    study allows."""
    run_dir, out = config.run_dir, layout.merge_dir(config.run_dir)
    frozen, shards = read_json(layout.frozen_run(run_dir)), read_json(layout.shards(run_dir))
    study_episodes, choose_r_episodes = study_and_choose_r_episodes(config)
    representations = load_representations(config)
    conditions = study_conditions(representations, config.experiment.edit_size,
                                  config.experiment.number_of_random_draws)
    content = frozen_content(config, ids(study_episodes), ids(choose_r_episodes),
                             frozen_feature_sets(representations), [c.key for c in conditions],
                             random_draws(config, study_episodes + choose_r_episodes))
    require_same_experiment(frozen, content, None, where=str(layout.frozen_run(run_dir)))

    rows, mismatched = read_complete_shards(run_dir, shards["shards"], frozen["conditions"],
                                            shards["frozen_run_sha256"])
    # A worker that could not reproduce prepare's baseline for more episodes than prepare itself may
    # miss collection's is not a few close calls but a different machine: refuse to analyse it.
    allowed = (1 - config.experiment.min_baseline_label_match) * len(study_episodes)
    if len(mismatched) > allowed:
        raise RuntimeError(f"{len(mismatched)} episodes were excluded because a worker's baseline differed from "
                           f"prepare's (at most {allowed:.0f} allowed); see shard_*.excluded.json.")
    labeled, label_failed = label(rows, study_episodes, config, out, labels_stubbed)
    excluded = {**{e: f"baseline mismatch: {why}" for e, why in mismatched.items()}, **label_failed}
    out.mkdir(parents=True, exist_ok=True)
    (out / "labeled_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labeled))

    summary = write_tables(labeled, excluded, config, frozen, read_json(layout.baseline_check(run_dir)),
                           out, labels_stubbed)
    p = summary["primary"]
    print(f"Primary claim ({p['claim']}): {'HOLDS' if p['claim_holds'] else 'does not hold'}. "
          f"{len(excluded)} episodes excluded. Tables and summary.json in {out}.")


def write_tables(labeled: list[dict], excluded: dict[str, str], config: InterventionConfig, frozen: dict,
                 baseline_check: dict, out: Path, labels_stubbed: bool) -> dict:
    """Write each table (Table P, 1, 2, D, 3, reference rows, label counts) as CSV and markdown to
    ``out``, and return the summary also written to ``summary.json``."""
    e, plan = config.experiment, config.analysis
    roles = {name: spec.role for name, spec in config.representations.items()}
    saes = [name for name, role in roles.items() if role != "control"]
    outcomes = tables.Outcomes(labeled, e.edit_size, e.number_of_random_draws, plan)
    cosines = frozen["recorded"]["set_cosines"]
    results = {
        "table_P_primary": tables.table_p(outcomes),
        "table_1": tables.table_1(outcomes, roles),
        "reference_rows": tables.reference_rows(outcomes, saes),
        "table_2_prespecified": tables.table_2(outcomes, plan.between_sae_contrasts, "pre-specified"),
        "table_2_exploratory": tables.table_2(outcomes, plan.exploratory_contrasts, "exploratory †"),
        "table_D_directions": tables.table_d(outcomes, list(roles)),
        "table_3_diagnostics": tables.table_3(labeled, cosines),
        "label_counts": tables.label_counts(labeled),
    }
    for name, df in results.items():
        df.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(tables.to_markdown(df))
    records = {name: df.to_dict("records") for name, df in results.items()}
    summary = {
        **({"WARNING": "LABELS STUBBED: smoke run, numbers are meaningless"} if labels_stubbed else {}),
        "primary": {"claim": "some pre-specified SAE's features are causal",
                    "claim_holds": bool(results["table_P_primary"]["Survives Holm (α = 0.05)"].any()),
                    "family": "3 SAEs × 3 arms, strata pooled, one-sided, Holm", "table_P": records["table_P_primary"]},
        "secondary": {k: records[k] for k in ("table_1", "table_2_prespecified", "table_D_directions",
                                              "reference_rows", "table_3_diagnostics", "label_counts")},
        "exploratory": {"table_2_exploratory": records["table_2_exploratory"],
                        "note": "Fully fine-tuned † has its own feature sets; its Table 1 rows are exploratory."},
        "n_episodes_analysed": len(outcomes.episodes()), "excluded_episodes": excluded,
        "baseline_check": baseline_check, "edit_size": e.edit_size, "analysis_plan": config.as_dict()["analysis"],
        "labels_stubbed": labels_stubbed,
    }
    write_json(out / "summary.json", summary)
    return summary


# =====================================================================
# SMALL HELPERS
# =====================================================================

def ids(episodes: list[Episode]) -> list[str]:
    """The episodes' ids, in order."""
    return [episode.episode_id for episode in episodes]


def random_draws(config: InterventionConfig, episodes: list[Episode]) -> dict[str, list[int]]:
    """{episode id: its pool indices, one per random draw slot}, fixed by the seed and the question id."""
    e = config.experiment
    return {episode.episode_id: random_draws_for(episode.question_id, e.seed, e.random_draw_pool_size,
                                                 e.number_of_random_draws) for episode in episodes}


def on_device(representations: list[Representation], solver_model: SolverModel,
              roles: tuple[str, ...] | None = None) -> dict:
    """{name: representation} on the Solver's device, optionally only those with the given roles."""
    return {r.name: r.to(solver_model.model.device) for r in representations if roles is None or r.role in roles}


def label(rows: list[dict], episodes: list[Episode], config: InterventionConfig, work_dir: Path,
          labels_stubbed: bool) -> tuple[list[dict], dict]:
    """Label every row with the stored labels' Claude labeler (or, for smoke runs only, a stub)."""
    protocol = read_protocol(Path(config.inputs.stored_labels))
    if labels_stubbed:
        client_factory = lambda: StubClaudeClient(protocol.models_returned[0])
    else:
        client_factory = lambda: create_client(protocol, config.labeling)
    work_dir = Path(work_dir) / ("labeling_stubbed" if labels_stubbed else "labeling")
    return label_rows(rows, {episode.episode_id: episode for episode in episodes}, protocol, work_dir,
                      client_factory, config.labeling.concurrency)


def record_choose_r_run(run_dir: Path, episodes: list[Episode], conditions: list[Condition]) -> None:
    """Write choose-r's episodes and conditions; a rerun must be the same choose-r run."""
    record = {"episode_ids": ids(episodes), "conditions": [c.key for c in conditions]}
    path = layout.choose_r_episodes(run_dir)
    if path.exists() and read_json(path) != record:
        raise RuntimeError(f"{path} records a different choose-r run: set run_id: null "
                           "and rerun choose-r to start a new run.")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)


def require_no_choose_r_episode_in_study(run_dir: Path, study_episodes: list[Episode]) -> None:
    """Raise RuntimeError if a study episode is one of the choose-r episodes the run recorded."""
    overlap = set(read_json(layout.choose_r_episodes(run_dir))["episode_ids"]) & set(ids(study_episodes))
    if overlap:
        raise RuntimeError(f"{len(overlap)} choose-r episodes are in the study cohort, e.g. {sorted(overlap)[:3]}.")


def read_json(path: Path) -> dict:
    """Parse the JSON file at ``path``."""
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: dict) -> None:
    """Write ``value`` to ``path`` as indented JSON, with non-JSON values as strings."""
    Path(path).write_text(json.dumps(value, indent=2, default=str))


# =====================================================================
# RUN THE REQUESTED STAGE
# =====================================================================

def main() -> None:
    """Parse the command line, run the preflight (``--check-only`` stops there), create the run for
    choose-r with run_id: null, and run the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["choose-r", "prepare", "run", "merge"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, help="which shard to run (run only)")
    parser.add_argument("--check-only", action="store_true", help="run the preflight and exit")
    parser.add_argument("--stub-labeling", action="store_true",
                        help="SMOKE RUNS ONLY: label with a stub instead of Claude (results are meaningless)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    run_preflight(config, args.stage, args.shard, labels_stubbed=args.stub_labeling)
    if args.check_only:
        print(f"Preflight for {args.stage} passed."
              + (f" A new run will be created in {config.output_root}." if config.run_id is None else ""))
        return
    if args.stage == "choose-r" and config.run_id is None:
        config = start_run(config)
        print(f"New run {config.run_dir}.\nSet run_id: {config.run_id} in {args.config} before any other stage "
              "or rerun (a rerun with run_id: null starts another new run).", flush=True)
    if args.stage == "choose-r":
        choose_r(config, args.stub_labeling)
    elif args.stage == "prepare":
        prepare(config)
    elif args.stage == "run":
        run(config, args.shard)
    else:
        merge(config, args.stub_labeling)


if __name__ == "__main__":
    main()
