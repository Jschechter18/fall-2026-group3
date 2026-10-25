"""Expand compatible parameter groups and compare validation-only SAE trials."""

import csv
from dataclasses import asdict, fields, replace
from itertools import product
import math
from pathlib import Path

from mas_sae.experiments.artifacts import (
    create_sae_run_directory, get_git_provenance, update_run_manifest,
    write_json_atomic, write_run_config, write_sae_provenance,
)
from mas_sae.sae.hyperparamters import Hyperparameters, validate_hyperparameters
from mas_sae.sae.training import train_sae


def _mapping(value, allowed, label):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys.")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{label}: unknown fields {sorted(unknown)}.")
    return value


def expand_search_space(config: dict) -> list[tuple[str, Hyperparameters]]:
    """Validate the complete study before returning its resolved candidates."""
    _mapping(config, {"activation_run_name", "layer", "defaults", "groups", "selection"}, "config")
    source = config.get("activation_run_name")
    if not isinstance(source, str) or not source or source in {".", ".."} or Path(source).name != source:
        raise ValueError("activation_run_name must be a single directory name.")
    if type(config.get("layer")) is not int or config["layer"] < 0:
        raise ValueError("layer must be a nonnegative integer.")
    selection = _mapping(config.get("selection", {}), {"max_mean_active_features"}, "selection")
    if selection:
        budget = selection["max_mean_active_features"]
        if type(budget) not in (int, float) or not math.isfinite(budget) or budget < 0:
            raise ValueError("max_mean_active_features must be finite and nonnegative.")

    allowed = {field.name for field in fields(Hyperparameters)} - {"input_dim", "hidden_dim"}
    defaults = _mapping(config.get("defaults", {}), allowed, "defaults")
    base = Hyperparameters(**defaults)
    groups = config.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list.")

    candidates, names, seen = [], set(), set()
    for group in groups:
        _mapping(group, {"name", "fixed", "grid", "variants"}, "group")
        name = group.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Every group needs a unique nonempty name.")
        names.add(name)
        fixed = _mapping(group.get("fixed", {}), allowed, f"{name}.fixed")
        grid = _mapping(group.get("grid", {}), allowed, f"{name}.grid")
        if any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError(f"{name}: every grid field needs a nonempty list.")
        variants = group.get("variants", [{}])
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"{name}: variants must be a nonempty list.")
        for variant in variants:
            _mapping(variant, allowed, f"{name}.variant")
            if (fixed.keys() & grid.keys() or fixed.keys() & variant.keys()
                    or grid.keys() & variant.keys()):
                raise ValueError(f"{name}: fixed, grid and variant fields must not overlap.")
            for values in product(*grid.values()):
                overrides = {**fixed, **dict(zip(grid, values)), **variant}
                hp = replace(base, **overrides)
                validate_hyperparameters(hp)
                irrelevant = "top_k" if hp.sparsity_mode == "l1" else "sparsity_coefficient"
                if irrelevant in (defaults.keys() | overrides.keys()):
                    raise ValueError(f"{name}: {irrelevant} does not apply to {hp.sparsity_mode}.")
                # Canonicalize unused defaults for clear comparisons and duplicate detection.
                if hp.sparsity_mode == "l1":
                    hp.top_k = None
                else:
                    hp.sparsity_coefficient = 0.0
                identity = tuple(asdict(hp).items())
                if identity in seen:
                    raise ValueError(f"{name}: duplicate resolved configuration.")
                seen.add(identity)
                candidates.append((name, hp))
    return candidates


def compare_trials(trials, selection):
    """Annotate a Pareto frontier and, if requested, choose a budget-feasible winner."""
    valid = [trial for trial in trials if trial["status"] == "completed" and all(
        type(trial["validation"].get(key)) in (int, float)
        and math.isfinite(trial["validation"][key])
        for key in ("rec_loss", "mean_active_features")
    )]
    budget = selection.get("max_mean_active_features")
    for trial in trials:
        trial["pareto"] = False
        trial["eligible"] = False
    for trial in valid:
        rec = trial["validation"]["rec_loss"]
        activity = trial["validation"]["mean_active_features"]
        trial["eligible"] = budget is not None and activity <= budget
        trial["pareto"] = not any(
            other["validation"]["rec_loss"] <= rec
            and other["validation"]["mean_active_features"] <= activity
            and (other["validation"]["rec_loss"] < rec
                 or other["validation"]["mean_active_features"] < activity)
            for other in valid
        )
    return min((trial for trial in valid if trial["eligible"]),
               key=lambda trial: trial["validation"]["rec_loss"], default=None)


def _write_trials(study_directory, trials):
    write_json_atomic(study_directory / "trials.json", {"trials": trials})
    # Flat columns make group/mode/architecture comparisons easy in a notebook.
    rows = [{
        **{key: value for key, value in trial.items() if key not in {"hyperparameters", "validation"}},
        **trial["hyperparameters"],
        **{f"val_{key}": value for key, value in trial.get("validation", {}).items()},
    } for trial in trials]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = study_directory / "trials.csv.tmp"
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(study_directory / "trials.csv")


def run_tuning(config: dict, project_root: Path) -> Path:
    """Run a study sequentially and return its directory, not a finalized model."""
    candidates = expand_search_space(config)
    activation_location = (project_root / "data" / "activations"
                           / config["activation_run_name"] / f"layer_{config['layer']:02d}")
    activation_files = {split: activation_location / f"{split}.pt"
                        for split in ("train", "validation")}
    for path in activation_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    results_root = project_root / "results" / "sae" / "musique"
    provenance = get_git_provenance()
    study = create_sae_run_directory(
        run_name=config["activation_run_name"], layer=config["layer"],
        results_root=results_root, category="tuning",
    )
    trials = []
    try:
        write_run_config(study, config)
        write_sae_provenance(study, activation_files, provenance)
        for number, (group, hp) in enumerate(candidates, start=1):
            trials.append({
                "trial_id": number, "study_id": study.name, "group": group,
                "activation_run_name": config["activation_run_name"], "layer": config["layer"],
                "hyperparameters": asdict(hp), "status": "pending",
            })
        _write_trials(study, trials)
        for trial, (_, hp) in zip(trials, candidates):
            trial["status"] = "running"
            _write_trials(study, trials)
            print(f"Trial {trial['trial_id']}/{len(trials)}: {trial['group']}")
            try:
                result = train_sae(
                    hp=hp, activation_location=activation_location,
                    activation_run_name=config["activation_run_name"], layer=config["layer"],
                    results_root=results_root, project_root=project_root,
                    evaluate_test=False, register_version=False,
                    output_directory=study / "trials" / f"trial_{trial['trial_id']:04d}",
                    run_context={"study_directory": str(study.relative_to(project_root)),
                                 "trial_id": trial["trial_id"], "group": trial["group"]},
                )
            except (Exception, KeyboardInterrupt) as error:
                trial.update(status="failed", error=f"{type(error).__name__}: {error}")
                _write_trials(study, trials)
                raise
            trial.update(
                status="completed", best_epoch=result.best_epoch,
                run_directory=str(result.run_directory.relative_to(project_root)),
                checkpoint_path=str(result.checkpoint_path.relative_to(project_root)),
                validation=result.metrics["validation"],
            )
            compare_trials(trials, config.get("selection", {}))
            _write_trials(study, trials)
        winner = compare_trials(trials, config.get("selection", {}))
        write_json_atomic(study / "selection.json", {
            "policy": config.get("selection", {}),
            "status": ("selected" if winner else "no_eligible_trial")
                      if config.get("selection") else "review_pareto_frontier",
            "trial_id": winner["trial_id"] if winner else None,
            "run_directory": winner["run_directory"] if winner else None,
            "checkpoint_path": winner["checkpoint_path"] if winner else None,
        })
    except (Exception, KeyboardInterrupt) as error:
        update_run_manifest(study, status="failed", error_message=f"{type(error).__name__}: {error}")
        raise
    update_run_manifest(study, status="completed")
    return study
