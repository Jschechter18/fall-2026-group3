"""The experiment's YAML config, as typed sections. Unknown keys and missing required fields are refused.

The analysis plan lives here as well as in the script's docstring, so it is frozen with the run
(``frozen_run.json``) before any outcome is seen.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

import yaml

ROLES = ("control", "primary", "exploratory")
KINDS = ("activations", "custom", "gemma_scope")
ALTERNATIVES = ("greater", "two-sided")


@dataclass(frozen=True)
class Inputs:
    interactions: str        # stored intervention episodes (question, A1, C1, stored A2)
    contexts: str            # the exact paragraphs each question was collected with
    stored_labels: str       # the Claude labeling run that produced the stored labels
    feature_sets: str        # feature_sets.json from the feature-selection rerun (read-only)


@dataclass(frozen=True)
class Representation:
    kind: str                # activations | custom | gemma_scope (as in feature selection)
    path: str
    role: str                # control | primary | exploratory


@dataclass(frozen=True)
class Experiment:
    number_of_episodes: int
    number_of_shards: int
    edit_size: float | None  # r: |Δ| = r·|x|. None until the pilot has been run and r chosen.
    number_of_pilot_episodes: int = 50
    pilot_edit_sizes: tuple = (0.02, 0.05, 0.10, 0.20)
    number_of_targeted_features: int = 10
    number_of_random_draws: int = 3
    min_baseline_label_match: float = 0.98
    seed: int = 42


@dataclass(frozen=True)
class Labeling:
    aws_profile: str = "school"
    region: str = "us-east-1"
    concurrency: int = 32


@dataclass(frozen=True)
class Analysis:
    """The pre-specified analysis plan. Fixed before any outcome is seen."""
    primary_representations: tuple          # Table P: these SAEs × the 3 arms, Holm across all
    between_sae_contrasts: tuple            # Table 2, pre-specified family: (A, B) pairs
    exploratory_contrasts: tuple            # Table 2, exploratory family
    direction_contrasts: tuple              # Table D: (arm A, arm B) pairs, within each representation
    primary_alternative: str = "greater"    # one-sided: targeted flips more than random
    secondary_alternative: str = "two-sided"
    n_bootstrap: int = 10000
    n_permutations: int = 10000


@dataclass(frozen=True)
class Machine:
    min_gpu_memory_gb: float = 20.0
    min_free_disk_gb: float = 10.0


@dataclass(frozen=True)
class InterventionConfig:
    layer: int
    partition: str
    inputs: Inputs
    solver: dict             # the collection's Solver spec (mas_sae.models.roles), with a pinned revision
    representations: dict    # name -> Representation, in reporting order
    experiment: Experiment
    analysis: Analysis
    labeling: Labeling
    machine: Machine
    output_dir: Path

    def as_dict(self) -> dict:
        return {**asdict(self), "output_dir": str(self.output_dir)}

    def names_with_role(self, role: str) -> list[str]:
        return [name for name, r in self.representations.items() if r.role == role]


def _as_tuples(value):
    """YAML lists as tuples, so a frozen section is hashable and cannot be changed after loading."""
    return tuple(_as_tuples(v) for v in value) if isinstance(value, list) else value


def _section(cls, raw, where: str):
    """Build one dataclass section, refusing unknown keys and missing required fields."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping.")
    unknown = sorted(set(raw) - {f.name for f in fields(cls)})
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {unknown}")
    missing = [f.name for f in fields(cls)
               if f.default is MISSING and f.default_factory is MISSING and f.name not in raw]
    if missing:
        raise ValueError(f"{where} is missing required fields: {missing}")
    return cls(**{k: _as_tuples(v) for k, v in raw.items()})


def load_config(path: Path) -> InterventionConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return parse_config(raw, source=str(path))


def parse_config(raw: dict, source: str = "config") -> InterventionConfig:
    top = {f.name for f in fields(InterventionConfig)}
    unknown = sorted(set(raw) - top)
    missing = sorted(top - set(raw) - {"labeling", "machine"})
    if unknown:
        raise ValueError(f"Unknown keys in {source}: {unknown}")
    if missing:
        raise ValueError(f"{source} is missing required fields: {missing}")
    representations = {name: _section(Representation, spec, f"representations.{name}")
                       for name, spec in (raw["representations"] or {}).items()}
    config = InterventionConfig(
        layer=raw["layer"], partition=raw["partition"],
        inputs=_section(Inputs, raw["inputs"], "inputs"),
        solver=dict(raw["solver"]),
        representations=representations,
        experiment=_section(Experiment, raw["experiment"], "experiment"),
        analysis=_section(Analysis, raw["analysis"], "analysis"),
        labeling=_section(Labeling, raw.get("labeling", {}), "labeling"),
        machine=_section(Machine, raw.get("machine", {}), "machine"),
        output_dir=Path(raw["output_dir"]),
    )
    problems = validate(config)
    if problems:
        raise ValueError(f"Invalid {source}:\n  - " + "\n  - ".join(problems))
    return config


def validate(config: InterventionConfig) -> list[str]:
    """Every problem with the config's values, so they can all be reported at once."""
    problems = []
    e, a = config.experiment, config.analysis
    positive_int = lambda v: type(v) is int and v > 0
    if type(config.layer) is not int or config.layer < 0:
        problems.append("layer must be a nonnegative integer")
    for name, r in config.representations.items():
        if r.kind not in KINDS:
            problems.append(f"representations.{name}.kind must be one of {KINDS}")
        if r.role not in ROLES:
            problems.append(f"representations.{name}.role must be one of {ROLES}")
    if [n for n, r in config.representations.items() if r.kind == "activations"] != config.names_with_role("control"):
        problems.append("the raw activations, and only they, must have role: control")
    for key in ("number_of_episodes", "number_of_shards", "number_of_pilot_episodes",
                "number_of_targeted_features", "number_of_random_draws"):
        if not positive_int(getattr(e, key)):
            problems.append(f"experiment.{key} must be a positive integer")
    size_ok = lambda v: type(v) in (int, float) and 0 < v < 1
    if e.edit_size is not None and not size_ok(e.edit_size):
        problems.append("experiment.edit_size must be a number in (0, 1), or null until the pilot has run")
    if not e.pilot_edit_sizes or not all(size_ok(v) for v in e.pilot_edit_sizes):
        problems.append("experiment.pilot_edit_sizes must be numbers in (0, 1)")
    if not 0 < e.min_baseline_label_match <= 1:
        problems.append("experiment.min_baseline_label_match must be in (0, 1]")
    generation = config.solver.get("generation", {})
    if generation.get("do_sample", False) is not False:
        problems.append("solver.generation.do_sample must be false: every arm is one greedy generation")
    if not config.solver.get("revision"):
        problems.append("solver.revision must be pinned")
    primary = config.names_with_role("primary")
    if sorted(a.primary_representations) != sorted(primary):
        problems.append(f"analysis.primary_representations must be exactly the role: primary representations {primary}")
    for key in ("between_sae_contrasts", "exploratory_contrasts"):
        for pair in getattr(a, key):
            if len(pair) != 2 or any(n not in config.representations or n in config.names_with_role("control")
                                     for n in pair):
                problems.append(f"analysis.{key}: {list(pair)} must name two configured SAEs")
    for pair in a.direction_contrasts:
        if len(pair) != 2 or any(arm not in ("decrease", "increase", "both") for arm in pair):
            problems.append(f"analysis.direction_contrasts: {list(pair)} must name two arms")
    if a.primary_alternative not in ALTERNATIVES or a.secondary_alternative not in ALTERNATIVES:
        problems.append(f"analysis alternatives must be one of {ALTERNATIVES}")
    if not (positive_int(a.n_bootstrap) and positive_int(a.n_permutations)):
        problems.append("analysis.n_bootstrap and n_permutations must be positive integers")
    if not positive_int(config.labeling.concurrency):
        problems.append("labeling.concurrency must be a positive integer")
    return problems
