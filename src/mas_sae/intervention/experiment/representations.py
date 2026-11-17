"""The representations whose features are edited: raw activations (the control) and the SAEs.

Each has, from ``feature_sets.json`` (read-only, never reselected here), a targeted adoption set, a
targeted retention set and a pool of matched random draws for each (the first
``random_draw_pool_size``), and every set gets its direction u_S once per run (``edits.set_direction``).
Which pool draws a question uses is ``conditions.random_draws_for``.

Raw neurons: a targeted neuron's direction is e_i times the sign of its probe coefficient toward its
set's label (always +1 by how the sets were selected, but computed, not assumed). Random neurons use
+e_i, as random SAE features use their decoder row: neither control uses the probe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from mas_sae.experiments.artifacts import sha256_file
from mas_sae.intervention.generation import edits
from mas_sae.intervention.experiment.config import InterventionConfig
from mas_sae.probe.feature_selection import load_sae, sae_files

if TYPE_CHECKING:
    from mas_sae.intervention.experiment.config import Representation as RepresentationSpec

LABELS = ("adoption", "retention")


@dataclass(frozen=True)
class FeatureSet:
    """One feature set of a representation: its features and the unit direction u_S it is steered along."""
    label: str                 # adoption | retention: the label the set is associated with
    source: str                # targeted | random
    draw: int | None           # index in the random draw pool (None for targeted)
    features: list[int]
    direction: Tensor          # u_S, a unit vector in activation space


@dataclass
class Representation:
    """One representation as the run uses it: the SAE (None for raw activations), its width, its
    targeted set and pool of random sets per label, each with its direction, and its encoder hash."""
    name: str
    kind: str                  # activations | custom | gemma_scope
    role: str                  # control | primary | exploratory
    sae: torch.nn.Module | None
    width: int
    targeted: dict             # label -> FeatureSet
    random: dict               # label -> [FeatureSet, one per pool draw]
    encoder_id: str | None     # SAEs with the same encoder share it, and share z = encode(x)

    @property
    def is_sae(self) -> bool:
        """True for an SAE, False for raw activations."""
        return self.sae is not None

    def feature_set(self, label: str, source: str, draw: int | None) -> FeatureSet:
        """The targeted set for ``label``, or its random set at pool index ``draw`` (not a draw slot)."""
        return self.targeted[label] if source == "targeted" else self.random[label][draw]

    def set_cosine(self) -> float:
        """cos(u_adoption, u_retention) of the two targeted sets."""
        return edits.set_cosine(self.targeted["adoption"].direction, self.targeted["retention"].direction)

    def to(self, device: torch.device | str) -> "Representation":
        """Move the SAE and every set direction to ``device``, in place; returns self."""
        move = lambda s: FeatureSet(s.label, s.source, s.draw, s.features, s.direction.to(device))
        self.sae = None if self.sae is None else self.sae.to(device)
        self.targeted = {k: move(s) for k, s in self.targeted.items()}
        self.random = {k: [move(s) for s in draws] for k, draws in self.random.items()}
        return self


def read_feature_sets(path: Path) -> dict:
    """``feature_sets.json`` from feature selection: {representation name: its entry}."""
    return json.loads(Path(path).read_text())


def checkpoint_sha256(spec: RepresentationSpec) -> dict:
    """sha256 of each file a representation is loaded from, keyed as feature selection recorded them."""
    spec = {"kind": spec.kind, "path": spec.path}
    return {str(p): sha256_file(p) for p in sae_files(spec, Path("."))}


def encoder_id(sae: torch.nn.Module) -> str:
    """Hash of every encoder parameter (all state tensors except the decoder's)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(sae.state_dict().items()):
        if "dec" not in name:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_basis(spec: RepresentationSpec, layer: int) -> tuple[torch.nn.Module | None, int]:
    """(sae, width). Raw activations have no SAE: their width is the length of the center file."""
    if spec.kind == "activations":
        return None, int(np.load(spec.path).shape[0])
    sae = load_sae({"kind": spec.kind, "path": spec.path}, Path("."), layer)[0].eval()
    return sae, int(sae.latent_dim)


def feature_set_problems(name: str, entry: dict, width: int, n_targeted: int, pool_size: int) -> list[str]:
    """Everything wrong with one representation's feature sets in ``feature_sets.json``."""
    problems = []
    valid = lambda ids: (len(ids) == n_targeted and all(type(f) is int and 0 <= f < width for f in ids)
                         and len(set(ids)) == len(ids))
    targeted = {label: [f["feature"] for f in entry.get(f"{label}_features", [])] for label in LABELS}
    for label, ids in targeted.items():
        if not valid(ids):
            problems.append(f"{name}: {label} features must be {n_targeted} unique ints in [0, {width})")
    if set(targeted["adoption"]) & set(targeted["retention"]):
        problems.append(f"{name}: the adoption and retention sets overlap")
    both = set(targeted["adoption"]) | set(targeted["retention"])
    for label in LABELS:
        draws = entry.get(f"random_draws_{label}", [])
        if len(draws) < pool_size:
            problems.append(f"{name}: {len(draws)} random {label} draws, {pool_size} needed")
        for k, draw in enumerate(draws[:pool_size]):
            if not valid(draw):
                problems.append(f"{name}: random {label} draw {k} must be {n_targeted} unique ints in [0, {width})")
            if set(draw) & both:
                problems.append(f"{name}: random {label} draw {k} shares a feature with a targeted set")
    return problems


def _directions(sae, width: int, features: list[int], signs: list[int]) -> Tensor:
    if sae is None:
        return edits.raw_feature_directions(features, signs, width)
    return edits.sae_feature_directions(sae, features)


def build_representation(name: str, spec: RepresentationSpec, entry: dict, sae: torch.nn.Module | None,
                         width: int, pool_size: int) -> Representation:
    """The Representation for one ``feature_sets.json`` ``entry``, with every set's direction computed.

    Targeted raw neurons are signed by their probe coefficient toward the set's label; SAE features use
    their unit decoder rows. The random sets are the first ``pool_size`` draws per label, unsigned."""
    def targeted_set(label):
        rows = entry[f"{label}_features"]
        toward = 1 if label == "adoption" else -1          # coefficients are toward adoption
        signs = [int(np.sign(toward * r["coef_toward_accept"])) for r in rows]
        ids = [r["feature"] for r in rows]
        return FeatureSet(label, "targeted", None, ids, edits.set_direction(_directions(sae, width, ids, signs)))

    def random_set(label, k):
        ids = entry[f"random_draws_{label}"][k]
        return FeatureSet(label, "random", k, ids, edits.set_direction(_directions(sae, width, ids, [1] * len(ids))))

    return Representation(
        name=name, kind=spec.kind, role=spec.role, sae=sae, width=width,
        targeted={label: targeted_set(label) for label in LABELS},
        random={label: [random_set(label, k) for k in range(pool_size)] for label in LABELS},
        encoder_id=None if sae is None else encoder_id(sae))


def load_representations(config: InterventionConfig) -> list[Representation]:
    """Every configured representation, in config order, on the CPU. Assumes the preflight passed."""
    entries = read_feature_sets(Path(config.inputs.feature_sets))
    out = []
    for name, spec in config.representations.items():
        sae, width = load_basis(spec, config.layer)
        out.append(build_representation(name, spec, entries[name], sae, width,
                                        config.experiment.random_draw_pool_size))
    return out


def frozen_feature_sets(representations: list[Representation]) -> dict:
    """The feature sets as they are frozen into ``frozen_run.json``: integers and hashes only, so every
    machine recomputes them exactly (float diagnostics such as the set cosine are recorded separately)."""
    return {r.name: {"kind": r.kind, "role": r.role, "width": r.width, "encoder_id": r.encoder_id,
                     **{f"{label}_targeted": r.targeted[label].features for label in LABELS},
                     **{f"{label}_random": [s.features for s in r.random[label]] for label in LABELS}}
            for r in representations}
