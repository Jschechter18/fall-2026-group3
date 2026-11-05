"""The conditions generated for one episode. Every condition is one greedy generation of A2.

The arms are defined relative to the episode's stored label, so the list is the same for every
episode: "decrease" pushes away from the stored label's features, "increase" toward the target
label's features, and "both" does the two in one edit (see ``edits``).

  study, per episode:  1 baseline
                     + 1 reconstruction splice per SAE
                     + per representation × arm: the targeted sets and each random draw
                     = 1 + 4 + 5 × 3 × (1 + 3) = 65 with the configured representations

Random draws: a condition's ``draw`` is a slot k (0..n_draws−1). Each question has its own n_draws
distinct indices into the pool of matched random draws (``random_draws_for``, fixed by the seed and
the question id), and slot k uses the k-th of them, for both labels, so across questions the random
baseline samples the whole pool rather than the same few sets.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from mas_sae.intervention.generation.edits import ARMS


@dataclass(frozen=True)
class Condition:
    kind: str                        # baseline | reconstruction | edit
    representation: str | None = None
    arm: str | None = None           # decrease | increase | both
    source: str | None = None        # targeted | random
    draw: int | None = None          # random draw slot k; the question's k-th pool index (``random_draws_for``)
    edit_size: float | None = None   # r

    @property
    def key(self) -> str:
        """Identifies the condition within an episode, e.g. ``edit|L1 SAE|both|random|2|0.05``."""
        parts = [self.kind, self.representation, self.arm, self.source, self.draw, self.edit_size]
        return "|".join("" if p is None else str(p) for p in parts).rstrip("|")

    def as_dict(self) -> dict:
        return asdict(self)


BASELINE = Condition("baseline")


def edit_conditions(representation: str, edit_size: float, n_draws: int) -> list[Condition]:
    """Each arm with the targeted sets, then with each random draw."""
    return [Condition("edit", representation, arm, source, draw, edit_size)
            for arm in ARMS
            for source, draw in [("targeted", None), *(("random", k) for k in range(n_draws))]]


def study_conditions(representations, edit_size: float, n_draws: int) -> list[Condition]:
    conditions = [BASELINE]
    conditions += [Condition("reconstruction", r.name) for r in representations if r.is_sae]
    for r in representations:
        conditions += edit_conditions(r.name, edit_size, n_draws)
    return conditions


def choose_r_conditions(representations, edit_sizes, n_draws: int) -> list[Condition]:
    """Baseline, then every edit at every candidate size for the given (primary) SAEs."""
    conditions = [BASELINE]
    for size in edit_sizes:
        for r in representations:
            conditions += edit_conditions(r.name, size, n_draws)
    return conditions


def random_draws_for(question_id: str, seed: int, pool_size: int, n_draws: int) -> list[int]:
    """The question's ``n_draws`` distinct pool indices, one per random draw slot.

    Seeded by the experiment seed and the question id, so every machine and rerun gets the same ones."""
    digest = hashlib.sha256(f"random draws|{seed}|{question_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return [int(i) for i in rng.choice(pool_size, size=n_draws, replace=False)]


def pool_draw(condition: Condition, random_draws: list[int]) -> int | None:
    """The pool index a random condition uses on a question with ``random_draws``; None otherwise."""
    return random_draws[condition.draw] if condition.source == "random" else None


def sets_for(condition: Condition, representation, stored_label: str, random_draws: list[int]):
    """(toward, away, edited set name) for an edit condition on an episode with ``stored_label``.

    ``toward`` is the set of the label we flip to, ``away`` the stored label's set. For a random
    condition both come from the same pool index, the question's ``random_draws[k]`` for slot k:
    "both" pairs adoption draw i with retention draw i."""
    target_label = "retention" if stored_label == "adoption" else "adoption"
    draw = pool_draw(condition, random_draws)
    toward = representation.feature_set(target_label, condition.source, draw)
    away = representation.feature_set(stored_label, condition.source, draw)
    edited = {"increase": toward.label, "decrease": away.label, "both": "both"}[condition.arm]
    return toward, away, edited


def edited_features(condition: Condition, toward, away) -> list[int]:
    """The features whose direction an edit moves along."""
    return {"increase": toward.features, "decrease": away.features,
            "both": [*toward.features, *away.features]}[condition.arm]
