"""The conditions generated for one episode. Every condition is one greedy generation of A2.

The arms are defined relative to the episode's stored label, so the list is the same for every
episode: "decrease" pushes away from the stored label's features, "increase" toward the target
label's features, and "both" does the two in one edit (see ``edits``).

  study, per episode:  1 baseline
                     + 1 reconstruction splice per SAE
                     + per representation × arm: the targeted sets and each random draw
                     = 1 + 4 + 5 × 3 × (1 + 3) = 65 with the configured representations
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mas_sae.intervention.edits import ARMS


@dataclass(frozen=True)
class Condition:
    kind: str                        # baseline | reconstruction | edit
    representation: str | None = None
    arm: str | None = None           # decrease | increase | both
    source: str | None = None        # targeted | random
    draw: int | None = None          # random draw number
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


def pilot_conditions(representations, edit_sizes, n_draws: int) -> list[Condition]:
    """Baseline, then every edit at every pilot size for the given (primary) SAEs."""
    conditions = [BASELINE]
    for size in edit_sizes:
        for r in representations:
            conditions += edit_conditions(r.name, size, n_draws)
    return conditions


def sets_for(condition: Condition, representation, stored_label: str):
    """(toward, away, edited set name) for an edit condition on an episode with ``stored_label``.

    ``toward`` is the set of the label we flip to, ``away`` the stored label's set. For a random
    condition, draw k of each label: "both" pairs adoption draw k with retention draw k."""
    target_label = "retention" if stored_label == "adoption" else "adoption"
    toward = representation.feature_set(target_label, condition.source, condition.draw)
    away = representation.feature_set(stored_label, condition.source, condition.draw)
    edited = {"increase": toward.label, "decrease": away.label, "both": "both"}[condition.arm]
    return toward, away, edited


def edited_features(condition: Condition, toward, away) -> list[int]:
    """The features whose direction an edit moves along."""
    return {"increase": toward.features, "decrease": away.features,
            "both": [*toward.features, *away.features]}[condition.arm]
