"""Does the plain replay (no SAE, no edit) reproduce the stored A2s? If not, no edit means anything.

Passes only if every episode has a baseline, no prompt differs from collection's, and the label
match rate is at least the configured minimum. Labels are not known before merge, so an episode
counts as a label match only when its A2 is identical to the stored A2 (which reuses the stored
label); any other answer counts as a mismatch. This can only make the check stricter.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mas_sae.intervention.experiment.episodes import Episode


def normalize(text: str) -> str:
    """Lowercased, punctuation turned into spaces, whitespace collapsed: separates formatting-only
    differences from text differences."""
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def mismatch_kind(row: dict, episode: Episode) -> str:
    """How a baseline row compares with the stored episode: "prompt_differs" (checked first, only when
    collection's prompt hash is known), "exact_match", "formatting_only" or "text_differs"."""
    if episode.stored_prompt_sha256 is not None and row["prompt_sha256"] != episode.stored_prompt_sha256:
        return "prompt_differs"
    if row["answer"] == episode.stored_solver_attempt_2:
        return "exact_match"
    if normalize(row["answer"]) == normalize(episode.stored_solver_attempt_2):
        return "formatting_only"
    return "text_differs"


def check_baseline_reproduces_stored(rows: list[dict], episodes: list, min_label_match: float) -> tuple[dict, pd.DataFrame]:
    """(check, mismatches): the summary written to baseline_check.json, and every non-exact episode."""
    by_id = {r["episode_id"]: r for r in rows}
    kinds = {ep.episode_id: mismatch_kind(by_id[ep.episode_id], ep) for ep in episodes if ep.episode_id in by_id}
    counts = pd.Series(list(kinds.values()), dtype=object).value_counts().to_dict()
    label_match = counts.get("exact_match", 0) / len(episodes)
    failures = []
    if len(kinds) < len(episodes):
        failures.append(f"{len(episodes) - len(kinds)} episodes have no baseline")
    if counts.get("prompt_differs"):
        failures.append(f"{counts['prompt_differs']} prompts differ from collection's")
    if label_match < min_label_match:
        failures.append(f"label match {label_match:.3f} < {min_label_match}")
    not_compared = sum(ep.stored_prompt_sha256 is None for ep in episodes)
    check = {"n_episodes": len(episodes), "mismatch_counts": counts, "label_match_rate": label_match,
             "n_prompts_not_compared": not_compared,
             "criterion": {"min_label_match": min_label_match, "max_prompt_differences": 0,
                           "label_match": "an identical A2 (reuses the stored label); anything else is a mismatch"},
             "passed": not failures, "failures": failures}
    mismatches = pd.DataFrame([{"episode_id": ep.episode_id, "kind": kinds[ep.episode_id],
                                "stored_solver_attempt_2": ep.stored_solver_attempt_2,
                                "replayed_solver_attempt_2": by_id[ep.episode_id]["answer"]}
                               for ep in episodes if kinds.get(ep.episode_id, "exact_match") != "exact_match"],
                              columns=["episode_id", "kind", "stored_solver_attempt_2", "replayed_solver_attempt_2"])
    return check, mismatches
