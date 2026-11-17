"""The episodes the experiment runs on: stored intervention episodes with a stored adoption/retention label.

Eligible: the configured partition, Natural Critic, candidates presentation, one stored episode per
question, and a stored Claude label of adoption or retention. The eligible episodes are sorted by
question id and shuffled with the seed; the study takes positions 1..N of that order and choose-r
the next ones, so the two never overlap and neither depends on outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mas_sae.experiments.artifacts import sha256_file
from mas_sae.intervention.experiment.config import InterventionConfig

STRATA = ("adoption", "retention")
OPPOSITE = {"adoption": "retention", "retention": "adoption"}


@dataclass(frozen=True)
class Episode:
    """One eligible stored episode: the question and paragraphs it was collected with, A1, C1, the
    candidate order, the stored A2, and the stored Claude label (the stratum) with its reason."""
    episode_id: str
    question_id: str
    question: str
    paragraphs: list
    solver_attempt_1: str             # A1
    critic_advocated_answer: str      # C1
    candidate_order: str              # which of A1 / C1 the Solver saw as "Candidate A"
    stored_solver_attempt_2: str      # the stored A2
    stored_label: str                 # adoption | retention: the stratum
    stored_reason: str                # Claude's reason for the stored label
    stored_prompt_sha256: str | None  # sha256 of the A2 prompt token ids at collection

    @property
    def target_label(self) -> str:
        """The label an edit tries to flip to."""
        return OPPOSITE[self.stored_label]


def read_jsonl(path: Path) -> list[dict]:
    """Every non-blank line of a JSONL file, parsed."""
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_stored_labels(labels_dir: Path, interactions: Path) -> dict[str, dict]:
    """Claude's stored label row for each episode id, after checking it labeled these interactions."""
    settings = json.loads((labels_dir / "run_settings.json").read_text())
    if settings["episodes_sha256"] != sha256_file(interactions):
        raise ValueError(f"{labels_dir} labeled a different interactions file than {interactions}.")
    rows = read_jsonl(labels_dir / "raw_results.jsonl")
    labels = {r["episode_id"]: r for r in rows}
    if len(labels) != len(rows):
        raise ValueError(f"{labels_dir} has duplicate episode ids.")
    return labels


def eligible_episodes(interactions: Path, contexts: Path, labels_dir: Path, partition: str) -> list[Episode]:
    """Every eligible episode, sorted by question id."""
    interactions, contexts, labels_dir = Path(interactions), Path(contexts), Path(labels_dir)
    labels = read_stored_labels(labels_dir, interactions)
    paragraphs = {r["question_id"]: r["payload"] for r in read_jsonl(contexts)}
    seen, episodes = set(), []
    for r in read_jsonl(interactions):
        label = labels.get(r["episode_id"], {})
        if (r.get("partition") != partition or r.get("critic_condition") != "natural"
                or label.get("behavior_label") not in STRATA):
            continue
        qid = r["question_id"]
        if qid in seen:
            raise ValueError(f"{qid}: more than one stored episode.")
        seen.add(qid)
        if r.get("a2_presentation") != "candidates":
            raise ValueError(f"{qid}: a2_presentation is {r.get('a2_presentation')!r}; only candidates is supported.")
        context = paragraphs.get(qid)
        if context is None or context.get("question") != r["question"]:
            raise ValueError(f"{qid}: missing or mismatched collected context.")
        prompt = (r.get("generation_telemetry") or {}).get("solver_a2_prompt") or {}
        episodes.append(Episode(
            episode_id=r["episode_id"], question_id=qid, question=r["question"],
            paragraphs=context["paragraphs"], solver_attempt_1=r["solver_attempt_1"],
            critic_advocated_answer=r["critic_advocated_answer"], candidate_order=r["candidate_order"],
            stored_solver_attempt_2=r["solver_attempt_2"], stored_label=label["behavior_label"],
            stored_reason=label.get("claude_reason") or "", stored_prompt_sha256=prompt.get("input_ids_sha256")))
    return sorted(episodes, key=lambda ep: ep.question_id)


def seeded_order(episodes: list[Episode], seed: int) -> list[Episode]:
    """The eligible episodes in a random order fixed by ``seed``."""
    return [episodes[i] for i in np.random.default_rng(seed).permutation(len(episodes))]


def study_and_choose_r_episodes(config: InterventionConfig) -> tuple[list[Episode], list[Episode]]:
    """(study, choose-r): positions 1..N and N+1..N+P of the seeded order. Refuses if too few are eligible."""
    experiment = config.experiment
    order = seeded_order(eligible_episodes(Path(config.inputs.interactions), Path(config.inputs.contexts),
                                           Path(config.inputs.stored_labels), config.partition), experiment.seed)
    needed = experiment.number_of_episodes + experiment.number_of_choose_r_episodes
    if len(order) < needed:
        raise ValueError(f"Only {len(order)} eligible episodes; the study and choose-r need {needed}.")
    return order[:experiment.number_of_episodes], order[experiment.number_of_episodes:needed]
