"""steer_test: the stage-6 "both" arm at r = 0.2 on fixed TRAIN episodes, on the GPU.

Used by the LLM-edit harness (Claude's steer_test() calls) and to check the controls. It compares an
edited decoder with the unedited Pretrained SAE on the same episodes: the first `n` of the seeded
order of eligible train episodes (the study's seed), regenerated with stage 6's own generation code.
The first call also generates the baseline and the unedited SAE's rows (cached in unedited.jsonl);
each call then generates only the edit's rows (call_K.jsonl).

Flips: answers are classified by exact (normalized) match with A1 or the Critic's answer
(mas_sae.evaluation.scoring). On the stored train A2s this never contradicts a stored Claude label and
leaves 6% unclear, which are reported, not counted as flips.

Candidate margin: log p(the other answer's first token) − log p(the original answer's first token) at
the first generated position, from the candidate first tokens collection recorded
(candidate_first_token_ids); defined only where the two differ (decision_token_distinct, about half
the episodes). Unlike log p of the original first token alone, it does not move when an edit only
changes wording: in the Phase 3 smoke run a leading "The" cost the original first token 7 nats with
no change in behavior.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from mas_sae.evaluation.scoring import answer_matches
from mas_sae.experiments.records import read_jsonl
from mas_sae.intervention.experiment.conditions import BASELINE, Condition
from mas_sae.intervention.experiment.episodes import eligible_episodes, seeded_order
from mas_sae.intervention.experiment.representations import build_representation, read_feature_sets
from mas_sae.intervention.generation.episode import apply_condition, encode_once, make_row
from mas_sae.intervention.generation.solver import generate_baseline, generate_with_activation, load_solver
from mas_sae.intervention.run_files.results_file import ResultsFile
from mas_sae.llm_edit.edited_sae import SAE_DIR, SAE_NAME, jump_relu_sae

TRAIN_INTERACTIONS = Path(
    "artifacts/collection/mixed_final_001/prepared/results/mixed_final_001/train/interactions.jsonl")
TRAIN_CONTEXTS = Path("artifacts/llm_edit/train_contexts.jsonl")   # Phase 1
TRAIN_LABELS = Path("results/claude_validation/mixed_final_001/train")
UNEDITED, EDITED = "Pretrained SAE (unedited)", "Edited decoder"


def both_arm(representation: str, edit_size: float) -> Condition:
    return Condition("edit", representation, "both", "targeted", None, edit_size)


class Steerer:
    """`Steerer(study_config, n, out_dir).test(w_dec, call)` -> (summary dict, message text)."""

    def __init__(self, study, n_episodes: int, out_dir: Path, sae_dir: Path = SAE_DIR):
        self.study, self.out_dir, self.sae_dir = study, Path(out_dir), Path(sae_dir)
        self.edit_size = study.experiment.edit_size
        order = seeded_order(eligible_episodes(TRAIN_INTERACTIONS, TRAIN_CONTEXTS, TRAIN_LABELS, "train"),
                             study.experiment.seed)
        self.episodes = order[:n_episodes]
        del order
        gc.collect()                      # the full contexts file is large; keep only these episodes
        self.candidates = candidate_first_tokens(TRAIN_INTERACTIONS, {e.episode_id for e in self.episodes})
        self.solver_model = None
        self.unedited = None

    def test(self, w_dec: np.ndarray, call: int) -> tuple[dict, str]:
        self._load()
        edited = self._representation(EDITED, w_dec)
        unedited_rows = ResultsFile(self.out_dir / "unedited.jsonl")
        call_rows = ResultsFile(self.out_dir / f"call_{call}.jsonl")
        for episode in self.episodes:
            self._generate(episode, [BASELINE, both_arm(UNEDITED, self.edit_size)], {UNEDITED: self.unedited},
                           unedited_rows)
            self._generate(episode, [both_arm(EDITED, self.edit_size)], {EDITED: edited}, call_rows)
        summary = self._summarize(unedited_rows.rows, call_rows.rows)
        return summary, steer_message(call, summary, self.edit_size)

    def _load(self) -> None:
        if self.solver_model is None:
            from safetensors.numpy import load_file
            self.solver_model = load_solver(self.study)
            self.unedited = self._representation(UNEDITED, load_file(str(self.sae_dir / "params.safetensors"))["w_dec"])

    def _representation(self, name: str, w_dec: np.ndarray):
        """The Pretrained SAE's feature sets, with directions from the given decoder, on the Solver's device."""
        entry = read_feature_sets(Path(self.study.inputs.feature_sets))[SAE_NAME]
        sae = jump_relu_sae(w_dec, self.sae_dir)
        representation = build_representation(name, self.study.representations[SAE_NAME], entry, sae,
                                              sae.latent_dim, pool_size=1)    # random sets are not used here
        return representation.to(self.solver_model.model.device)

    def _generate(self, episode, conditions: list[Condition], representations: dict, results: ResultsFile) -> None:
        """generate_episode's steps (mas_sae.intervention.generation.episode), with the same functions, but
        each row also keeps the candidate margin, which needs the generation's first-token log-probabilities
        (generate_episode does not keep them; its shared stage-6 code is left unchanged)."""
        missing = [c for c in conditions if not results.has(episode.episode_id, c.key)]
        if not missing:
            return
        baseline, prefill = generate_baseline(self.solver_model, episode)
        x = prefill.final_activation
        x32 = x.float()
        codes = encode_once(x32, representations)
        for condition in missing:
            if condition.kind == "baseline":
                generation, row = baseline, make_row(episode, condition, baseline, baseline, None, None)
            else:
                x_new, diagnostics, edited_set = apply_condition(condition, x32, representations[condition.representation],
                                                                 episode.stored_label, codes, [])
                generation = generate_with_activation(self.solver_model, episode, x, x_new, prefill)
                if generation.prompt_sha256 != baseline.prompt_sha256:
                    raise RuntimeError(f"{episode.episode_id}: a condition's prompt differs from the baseline's.")
                row = make_row(episode, condition, generation, baseline, diagnostics, edited_set)
            results.append({**row, "candidate_margin": self._candidate_margin(generation, episode)})

    def _candidate_margin(self, generation, episode) -> float | None:
        tokens = self.candidates[episode.episode_id]
        if tokens is None:
            return None
        original, other = ("a1", "critic") if episode.stored_label == "retention" else ("critic", "a1")
        return generation.logprob_of(tokens[other]) - generation.logprob_of(tokens[original])

    def _summarize(self, unedited_rows: list[dict], edited_rows: list[dict]) -> dict:
        by_key = {(r["episode_id"], r["condition"]): r for r in unedited_rows + edited_rows}
        keys = {"unedited": both_arm(UNEDITED, self.edit_size).key, "edited": both_arm(EDITED, self.edit_size).key}
        counts = {name: {"flips": 0, "adoption_to_retention": 0, "retention_to_adoption": 0, "unclear": 0,
                         "margin_changes": [], "logprob_shifts": []} for name in keys}
        reproduced = 0
        for episode in self.episodes:
            baseline = by_key[(episode.episode_id, BASELINE.key)]
            reproduced += baseline["answer"] == episode.stored_solver_attempt_2
            for name, key in keys.items():
                row, c = by_key[(episode.episode_id, key)], counts[name]
                behavior = classify_answer(row["answer"], episode)
                if behavior is None:
                    c["unclear"] += 1
                elif behavior != episode.stored_label:
                    c["flips"] += 1
                    c[f"{episode.stored_label}_to_{behavior}"] += 1
                if row["candidate_margin"] is not None:
                    c["margin_changes"].append(row["candidate_margin"] - baseline["candidate_margin"])
                # Logged for diagnostics only, not shown to Claude: it also moves with wording (module docstring).
                c["logprob_shifts"].append(row["baseline_first_token_logprob"] - baseline["first_token_logprob"])
        for c in counts.values():
            changes = c.pop("margin_changes")
            c["mean_margin_change"] = float(np.mean(changes)) if changes else None
            c["mean_logprob_shift"] = float(np.mean(c.pop("logprob_shifts")))
        return {"episodes": len(self.episodes),
                "adopted": sum(e.stored_label == "adoption" for e in self.episodes),
                "margin_episodes": sum(self.candidates[e.episode_id] is not None for e in self.episodes),
                "baseline_reproduces_stored_a2": reproduced, **counts}


def candidate_first_tokens(interactions: Path, episode_ids: set[str]) -> dict[str, dict | None]:
    """{episode id: {"a1": token id, "critic": token id}}, or None where the two first tokens are the same."""
    out = {}
    for row in read_jsonl(interactions):
        if row["episode_id"] in episode_ids:
            tokens = row.get("candidate_first_token_ids") or {}
            distinct = row.get("decision_token_distinct") and tokens.get("a1") != tokens.get("critic")
            out[row["episode_id"]] = tokens if distinct else None
    if missing := episode_ids - set(out):
        raise SystemExit(f"{len(missing)} steer_test episodes are not in {interactions}")
    return out


def classify_answer(answer: str, episode) -> str | None:
    """retention if A2 matches A1 only, adoption if it matches the Critic's answer only, else None (unclear)."""
    keeps = answer_matches(answer, episode.solver_attempt_1)
    adopts = answer_matches(answer, episode.critic_advocated_answer)
    if keeps != adopts:
        return "retention" if keeps else "adoption"
    return None


def steer_message(call: int, s: dict, edit_size: float) -> str:
    n, adopted = s["episodes"], s["adopted"]
    e, u = s["edited"], s["unedited"]
    return (f"steer_test #{call} on {n} train episodes (the 'both' arm, r = {edit_size}): your current decoder "
            f"vs. the unedited SAE.\n"
            f"  flips (the second answer moved to the opposite behavior): yours {e['flips']}, unedited {u['flips']}\n"
            f"    originally adopted, now retains: yours {e['adoption_to_retention']}, "
            f"unedited {u['adoption_to_retention']} (of {adopted})\n"
            f"    originally retained, now adopts: yours {e['retention_to_adoption']}, "
            f"unedited {u['retention_to_adoption']} (of {n - adopted})\n"
            f"  answers matching neither original answer (not counted as flips): yours {e['unclear']}, "
            f"unedited {u['unclear']}\n"
            f"  mean change in the candidate margin, log p(other answer's first token) − log p(original answer's "
            f"first token), on the {s['margin_episodes']} episodes whose two answers start with different tokens: "
            f"yours {margin_text(e['mean_margin_change'])}, unedited {margin_text(u['mean_margin_change'])} "
            f"(positive = pushed toward the other answer)\n"
            f"Flips are counted by exact (normalized) answer match with the first answer or the Critic's answer.")


def margin_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"
