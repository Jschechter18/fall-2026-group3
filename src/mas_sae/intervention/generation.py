"""Generating one episode's conditions: the loop shared by the pilot and every run worker.

Per episode:
  1. Generate the baseline (no SAE, no edit). This also records layers 0..25 of the prefill and the
     final prompt activation x, which every other condition starts from.
  2. On a run worker, require the baseline to equal prepare's (same prompt, answer and first token):
     otherwise differences could come from the machine, not the edit.
  3. Encode x once per SAE encoder (the Pretrained SAE and the fine-tuned decoder share one).
  4. For each condition not yet in the results file: compute x', generate A2 with x replaced by x',
     and append the row at once.
"""

from __future__ import annotations

import torch

from mas_sae.intervention import edits
from mas_sae.intervention.conditions import Condition, edited_features, sets_for
from mas_sae.intervention.solver import generate_baseline, generate_with_activation


@torch.no_grad()
def encode_once(x: torch.Tensor, representations: dict) -> dict:
    """{representation name: (z, decode(z))} for every SAE; each distinct encoder runs once."""
    codes, by_encoder = {}, {}
    for name, rep in representations.items():
        if not rep.is_sae:
            continue
        if rep.encoder_id not in by_encoder:
            by_encoder[rep.encoder_id] = rep.sae.encoder(x[None])[0]
        z = by_encoder[rep.encoder_id]
        codes[name] = (z, rep.sae.decoder(z[None])[0].float())
    return codes


def apply_condition(condition: Condition, x: torch.Tensor, rep, stored_label: str, codes: dict):
    """(x', diagnostics, edited set name) for a reconstruction or edit condition. ``x`` is float32."""
    z, reconstruction = codes.get(rep.name, (None, None))
    if condition.kind == "reconstruction":
        return reconstruction, edits.diagnose(x, reconstruction - x, [], z, reconstruction), None
    toward, away, edited_set = sets_for(condition, rep, stored_label)
    delta = edits.edit_delta(condition.arm, x, toward.direction, away.direction, condition.edit_size)
    diagnostics = edits.diagnose(x, delta, edited_features(condition, toward, away), z, reconstruction)
    return x + delta, diagnostics, edited_set


def make_row(episode, condition: Condition, generation, baseline, diagnostics, edited_set) -> dict:
    """One results-file line: what was generated, under which condition, and how the edit went."""
    return {
        "episode_id": episode.episode_id, "question_id": episode.question_id, "stored_label": episode.stored_label,
        "condition": condition.key, **condition.as_dict(), "edited_set": edited_set,
        "answer": generation.answer, "prompt_sha256": generation.prompt_sha256,
        "generated_token_ids": generation.generated_token_ids,
        "reached_token_budget": generation.reached_token_budget,
        "first_token_id": generation.first_token_id,
        "first_token_logprob": generation.logprob_of(generation.first_token_id),
        "baseline_first_token_logprob": generation.logprob_of(baseline.first_token_id),
        "applied_relative_size": generation.applied_relative_size,     # as written into the model (bf16)
        "edited_features_active": None if diagnostics is None else diagnostics.edited_features_active,
        "reconstruction_error": None if diagnostics is None else diagnostics.reconstruction_error,
        "seconds": round(generation.seconds, 4),
    }


def require_same_baseline(baseline, prepared: dict, episode) -> None:
    same = (baseline.prompt_sha256 == prepared["prompt_sha256"] and baseline.answer == prepared["answer"]
            and baseline.first_token_id == prepared["first_token_id"])
    if not same:
        raise RuntimeError(f"{episode.episode_id}: this worker's baseline differs from prepare's "
                           "(prompt, answer or first token); its edits would not be comparable.")


def generate_episode(solver_model, episode, conditions: list[Condition], representations: dict, results,
                     prepared_baseline: dict | None = None, use_prefill_cache: bool = True) -> int:
    """Generate every condition of ``episode`` not yet in ``results``. Returns how many were generated.

    ``use_prefill_cache=False`` computes every prefill in full (the plain path the parity test uses)."""
    todo = [c for c in conditions if not results.has(episode.episode_id, c.key)]
    if not todo:
        return 0
    baseline, prefill = generate_baseline(solver_model, episode)
    if prepared_baseline is not None:
        require_same_baseline(baseline, prepared_baseline, episode)
    x = prefill.final_activation
    x32 = x.float()
    codes = encode_once(x32, representations)
    for condition in todo:
        if condition.kind == "baseline":
            results.append(make_row(episode, condition, baseline, baseline, None, None))
            continue
        x_new, diagnostics, edited_set = apply_condition(condition, x32, representations[condition.representation],
                                                         episode.stored_label, codes)
        generation = generate_with_activation(solver_model, episode, x, x_new,
                                              prefill if use_prefill_cache else None)
        if generation.prompt_sha256 != baseline.prompt_sha256:
            raise RuntimeError(f"{episode.episode_id}: a condition's prompt differs from the baseline's.")
        results.append(make_row(episode, condition, generation, baseline, diagnostics, edited_set))
    return len(todo)
