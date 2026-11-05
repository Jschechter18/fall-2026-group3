"""Generating one episode's conditions: the loop shared by choose-r and every run worker.

Per episode:
  1. Generate the baseline (no SAE, no edit). This also records layers 0..25 of the prefill and the
     final prompt activation x, which every other condition starts from.
  2. On a run worker, require the baseline to equal prepare's (same prompt, answer and first token):
     otherwise differences could come from the machine, not the edit. A mismatch raises
     BaselineMismatch, and the worker excludes that episode instead of stopping.
  3. Encode x once per SAE encoder (the Pretrained SAE and the fine-tuned decoder share one).
  4. For each condition not yet in the results file: compute x', generate A2 with x replaced by x',
     and append the row at once.
"""

from __future__ import annotations

import torch

from mas_sae.intervention.generation import edits
from mas_sae.intervention.experiment.conditions import Condition, edited_features, pool_draw, sets_for
from mas_sae.intervention.generation.solver import generate_baseline, generate_with_activation


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


def apply_condition(condition: Condition, x: torch.Tensor, rep, stored_label: str, codes: dict,
                    random_draws: list[int]):
    """(x', diagnostics, edited set name) for a reconstruction or edit condition. ``x`` is float32;
    ``random_draws`` are the question's pool indices (``conditions.random_draws_for``)."""
    z, reconstruction = codes.get(rep.name, (None, None))
    if condition.kind == "reconstruction":
        return reconstruction, edits.diagnose(x, reconstruction - x, [], z, reconstruction), None
    toward, away, edited_set = sets_for(condition, rep, stored_label, random_draws)
    delta = edits.edit_delta(condition.arm, x, toward.direction, away.direction, condition.edit_size)
    diagnostics = edits.diagnose(x, delta, edited_features(condition, toward, away), z, reconstruction)
    return x + delta, diagnostics, edited_set


def make_row(episode, condition: Condition, generation, baseline, diagnostics, edited_set,
             random_pool_draw: int | None = None) -> dict:
    """One results-file line: what was generated, under which condition, and how the edit went."""
    return {
        "episode_id": episode.episode_id, "question_id": episode.question_id, "stored_label": episode.stored_label,
        "condition": condition.key, **condition.as_dict(), "edited_set": edited_set,
        "random_pool_draw": random_pool_draw,                          # pool index of a random condition
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


class BaselineMismatch(RuntimeError):
    """This worker's baseline of one episode differs from prepare's: the run worker excludes that episode."""


def require_same_baseline(baseline, prepared: dict, episode) -> None:
    differs = [name for name, ours, theirs in (("prompt", baseline.prompt_sha256, prepared["prompt_sha256"]),
                                               ("answer", baseline.answer, prepared["answer"]),
                                               ("first token", baseline.first_token_id, prepared["first_token_id"]))
               if ours != theirs]
    if differs:
        raise BaselineMismatch(f"{episode.episode_id}: this worker's baseline differs from prepare's "
                               f"({', '.join(differs)}); its edits would not be comparable.")


def generate_episode(solver_model, episode, conditions: list[Condition], representations: dict, results,
                     random_draws: list[int], prepared_baseline: dict | None = None,
                     use_prefill_cache: bool = True) -> int:
    """Generate every condition of ``episode`` not yet in ``results``. Returns how many were generated.

    ``random_draws`` are the question's pool indices, one per random draw slot (``conditions.random_draws_for``).
    ``use_prefill_cache=False`` computes every prefill in full (the plain path the parity test uses)."""
    # Resume: skip conditions already in the results file; stop early if none are left.
    missing_conditions = [c for c in conditions if not results.has(episode.episode_id, c.key)]
    if not missing_conditions:
        return 0

    # Baseline: plain generation that also records layers 0..25 of the prefill for reuse below.
    baseline, prefill = generate_baseline(solver_model, episode)
    # Run workers only: the baseline must match prepare's, or this machine's results aren't comparable.
    # Raises BaselineMismatch before any of this episode's rows are written; the worker excludes it.
    if prepared_baseline is not None:
        require_same_baseline(baseline, prepared_baseline, episode)

    # The activation every condition edits (bf16, as in the model), a float32 copy for the edit math,
    # and each SAE's code and reconstruction of it, computed once for all conditions.
    x = prefill.final_activation
    x32 = x.float()
    codes = encode_once(x32, representations)

    for condition in missing_conditions:
        # The baseline condition is the generation above: just record it.
        if condition.kind == "baseline":
            results.append(make_row(episode, condition, baseline, baseline, None, None))
            continue
        # Build x' (an SAE reconstruction or an edit of x) and diagnostics on how the edit went.
        x_new, diagnostics, edited_set = apply_condition(condition, x32, representations[condition.representation],
                                                         episode.stored_label, codes, random_draws)
        # Regenerate A2 with x replaced by x', replaying the cached layers 0..25 instead of recomputing them.
        generation = generate_with_activation(solver_model, episode, x, x_new,
                                              prefill if use_prefill_cache else None)
        # Sanity check: the edited run must have seen the exact same prompt as the baseline.
        if generation.prompt_sha256 != baseline.prompt_sha256:
            raise RuntimeError(f"{episode.episode_id}: a condition's prompt differs from the baseline's.")
        # Write the row right away so a crash loses at most this one condition.
        results.append(make_row(episode, condition, generation, baseline, diagnostics, edited_set,
                                pool_draw(condition, random_draws)))
    return len(missing_conditions)
