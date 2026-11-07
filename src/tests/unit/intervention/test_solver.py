"""Generation: the prefill cache gives the plain path's rows, the edit hook touches only the prefill's final token."""

import pytest
import torch

from . import fakes
from mas_sae.intervention.experiment.conditions import study_conditions
from mas_sae.intervention.generation.episode import encode_once, generate_episode, require_same_baseline
from mas_sae.intervention.run_files.results_file import ResultsFile
from mas_sae.intervention.generation.solver import ReplaceFinalPromptActivation, generate_baseline, generate_with_activation

SIZE = 0.5          # large, so edits visibly change this tiny model's answers


def _rows_without_time(path):
    return [{k: v for k, v in r.items() if k != "seconds"} for r in ResultsFile(path).rows]


def test_prefill_cache_rows_equal_the_plain_generation_rows(tmp_path):
    solver_model, reps = fakes.solver_model(), fakes.representations()
    conditions = study_conditions(list(reps.values()), SIZE, fakes.N_DRAWS)
    episodes = [fakes.episode(0, "adoption"), fakes.episode(1, "retention")]
    for name, cached in (("cached", True), ("plain", False)):
        results = ResultsFile(tmp_path / f"{name}.jsonl")
        for ep in episodes:
            generate_episode(solver_model, ep, conditions, reps, results, fakes.DRAWS, use_prefill_cache=cached)
    cached, plain = _rows_without_time(tmp_path / "cached.jsonl"), _rows_without_time(tmp_path / "plain.jsonl")
    assert len(cached) == len(episodes) * len(conditions) and cached == plain
    baseline = {r["episode_id"]: r["answer"] for r in cached if r["kind"] == "baseline"}
    assert any(r["answer"] != baseline[r["episode_id"]] for r in cached if r["kind"] == "edit")


def test_every_edit_is_written_at_size_r(tmp_path):
    solver_model, reps = fakes.solver_model(), fakes.representations()
    results = ResultsFile(tmp_path / "rows.jsonl")
    generate_episode(solver_model, fakes.episode(), study_conditions(list(reps.values()), SIZE, fakes.N_DRAWS),
                     reps, results, fakes.DRAWS)
    edits = [r for r in results.rows if r["kind"] == "edit"]
    assert all(r["applied_relative_size"] == pytest.approx(SIZE, rel=1e-5) for r in edits)     # float32 model
    sae_edits = [r for r in edits if r["representation"] != "Raw"]
    assert all(isinstance(r["edited_features_active"], int) for r in sae_edits)
    assert all(r["edited_features_active"] is None for r in edits if r["representation"] == "Raw")


def test_each_random_row_records_the_pool_draw_it_used(tmp_path):
    solver_model, reps = fakes.solver_model(), fakes.representations()
    results = ResultsFile(tmp_path / "rows.jsonl")
    generate_episode(solver_model, fakes.episode(), study_conditions(list(reps.values()), SIZE, fakes.N_DRAWS),
                     reps, results, [3, 2])
    assert {(r["draw"], r["random_pool_draw"]) for r in results.rows if r["source"] == "random"} == {(0, 3), (1, 2)}
    assert all(r["random_pool_draw"] is None for r in results.rows if r["source"] != "random")


def test_rerun_generates_nothing_and_a_partial_episode_finishes(tmp_path):
    solver_model, reps = fakes.solver_model(), fakes.representations()
    conditions = study_conditions(list(reps.values()), SIZE, fakes.N_DRAWS)
    results = ResultsFile(tmp_path / "rows.jsonl")
    assert generate_episode(solver_model, fakes.episode(), conditions[:5], reps, results, fakes.DRAWS) == 5
    assert generate_episode(solver_model, fakes.episode(), conditions, reps, results, fakes.DRAWS) == len(conditions) - 5
    assert generate_episode(solver_model, fakes.episode(), conditions, reps, ResultsFile(tmp_path / "rows.jsonl"),
                            fakes.DRAWS) == 0


def _next_layer_prefill_input(solver_model, run):
    """What the layer after the edited one receives in the prefill: the edited layer's output, after the edit."""
    seen = []
    next_layer = solver_model.model.model.language_model.layers[2]
    handle = next_layer.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
    try:
        run()
    finally:
        handle.remove()
    return seen[0]


def test_edit_hook_changes_only_the_final_prompt_token_of_the_prefill():
    solver_model, ep = fakes.solver_model(layer=1), fakes.episode()
    _, prefill = generate_baseline(solver_model, ep)
    x = prefill.final_activation
    x_new = x + 1.0
    plain = _next_layer_prefill_input(solver_model, lambda: generate_baseline(solver_model, ep))
    edited = _next_layer_prefill_input(
        solver_model, lambda: generate_with_activation(solver_model, ep, x, x_new, prefill=None))
    assert torch.equal(edited[0, -1], x_new) and torch.equal(edited[0, :-1], plain[0, :-1])
    hook = ReplaceFinalPromptActivation(solver_model.layer_module, x, x_new)
    with hook:
        solver_model.solver.choose(ep.question, ep.paragraphs, ep.solver_attempt_1, ep.critic_advocated_answer,
                                   ep.candidate_order)
    assert hook.calls > 1                                   # it saw the decode steps too, and left them alone


def test_edit_hook_refuses_an_activation_it_was_not_computed_from():
    solver_model, ep = fakes.solver_model(), fakes.episode()
    _, prefill = generate_baseline(solver_model, ep)
    x = prefill.final_activation
    with pytest.raises(RuntimeError, match="differs from the one the edit was computed from"):
        generate_with_activation(solver_model, ep, x + 1e-3, x, prefill=None)


def test_a_worker_whose_baseline_differs_from_prepares_stops():
    solver_model, ep = fakes.solver_model(), fakes.episode()
    baseline, _ = generate_baseline(solver_model, ep)
    prepared = {"prompt_sha256": baseline.prompt_sha256, "answer": baseline.answer,
                "first_token_id": baseline.first_token_id}
    require_same_baseline(baseline, prepared, ep)
    with pytest.raises(RuntimeError, match="differs from prepare's"):
        require_same_baseline(baseline, {**prepared, "answer": baseline.answer + "!"}, ep)


def test_saes_sharing_an_encoder_are_encoded_once(monkeypatch):
    reps = fakes.representations()
    calls = []
    for name in ("SAE", "SAE decoder"):
        sae = reps[name].sae
        real = sae.encoder
        monkeypatch.setattr(sae, "encoder", lambda x, real=real: calls.append(1) or real(x))
    codes = encode_once(torch.randn(fakes.HIDDEN), reps)
    assert len(calls) == 1 and set(codes) == {"SAE", "SAE decoder"}
    assert torch.equal(codes["SAE"][0], codes["SAE decoder"][0])                 # same z
    assert not torch.equal(codes["SAE"][1], codes["SAE decoder"][1])             # different reconstructions
