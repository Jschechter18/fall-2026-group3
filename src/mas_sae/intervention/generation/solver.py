"""The Solver: load it, generate A2, and edit layer 25 at the final prompt token of the prefill only.

A2 is produced exactly as in collection, by ``Solver.choose(question, paragraphs, A1, C1,
candidate_order)`` with the collection's model, revision, chat template and greedy settings; every
generation's prompt token hash is recorded, and the baseline check requires it to equal collection's.

The edit is a forward hook on the layer: on its first call (the prefill) it replaces the output at
the final prompt token, and it does nothing on decode steps. Earlier positions are never edited.
Below the last layer the edited token's keys and values still reach every decode step, so the answer
can change even when the first token does not.

Caching: layers 0..25 of the prefill are the same in every condition of an episode, since the edit
happens at layer 25's output. ``PrefillCache`` records them during the episode's baseline generation
and replays them in every other condition, which then computes only layers 26+ and the decode steps.
Generation is otherwise unchanged, so rows equal the uncached computation's (``tests/unit/intervention/test_solver.py``).
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from mas_sae.intervention.experiment.config import InterventionConfig
    from mas_sae.intervention.experiment.episodes import Episode


@dataclass
class SolverModel:
    """The Solver as collection configured it, its model, and the module name of the edited layer."""
    solver: object          # mas_sae.agents.solver.Solver, configured as in collection
    model: torch.nn.Module
    site: str               # module name of the edited layer, e.g. model.language_model.layers.25
    spec: dict              # the resolved Solver spec

    @property
    def layer_module(self) -> torch.nn.Module:
        """The module at ``site``: the layer whose output is edited."""
        return dict(self.model.named_modules())[self.site]


def load_solver(config: InterventionConfig) -> SolverModel:
    """Load the Solver model and agent from ``config.solver`` (``max_new_tokens`` defaults to 64 if
    the spec does not set it) and find the module name of layer ``config.layer``."""
    from mas_sae.activations.sites import resolve_solver_sites
    from mas_sae.agents.solver import Solver
    from mas_sae.models.roles import configure_agent, load_spec, resolve_role_spec

    spec = resolve_role_spec("solver", config.solver, default_max_new_tokens=64)
    model, processor = load_spec(spec)
    solver = Solver(model, processor)
    configure_agent(solver, spec)
    site = resolve_solver_sites(model, spec["loader"], [config.layer])[0]
    return SolverModel(solver, model, site, spec)


@dataclass
class Generation:
    """One generated A2 and what was recorded with it: prompt hash, generated token ids, the
    first-token log-probabilities, the edit size as applied (None without an edit) and wall time."""
    answer: str
    prompt_sha256: str | None
    generated_token_ids: list[int]
    reached_token_budget: bool
    first_token_logprobs: Tensor          # log p over the vocabulary for the first generated token
    applied_relative_size: float | None   # |x' − x| / |x| as written into the model (after the dtype cast)
    seconds: float

    @property
    def first_token_id(self) -> int | None:
        """The first generated token id; None if nothing was generated."""
        return self.generated_token_ids[0] if self.generated_token_ids else None

    def logprob_of(self, token_id: int | None) -> float | None:
        """log p of ``token_id`` as the first generated token; None if ``token_id`` is None."""
        return None if token_id is None else float(self.first_token_logprobs[token_id])


class ReplaceFinalPromptActivation:
    """Forward hook: on the prefill only, replace the layer's output at the final prompt token.

    ``expected`` is the activation the replacement was computed from; the hook refuses to edit any
    other one, so a replacement can never be applied to a different prompt or a recomputed activation.
    """

    def __init__(self, module: torch.nn.Module, expected: Tensor, replacement: Tensor):
        self.module, self.expected, self.replacement = module, expected, replacement
        self.calls, self.applied_relative_size, self._handle = 0, None, None

    def _hook(self, module, inputs, output):
        self.calls += 1
        if self.calls > 1:                                   # a decode step
            return None
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 2:
            raise ValueError(f"Expected a single-prompt prefill (1, seq>1, d), got {tuple(hidden.shape)}.")
        x = hidden[0, -1]
        if not torch.equal(x, self.expected.to(x)):
            raise RuntimeError("The final prompt activation differs from the one the edit was computed from.")
        new = self.replacement.to(device=x.device, dtype=hidden.dtype)
        self.applied_relative_size = float((new.float() - x.float()).norm() / x.float().norm())
        hidden = hidden.clone()
        hidden[0, -1] = new
        return (hidden, *output[1:]) if isinstance(output, (tuple, list)) else hidden

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        if exc[0] is None and self.calls == 0:
            raise RuntimeError("The edit hook never ran: the layer is not on the generation path.")


class FirstTokenLogprobs:
    """The model's log-probabilities for the first generated token, before any logits processing."""

    def __init__(self, model: torch.nn.Module):
        self.model, self.logprobs, self._handle = model, None, None

    def _hook(self, module, inputs, output):
        if self.logprobs is None:
            logits = output.logits if hasattr(output, "logits") else output[0]
            self.logprobs = logits[0, -1].float().log_softmax(-1)

    def __enter__(self):
        self._handle = self.model.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()


class PrefillCache:
    """Layers 0..L of one prompt's prefill, recorded once and replayed in later generations.

    ``capture`` wraps each layer up to L during one plain generation: it records the key/value cache
    update each layer makes and layer L's output. In ``replay``, the prefill of a generation of the
    same prompt makes those recorded cache updates instead of computing the layers, and layer L returns
    its recorded output, which forward hooks (the edit) then see as usual. Decode steps are unchanged.
    """

    def __init__(self, model: torch.nn.Module, site: str):
        parent, _, index = site.rpartition(".")
        layers = dict(model.named_modules()).get(parent)
        if not isinstance(layers, torch.nn.ModuleList) or not index.isdigit() or int(index) >= len(layers):
            raise ValueError(f"{site!r} is not a layer of a decoder layer list.")
        self.layers = list(layers)[: int(index) + 1]
        self.cache_updates: list | None = None
        self.output: Tensor | None = None                   # layer L's prefill output, before any hook

    @property
    def final_activation(self) -> Tensor:
        """Layer L's output at the final prompt token, shape [d]: the x every edit starts from."""
        return self.output[0, -1]

    @contextlib.contextmanager
    def capture(self) -> Iterator[PrefillCache]:
        """Context manager around one plain generation: its prefill runs normally while each layer
        up to L records its key/value cache update and its output. On exit, raises RuntimeError
        unless the prefill reached every one of those layers, then keeps them for ``replay``."""
        updates, outputs = {}, {}

        def wrap(i, forward):
            def capturing_forward(hidden_states, *args, **kwargs):
                if i in outputs:                             # a decode step
                    return forward(hidden_states, *args, **kwargs)
                cache = kwargs.get("past_key_values")
                if cache is None:
                    raise ValueError("The prefill ran without a key/value cache.")
                real_update = cache.update

                def recording_update(*a, **kw):
                    updates[i] = (a, kw)
                    return real_update(*a, **kw)

                cache.update = recording_update
                try:
                    outputs[i] = forward(hidden_states, *args, **kwargs)
                finally:
                    del cache.update
                return outputs[i]
            return capturing_forward

        with self._wrapped_layers(wrap):
            yield self
        n = len(self.layers)
        if set(updates) != set(range(n)) or set(outputs) != set(range(n)):
            raise RuntimeError("The capture did not see one prefill through every layer up to the edit.")
        self.cache_updates, self.output = [updates[i] for i in range(n)], outputs[n - 1]

    @contextlib.contextmanager
    def replay(self) -> Iterator[PrefillCache]:
        """Context manager around a later generation of the captured prompt: its prefill makes the
        recorded cache updates instead of computing layers 0..L, and layer L returns its recorded
        output (forward hooks on it still run). Raises RuntimeError if nothing was captured or a
        captured layer was skipped, ValueError if the prefill's shape is not the captured prompt's."""
        if self.output is None:
            raise RuntimeError("Capture a generation before replaying it.")
        replayed, last = set(), len(self.layers) - 1

        def wrap(i, forward):
            def replaying_forward(hidden_states, *args, **kwargs):
                if i in replayed:                            # a decode step
                    return forward(hidden_states, *args, **kwargs)
                if hidden_states.shape[:2] != self.output.shape[:2]:
                    raise ValueError("This prefill is not the captured prompt.")
                replayed.add(i)
                a, kw = self.cache_updates[i]
                kwargs["past_key_values"].update(*a, **kw)
                # layers below L return their input unchanged: only layer L's output is ever read
                return self.output.clone() if i == last else hidden_states
            return replaying_forward

        with self._wrapped_layers(wrap):
            yield self
        if len(replayed) != len(self.layers):
            raise RuntimeError("The replay did not run a prefill through every captured layer.")

    @contextlib.contextmanager
    def _wrapped_layers(self, wrap):
        saved = []
        try:
            for i, layer in enumerate(self.layers):
                saved.append((layer, layer.__dict__.get("forward")))
                layer.forward = wrap(i, layer.forward)
            yield
        finally:
            for layer, own in saved:
                if own is None:
                    del layer.forward
                else:
                    layer.forward = own


def _choose(solver_model: SolverModel, episode) -> str:
    return solver_model.solver.choose(episode.question, episode.paragraphs, episode.solver_attempt_1,
                                      episode.critic_advocated_answer, episode.candidate_order)


def _generation(solver_model: SolverModel, answer: str, logprobs: FirstTokenLogprobs,
                applied: float | None, seconds: float) -> Generation:
    solver = solver_model.solver
    return Generation(answer=answer, prompt_sha256=(solver.last_prompt or {}).get("input_ids_sha256"),
                      generated_token_ids=list(solver.last_generated_token_ids or []),
                      reached_token_budget=bool((solver.last_generation or {}).get("reached_token_budget")),
                      first_token_logprobs=logprobs.logprobs, applied_relative_size=applied, seconds=seconds)


def generate_baseline(solver_model: SolverModel, episode: Episode) -> tuple[Generation, PrefillCache]:
    """The plain A2 (no SAE, no edit), recording layers 0..L of its prefill for the episode's other conditions."""
    prefill = PrefillCache(solver_model.model, solver_model.site)
    start = time.perf_counter()
    with prefill.capture(), FirstTokenLogprobs(solver_model.model) as logprobs:
        answer = _choose(solver_model, episode)
    return _generation(solver_model, answer, logprobs, None, time.perf_counter() - start), prefill


def generate_with_activation(solver_model: SolverModel, episode: Episode, x: Tensor, x_new: Tensor,
                             prefill: PrefillCache | None) -> Generation:
    """A2 with the final prompt activation ``x`` replaced by ``x_new``.

    With ``prefill`` the cached layers are replayed; with None the full prefill is computed (the
    plain path, which the parity test compares against)."""
    start = time.perf_counter()
    replay = prefill.replay() if prefill is not None else contextlib.nullcontext()
    edit = ReplaceFinalPromptActivation(solver_model.layer_module, x, x_new)
    with replay, edit, FirstTokenLogprobs(solver_model.model) as logprobs:
        answer = _choose(solver_model, episode)
    return _generation(solver_model, answer, logprobs, edit.applied_relative_size, time.perf_counter() - start)
