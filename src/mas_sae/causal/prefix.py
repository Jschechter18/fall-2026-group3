"""Reuse the prefill below a final-prompt-token splice across one episode's conditions (#214).

A splice at layer L edits only layer L's output at the final prompt token, so layers 0..L of the
prefill compute the same keys, values and layer-L output in every condition of an episode.
``PrefixCache.capture`` records them during one plain generation. Inside ``replay``, the prefill of a
generation of the same prompt makes each of those layers' recorded cache update instead of computing
it, and layer L returns its recorded output, which forward hooks (the splice) then edit as before.
Layers above L run on the same inputs with the same shapes, and every decode step runs unchanged, so
generation is bitwise the full computation's whenever the model is deterministic on its device.
"""

from __future__ import annotations

import contextlib

import torch


class PrefixCache:
    """Layers 0..L of one prompt's prefill, for the decoder layer ``site`` (``<layer list>.<L>``)."""

    def __init__(self, model: torch.nn.Module, site: str):
        parent, _, index = site.rpartition(".")
        layers = dict(model.named_modules()).get(parent)
        if not isinstance(layers, torch.nn.ModuleList) or not index.isdigit() or int(index) >= len(layers):
            raise ValueError(f"{site!r} is not a layer of a decoder layer list.")
        self.layers = list(layers)[: int(index) + 1]
        self.updates: list[tuple] | None = None        # each layer's cache update arguments
        self.output: torch.Tensor | None = None        # layer L's output, before any hook

    @property
    def final_activation(self) -> torch.Tensor:
        """Layer L's output at the final prompt token, (1, d): what the splice edits."""
        if self.output is None:
            raise RuntimeError("Nothing was captured.")
        return self.output[:, -1, :]

    @contextlib.contextmanager
    def capture(self):
        """Record the prefill of the one generation run inside this block."""
        updates, outputs = {}, {}

        def wrap(i, real):
            def forward(hidden_states, *args, **kwargs):
                if i in outputs:                                     # a decode step
                    return real(hidden_states, *args, **kwargs)
                if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[1] < 2:
                    raise ValueError(f"Expected a single-prompt prefill, got {tuple(hidden_states.shape)}.")
                cache = kwargs.get("past_key_values")
                if cache is None:
                    raise ValueError("The prefill ran without a cache; nothing can be reused.")
                update = cache.update

                def record(*a, **kw):
                    if i in updates:
                        raise RuntimeError(f"Layer {i} updated the cache twice in one forward pass.")
                    updates[i] = (a, kw)
                    return update(*a, **kw)

                cache.update = record
                try:
                    out = real(hidden_states, *args, **kwargs)
                finally:
                    del cache.update
                outputs[i] = out
                return out
            return forward

        with self._patched(wrap):
            yield self
        n = len(self.layers)
        if set(updates) != set(range(n)) or set(outputs) != set(range(n)):
            raise RuntimeError("Capture did not see one prefill through every layer up to the splice.")
        if not isinstance(outputs[n - 1], torch.Tensor):
            raise TypeError("Decoder layers must return the hidden state tensor.")
        self.updates, self.output = [updates[i] for i in range(n)], outputs[n - 1]

    @contextlib.contextmanager
    def replay(self):
        """Replay the captured layers in the prefill of the one generation run inside this block.

        The recorded key/value tensors are passed to the new cache as they are: every cache layer
        copies or concatenates them, never writes to them.
        """
        if self.updates is None:
            raise RuntimeError("Capture a plain generation before replaying it.")
        used, last = set(), len(self.layers) - 1

        def wrap(i, real):
            def forward(hidden_states, *args, **kwargs):
                if i in used:                                        # a decode step
                    return real(hidden_states, *args, **kwargs)
                if hidden_states.shape[:2] != self.output.shape[:2]:
                    raise ValueError("This prefill is not the captured prompt.")
                used.add(i)
                a, kw = self.updates[i]
                kwargs["past_key_values"].update(*a, **kw)
                # below L the output is never read: layer L returns the captured one
                return self.output.clone() if i == last else hidden_states
            return forward

        with self._patched(wrap):
            yield self
        if len(used) != len(self.layers):
            raise RuntimeError("Replay did not run a prefill through every captured layer.")

    @contextlib.contextmanager
    def _patched(self, wrap):
        """Route each captured layer's ``forward`` through ``wrap(i, forward)``; hooks still run around it."""
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
