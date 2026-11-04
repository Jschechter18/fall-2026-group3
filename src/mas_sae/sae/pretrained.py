"""Pretrained Gemma Scope 2 SAEs, exposed through the same encoder/decoder interface as
``SparseAutoencoder`` so probes and interventions treat both kinds of SAE identically.

Download one SAE folder (``config.json`` + ``params.safetensors``) from
https://huggingface.co/google/gemma-scope-2-4b-it, e.g.
``resid_post_all/layer_33_width_16k_l0_big``. Its ``hf_hook_point_in`` must be the block output
our activations were captured from (``model.layers.<N>.output``). Gemma Scope SAEs take raw
residual-stream activations: there is no input normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn


def _window(pre: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
    """Rectangle kernel: 1 where |pre - threshold| < bandwidth / 2."""
    return ((pre - threshold).abs() < bandwidth / 2).to(pre.dtype)


class _JumpReLUSTE(torch.autograd.Function):
    """pre * H(pre - threshold), with the pseudo-derivative of Rajamanoharan et al. (2024) for threshold."""

    @staticmethod
    def forward(ctx, pre, threshold, bandwidth):
        ctx.save_for_backward(pre, threshold)
        ctx.bandwidth = bandwidth
        return pre * (pre > threshold)

    @staticmethod
    def backward(ctx, grad):
        pre, threshold = ctx.saved_tensors
        grad_pre = grad * (pre > threshold)
        grad_threshold = -(threshold / ctx.bandwidth) * _window(pre, threshold, ctx.bandwidth) * grad
        return grad_pre, grad_threshold.sum(dim=0), None


class _StepSTE(torch.autograd.Function):
    """H(pre - threshold): no gradient to pre, kernel pseudo-derivative to threshold (for the L0 penalty)."""

    @staticmethod
    def forward(ctx, pre, threshold, bandwidth):
        ctx.save_for_backward(pre, threshold)
        ctx.bandwidth = bandwidth
        return (pre > threshold).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad):
        pre, threshold = ctx.saved_tensors
        grad_threshold = -(1 / ctx.bandwidth) * _window(pre, threshold, ctx.bandwidth) * grad
        return torch.zeros_like(pre), grad_threshold.sum(dim=0), None


class JumpReLUSAE(nn.Module):
    """Gemma Scope JumpReLU SAE: z = relu(pre) * (pre > threshold), pre = x @ W_enc + b_enc."""

    def __init__(self, w_enc: Tensor, b_enc: Tensor, w_dec: Tensor, b_dec: Tensor, threshold: Tensor):
        super().__init__()
        if w_enc.shape != w_dec.T.shape or b_enc.shape != threshold.shape:
            raise ValueError("Inconsistent JumpReLU SAE parameter shapes.")
        self.input_dim, self.latent_dim = w_enc.shape
        self.sparsity_mode = "jump_relu"
        self.w_enc = nn.Parameter(w_enc, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.w_dec = nn.Parameter(w_dec, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.threshold = nn.Parameter(threshold, requires_grad=False)

    def encoder(self, activation: Tensor) -> Tensor:
        pre = activation @ self.w_enc + self.b_enc
        # relu as well as the gate, so a negative threshold can never let a negative latent through
        return torch.relu(pre) * (pre > self.threshold)

    def encoder_ste(self, activation: Tensor, bandwidth: float,
                    threshold: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Training-only encoder. Returns (features, per-example L0) with straight-through gradients.

        The forward values equal ``encoder`` whenever the threshold is positive. ``threshold``
        overrides ``self.threshold`` so a caller can train it in log space.
        """
        threshold = self.threshold if threshold is None else threshold
        pre = activation @ self.w_enc + self.b_enc
        features = _JumpReLUSTE.apply(pre, threshold, bandwidth)
        return features, _StepSTE.apply(pre, threshold, bandwidth).sum(dim=1)

    def decoder(self, sparse_features: Tensor) -> Tensor:
        return sparse_features @ self.w_dec + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        sparse_features = self.encoder(x)
        return sparse_features, self.decoder(sparse_features)


def load_gemma_scope_sae(sae_dir: str | Path, expected_layer: int | None = None) -> tuple[JumpReLUSAE, dict]:
    """Load a downloaded Gemma Scope SAE folder. Returns (sae, config)."""
    from safetensors.torch import load_file

    sae_dir = Path(sae_dir)
    config = json.loads((sae_dir / "config.json").read_text())
    if config.get("architecture") != "jump_relu":
        raise ValueError(f"Expected a jump_relu SAE, got {config.get('architecture')!r}.")
    if expected_layer is not None and config.get("hf_hook_point_in") != f"model.layers.{expected_layer}.output":
        raise ValueError(f"SAE hook point {config.get('hf_hook_point_in')!r} is not layer {expected_layer}'s "
                         "block output, which is where our activations were captured.")
    p = load_file(sae_dir / "params.safetensors")
    sae = JumpReLUSAE(p["w_enc"], p["b_enc"], p["w_dec"], p["b_dec"], p["threshold"]).eval()
    return sae, config
