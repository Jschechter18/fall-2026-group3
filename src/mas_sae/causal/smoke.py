"""Tiny stand-ins for a CPU smoke test of the causal replay (issue #90).

``smoke_components`` builds a randomly initialised Gemma 3 with the real architecture, deep enough
for the requested layer (so the module path, hook and ``generate`` loop are the real ones) but with
a 16-dimensional residual stream and a byte-level processor; the caller passes the layer, the SAEs
and their feature sets (``tiny_saes`` and ``tiny_features`` make random ones). Everything else (episode
loading, labeling, resumable output, the table) is the real pipeline. The numbers it produces are
meaningless: it checks that the replay runs end to end and writes the expected files and columns.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import BatchEncoding, Gemma3Config, Gemma3ForConditionalGeneration

from mas_sae.activations.sites import resolve_solver_sites
from mas_sae.agents.solver import Solver
from mas_sae.causal.replay import Components
from mas_sae.sae.pretrained import JumpReLUSAE
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder

HIDDEN = 16


class ByteProcessor:
    """Prompt bytes as token ids (last ``max_tokens`` bytes only, to keep CPU prefill fast)."""

    def __init__(self, max_tokens: int = 384):
        self.max_tokens = max_tokens

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        ids = torch.tensor([list(text.encode("utf-8"))[-self.max_tokens:]], dtype=torch.long)
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return bytes(int(t) % 256 for t in token_ids).decode("utf-8", errors="ignore")


def tiny_gemma(layers: int, seed: int = 0) -> Gemma3ForConditionalGeneration:
    cfg = Gemma3Config(
        text_config=dict(vocab_size=256, hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=layers,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, sliding_window=64,
                         max_position_embeddings=1024),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                           image_size=28, patch_size=14),
        mm_tokens_per_image=4)
    torch.manual_seed(seed)
    return Gemma3ForConditionalGeneration(cfg).eval()


def tiny_saes(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    return {
        "TopK SAE": SparseAutoencoder(HIDDEN, 64, 64, sparsity_mode="topk", top_k=8).eval(),
        "L1 SAE": SparseAutoencoder(HIDDEN, 64, 64, sparsity_mode="l1").eval(),
        "Pretrained SAE": JumpReLUSAE(torch.randn(HIDDEN, 64, generator=g), torch.zeros(64),
                                      torch.randn(64, HIDDEN, generator=g) / 8, torch.zeros(HIDDEN),
                                      torch.zeros(64)).eval(),
    }


def tiny_features(saes: dict, n_targeted: int = 3, n_random_draws: int = 2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, sae in saes.items():
        perm = rng.permutation(sae.latent_dim)
        targeted = perm[:n_targeted].tolist()
        rest = perm[n_targeted:]
        out[name] = {"targeted": targeted,
                     "random": [rng.choice(rest, n_targeted, replace=False).tolist() for _ in range(n_random_draws)]}
    return out


def smoke_components(paragraphs: dict, layer: int, saes: dict, features: dict,
                     max_new_tokens: int = 4) -> Components:
    """Components for a tiny Gemma 3 with ``layer`` as its last layer, spliced at ``layer``."""
    model = tiny_gemma(layers=layer + 1)
    solver = Solver(model, ByteProcessor(), max_new_tokens=max_new_tokens)
    site = resolve_solver_sites(model, "gemma3", [layer])[0]
    return Components(solver=solver, model=model, site=site, saes=saes, features=features, paragraphs=paragraphs,
                      info={"smoke": "tiny random Gemma 3 + random SAEs; outputs check the pipeline only"})
