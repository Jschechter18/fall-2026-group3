"""The original Pretrained SAE, and saving an edited copy of it (only the decoder rows differ).

Shared by the LLM-edit harness and the controls (mechanical and random edits): each saves the
original SAE with an edited decoder in the Gemma Scope layout (params.safetensors + config.json),
which stages 5 and 6 load as `kind: gemma_scope`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from mas_sae.intervention.experiment.representations import read_feature_sets
from mas_sae.sae.finetuning.loop import save_model
from mas_sae.sae.pretrained import JumpReLUSAE

SAE_NAME = "Pretrained SAE"
SAE_DIR = Path("results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big")


def original_sae(feature_sets_path: Path, sae_dir: Path = SAE_DIR):
    """(w_dec, config.json as a dict, selected features) of the Pretrained SAE.

    Refuses if feature_sets.json's Pretrained SAE is not this folder."""
    entry = read_feature_sets(Path(feature_sets_path))[SAE_NAME]
    if entry["path"] != str(sae_dir):
        raise SystemExit(f"feature_sets.json's {SAE_NAME} is {entry['path']}, not {sae_dir}")
    selected = {label: [f["feature"] for f in entry[f"{label}_features"]] for label in ("adoption", "retention")}
    w_dec = load_file(str(Path(sae_dir) / "params.safetensors"))["w_dec"]
    config = json.loads((Path(sae_dir) / "config.json").read_text())
    return w_dec, config, selected


def jump_relu_sae(w_dec: np.ndarray, sae_dir: Path = SAE_DIR) -> JumpReLUSAE:
    """The original SAE's encoder, threshold and decoder bias, with the given decoder."""
    p = load_file(str(Path(sae_dir) / "params.safetensors"))
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return JumpReLUSAE(tensor(p["w_enc"]), tensor(p["b_enc"]), tensor(w_dec), tensor(p["b_dec"]),
                       tensor(p["threshold"])).eval()


def save_edited_sae(w_dec: np.ndarray, sae_config: dict, destination: Path, note: dict,
                    sae_dir: Path = SAE_DIR) -> Path:
    """Save the original SAE with decoder `w_dec`; `note` (where the edit came from) goes into config.json."""
    save_model(jump_relu_sae(w_dec, sae_dir), {**sae_config, "edited_from": str(sae_dir), **note}, Path(destination))
    return Path(destination)
