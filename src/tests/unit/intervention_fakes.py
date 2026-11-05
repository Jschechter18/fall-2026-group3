"""Tiny CPU stand-ins for the intervention tests: a random Gemma 3, a byte-level processor, small SAEs.

The model has the real Gemma 3 architecture (so the layer path, hook and ``generate`` loop are the
real ones) with a 16-dimensional residual stream. Its outputs are meaningless.
"""

import torch
from transformers import BatchEncoding, Gemma3Config, Gemma3ForConditionalGeneration

from mas_sae.agents.solver import Solver
from mas_sae.intervention.episodes import Episode
from mas_sae.intervention.representations import build_representation
from mas_sae.intervention.config import Representation as RepresentationSpec
from mas_sae.intervention.solver import SolverModel
from mas_sae.sae.pretrained import JumpReLUSAE

HIDDEN, WIDTH, N_FEATURES, N_DRAWS = 16, 40, 3, 2


class ByteProcessor:
    """Prompt bytes as token ids (the last 200 bytes, to keep the CPU prefill fast)."""

    def apply_chat_template(self, messages, **kwargs):
        content = messages[0]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        ids = torch.tensor([list(text.encode("utf-8"))[-200:]], dtype=torch.long)
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, token_ids, skip_special_tokens=False):
        return bytes(int(t) % 256 for t in token_ids).decode("utf-8", errors="ignore")


def tiny_gemma(layers=3, seed=0):
    cfg = Gemma3Config(
        text_config=dict(vocab_size=256, hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=layers,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, sliding_window=64,
                         max_position_embeddings=1024),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                           image_size=28, patch_size=14),
        mm_tokens_per_image=4)
    torch.manual_seed(seed)
    return Gemma3ForConditionalGeneration(cfg).eval()


def solver_model(layer=1, layers=3, max_new_tokens=6):
    model = tiny_gemma(layers)
    return SolverModel(Solver(model, ByteProcessor(), max_new_tokens=max_new_tokens), model,
                       f"model.language_model.layers.{layer}", {"id": "tiny"})


def episode(i=0, stored_label="adoption", stored_a2="x"):
    return Episode(episode_id=f"ep{i}", question_id=f"q{i}", question=f"Question {i}?",
                   paragraphs=[{"idx": 0, "title": "T", "paragraph_text": f"Some text {i}."}],
                   solver_attempt_1="Paris", critic_advocated_answer="Rome", candidate_order="a1_first",
                   stored_solver_attempt_2=stored_a2, stored_label=stored_label, stored_reason="stored reason",
                   stored_prompt_sha256=None)


def jump_relu(seed=0, w_dec_seed=None):
    g = torch.Generator().manual_seed(seed)
    w_enc, b_enc = torch.randn(HIDDEN, WIDTH, generator=g), torch.zeros(WIDTH)
    w_dec = torch.randn(WIDTH, HIDDEN, generator=torch.Generator().manual_seed(w_dec_seed or seed + 100))
    return JumpReLUSAE(w_enc, b_enc, w_dec, torch.zeros(HIDDEN), torch.full((WIDTH,), 0.1)).eval()


def feature_entry(width):
    """A feature_sets.json entry (ids < 16, so it fits raw neurons too): disjoint targeted sets, draws avoid both."""
    return {"adoption_features": [{"feature": f, "coef_toward_accept": 0.5} for f in range(0, N_FEATURES)],
            "retention_features": [{"feature": f, "coef_toward_accept": -0.5} for f in range(3, 3 + N_FEATURES)],
            "random_draws_adoption": [[6, 7, 8], [9, 10, 11]],
            "random_draws_retention": [[12, 13, 14], [15, 7, 9]]}


def representations():
    """Raw neurons, an SAE, and a second SAE with the same encoder but another decoder."""
    spec = lambda kind, role: RepresentationSpec(kind=kind, path="unused", role=role)
    return {r.name: r for r in [
        build_representation("Raw", spec("activations", "control"), feature_entry(HIDDEN), None, HIDDEN, N_DRAWS),
        build_representation("SAE", spec("gemma_scope", "primary"), feature_entry(WIDTH), jump_relu(0), WIDTH, N_DRAWS),
        build_representation("SAE decoder", spec("gemma_scope", "primary"), feature_entry(WIDTH),
                             jump_relu(0, w_dec_seed=7), WIDTH, N_DRAWS),
    ]}


# ---- a complete set of input files, for the preflight and episode tests (paths relative to the cwd)

def write_workspace(root, n_episodes=8, presentation="candidates"):
    """Write every input a config points at under ``root`` (the cwd) and return the config mapping."""
    import json
    import numpy as np
    from safetensors.torch import save_file
    from mas_sae.evaluation import claude_validator as cv
    from mas_sae.experiments.artifacts import sha256_file

    root.joinpath("in", "labels").mkdir(parents=True)
    labels = ["adoption", "retention", "agreement", "out"]
    episodes = [{"episode_id": f"ep{i}", "question_id": f"q{i}", "question": f"Question {i}?", "partition": "intervention",
                 "critic_condition": "natural", "a2_presentation": presentation, "candidate_order": "a1_first",
                 "solver_attempt_1": "Paris", "critic_advocated_answer": "Rome", "solver_attempt_2": "Paris",
                 "generation_telemetry": {"solver_a2_prompt": {"input_ids_sha256": f"sha{i}"}}}
                for i in range(n_episodes)]
    write = lambda path, rows: path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write(root / "in/interactions.jsonl", episodes)
    write(root / "in/contexts.jsonl", [{"question_id": e["question_id"], "payload": {
        "question": e["question"], "paragraphs": [{"idx": 0, "title": "T", "paragraph_text": "text"}]}} for e in episodes])
    write(root / "in/labels/raw_results.jsonl", [{"episode_id": e["episode_id"], "behavior_label": labels[i % 4],
                                                  "claude_reason": "r"} for i, e in enumerate(episodes)])
    (root / "in/labels/run_settings.json").write_text(json.dumps({
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "client": "ClaudeClient", "max_tokens": 512,
        "temperature": 0.0, "prompt_version": cv.prompt_version(True, "candidates"),
        "prompt_sha256": cv.prompt_sha256(True, "candidates"), "max_attempts": 3, "partition": "intervention",
        "episodes_sha256": sha256_file(root / "in/interactions.jsonl")}))
    (root / "in/labels/summary.json").write_text(json.dumps({"models_returned": ["claude-sonnet-4-5-20250929"]}))

    np.save(root / "in/center.npy", np.zeros(HIDDEN, np.float32))
    sae_dir = root / "in/sae"
    sae_dir.mkdir()
    (sae_dir / "config.json").write_text(json.dumps({"architecture": "jump_relu",
                                                     "hf_hook_point_in": "model.layers.25.output"}))
    sae = jump_relu(0)
    save_file({k: v.contiguous() for k, v in sae.state_dict().items()}, str(sae_dir / "params.safetensors"))
    entry = lambda path, files: {"path": path, "sha256": {f: sha256_file(f) for f in files}, **feature_entry(HIDDEN)}
    (root / "in/feature_sets.json").write_text(json.dumps({
        "Raw": entry("in/center.npy", ["in/center.npy"]),
        "SAE": entry("in/sae", ["in/sae/config.json", "in/sae/params.safetensors"])}))
    return {
        "layer": 25, "partition": "intervention",
        "inputs": {"interactions": "in/interactions.jsonl", "contexts": "in/contexts.jsonl",
                   "stored_labels": "in/labels", "feature_sets": "in/feature_sets.json"},
        "solver": {"id": "org/model", "revision": "abc123", "loader": "gemma3", "dtype": "bfloat16",
                   "device": "cuda:0", "cache_dir": "cache", "generation": {"max_new_tokens": 8, "do_sample": False}},
        "representations": {"Raw": {"kind": "activations", "path": "in/center.npy", "role": "control"},
                            "SAE": {"kind": "gemma_scope", "path": "in/sae", "role": "primary"}},
        "experiment": {"number_of_episodes": 2, "number_of_pilot_episodes": 1, "number_of_shards": 1,
                       "edit_size": 0.05, "number_of_targeted_features": N_FEATURES, "number_of_random_draws": N_DRAWS},
        "analysis": {"primary_representations": ["SAE"], "between_sae_contrasts": [], "exploratory_contrasts": [],
                     "direction_contrasts": [["both", "decrease"]], "n_bootstrap": 10, "n_permutations": 10},
        "machine": {"min_gpu_memory_gb": 1, "min_free_disk_gb": 0},
        "output_dir": "out",
    }
