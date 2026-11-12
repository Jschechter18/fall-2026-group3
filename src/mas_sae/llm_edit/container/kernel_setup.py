"""Runs once inside the sandbox's Jupyter kernel, before the LLM's first snippet (and after a restart).

Self-contained on purpose: the container has no network and no repository, so mas_sae is not
importable here. It installs the no-training guards, loads the read-only inputs, and defines the
helpers the LLM may call. Paths are the container's mount points (see mas_sae.llm_edit.sandbox).

Guards (they stop accidental training, not every hand-written optimization loop; the snippet time
limit, the steer_test budget and the transcript audit cover the rest):
  - autograd off: torch.set_grad_enabled(False), and every way back on raises;
  - Tensor.backward and torch.autograd.grad raise;
  - torch.optim, scipy.optimize and sklearn cannot be imported (a sys.meta_path finder), and any
    copy already imported (`import torch` imports torch.optim itself) is purged.

Edits: apply_edit checks the rules, updates this kernel's decoder, and appends the edit to
/outbox/edits.jsonl. The host re-checks every edit against the original SAE; only its record counts.
"""

import importlib.abc
import json
import sys
from pathlib import Path


def _install_guards():
    import torch

    blocked = ("torch.optim", "scipy.optimize", "sklearn")
    is_blocked = lambda name: any(name == b or name.startswith(b + ".") for b in blocked)

    class _BlockTraining(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if is_blocked(name):
                raise ImportError(f"{name} is blocked in this sandbox: no training, fitting or optimizers.")
            return None

    sys.meta_path.insert(0, _BlockTraining())
    for name in [m for m in sys.modules if is_blocked(m)]:
        del sys.modules[name]
    if hasattr(torch, "optim"):
        delattr(torch, "optim")

    torch.set_grad_enabled(False)

    def _refuse(*args, **kwargs):
        raise RuntimeError("Gradients are disabled in this sandbox: no training or optimization.")

    torch.Tensor.backward = _refuse
    torch.autograd.backward = _refuse
    torch.autograd.grad = _refuse
    torch.enable_grad = _refuse
    original_set_grad_enabled = torch.set_grad_enabled

    def _set_grad_enabled(mode):
        if mode:
            _refuse()
        return original_set_grad_enabled(False)

    torch.set_grad_enabled = _set_grad_enabled


_install_guards()

import numpy as np
import pandas as pd
import scipy.sparse
from safetensors.numpy import load_file

for _name in [m for m in sys.modules if m.startswith(("scipy.optimize", "sklearn"))]:
    del sys.modules[_name]   # in case a library above imported one before the finder could stop it

INPUTS, SAE_DIR, OUTBOX = Path("/inputs"), Path("/sae"), Path("/outbox")
CHUNK_ROWS = 2048             # helpers work in chunks to stay well inside the container's memory limit


def _read_only(array):
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


# ---------------------------------------------------------------------
# Inputs (read-only)
# ---------------------------------------------------------------------
_params = load_file(str(SAE_DIR / "params.safetensors"))
_rules = json.loads((INPUTS / "rules.json").read_text())

W_enc = _read_only(_params["w_enc"])          # input × latent
b_enc = _read_only(_params["b_enc"])
threshold = _read_only(_params["threshold"])
b_dec = _read_only(_params["b_dec"])
_ORIGINAL_W_DEC = _read_only(_params["w_dec"])  # latent × input; edits are checked against this
_decoder = _ORIGINAL_W_DEC.copy()               # the current (edited) decoder; changed only by apply_edit
W_dec = _read_only(_decoder.copy())             # what snippets see: a read-only copy of the current decoder
del _params

X_train = _read_only(np.load(INPUTS / "X_train.npy"))
y_train = _read_only(np.load(INPUTS / "y_train.npy"))
meta_train = pd.read_json(INPUTS / "meta_train.jsonl", lines=True)
Z_train = scipy.sparse.load_npz(INPUTS / "Z_train.npz").tocsc()   # encode(X_train), column access is fast
selected = _rules["selected"]
hints = {int(k): v for k, v in json.loads((INPUTS / "hints.json").read_text()).items()}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def encode(x):
    """SAE codes of activations x (rows × 2560): z = relu(pre) · [pre > threshold], pre = x @ W_enc + b_enc.

    Returns a dense float32 array (rows × 16384). For the train rows use Z_train, which is this, precomputed."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    out = np.empty((len(x), W_enc.shape[1]), dtype=np.float32)
    for start in range(0, len(x), CHUNK_ROWS):
        pre = x[start:start + CHUNK_ROWS] @ W_enc + b_enc
        out[start:start + CHUNK_ROWS] = np.maximum(pre, 0) * (pre > threshold)
    return out


def decode(z):
    """Reconstruction z @ W_dec + b_dec with the CURRENT (edited) decoder. z may be dense or scipy sparse."""
    out = np.empty((z.shape[0], _decoder.shape[1]), dtype=np.float32)
    for start in range(0, z.shape[0], CHUNK_ROWS):
        part = z[start:start + CHUNK_ROWS]
        out[start:start + CHUNK_ROWS] = (part @ _decoder if scipy.sparse.issparse(part)
                                         else np.asarray(part, dtype=np.float32) @ _decoder) + b_dec
    return out


def separation(i):
    """How feature i separates adoption (y=1) from retention (y=0) on train. Measured, nothing fitted.

    firing %, mean activation (zeros included) per label, and AUC = P(activation of a random adoption
    row > that of a random retention row), ties counted half."""
    activation = Z_train[:, i].toarray().ravel()
    adoption, retention = activation[y_train == 1], activation[y_train == 0]
    sorted_retention = np.sort(retention)
    below = np.searchsorted(sorted_retention, adoption, side="left")
    at_or_below = np.searchsorted(sorted_retention, adoption, side="right")
    auc = float(np.mean(below + 0.5 * (at_or_below - below)) / len(retention))
    return {"feature": int(i), "fires_pct_adoption": 100 * float(np.mean(adoption > 0)),
            "fires_pct_retention": 100 * float(np.mean(retention > 0)),
            "mean_adoption": float(adoption.mean()), "mean_retention": float(retention.mean()), "auc": auc}


def top_examples(i, k=5):
    """The k train episodes where feature i is most active, with their texts (truncated)."""
    activation = Z_train[:, i].toarray().ravel()
    rows = np.argsort(-activation)[:k]
    table = meta_train.iloc[rows].copy()
    table.insert(0, "activation", activation[rows])
    for column, width in (("question", 300), ("solver_attempt_1", 150),
                          ("critic_advocated_answer", 150), ("solver_attempt_2", 150)):
        table[column] = table[column].astype(str).str.slice(0, width)
    return table.reset_index(drop=True)


def _unit(v):
    return v / np.linalg.norm(v)


def _cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _set_directions(decoder):
    """u_A, u_R and the 'both'-arm direction normalize(u_A − u_R), as stage 6 builds them."""
    u = {label: _unit(sum(_unit(decoder[f].astype(np.float64)) for f in features))
         for label, features in selected.items()}
    return u["adoption"], u["retention"], _unit(u["adoption"] - u["retention"])


def _edit_problems(feature, row):
    """Same rules as mas_sae.llm_edit.edit_rules.edit_problems (the host re-checks with that one)."""
    if feature not in set(selected["adoption"]) | set(selected["retention"]):
        return [f"feature {feature} is not one of the 20 selected features"]
    original = _ORIGINAL_W_DEC[feature].astype(np.float64)
    if row.shape != original.shape:
        return [f"row has shape {row.shape}, expected {original.shape}"]
    if not np.all(np.isfinite(row)):
        return ["row has non-finite values"]
    problems = []
    ratio = np.linalg.norm(row) / np.linalg.norm(original)
    if abs(ratio - 1) > _rules["length_tolerance"]:
        problems.append(f"row length changed by a factor of {ratio:.6f} "
                        f"(must stay within 1 ± {_rules['length_tolerance']})")
    if _cosine(row, original) < _rules["cosine_cap"]:
        problems.append(f"cosine with the original row is {_cosine(row, original):.4f}, "
                        f"below the cap {_rules['cosine_cap']}")
    return problems


def _set_decoder_row(feature, row):
    global _decoder, W_dec
    _decoder[feature] = row
    W_dec = _read_only(_decoder.copy())


def _read_outbox():
    path = OUTBOX / "edits.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def apply_edit(feature, new_row, why):
    """Replace feature's decoder row with new_row (2560 numbers) and record why. The only way to edit.

    Rules: feature must be one of the 20 in `selected`; keep the row's length; keep cosine >= the cap
    with the ORIGINAL row. Raises ValueError (and changes nothing) if a rule is broken. Returns a
    report of how the row and the set directions changed."""
    feature = int(feature)
    row = np.asarray(new_row, dtype=np.float32).ravel()
    problems = _edit_problems(feature, row.astype(np.float64))
    if problems:
        raise ValueError("Edit rejected: " + "; ".join(problems))

    before = _set_directions(_decoder)
    _set_decoder_row(feature, row)
    after = _set_directions(_decoder)
    original = _set_directions(_ORIGINAL_W_DEC)

    number = len(_read_outbox()) + 1
    with open(OUTBOX / "edits.jsonl", "a") as outbox:
        outbox.write(json.dumps({"edit": number, "feature": feature, "row": row.tolist(), "why": str(why)}) + "\n")

    names = ("u_adoption", "u_retention", "both-arm direction")
    report = {"edit": number, "feature": feature,
              "row_cosine_with_original": _cosine(row, _ORIGINAL_W_DEC[feature]),
              "set_directions": {name: {"cosine_with_original": _cosine(a, o), "cosine_with_before_this_edit": _cosine(a, b)}
                                 for name, a, b, o in zip(names, after, before, original)}}
    print(json.dumps(report, indent=1))
    return report


def _replay_outbox():
    """After a kernel restart: restore the decoder from the edits already recorded."""
    for entry in _read_outbox():
        _set_decoder_row(entry["feature"], np.asarray(entry["row"], dtype=np.float32))


_replay_outbox()
print(f"Sandbox ready. X_train {X_train.shape}, y_train {y_train.shape} (1 = adoption), "
      f"Z_train {Z_train.shape} sparse, W_enc {W_enc.shape}, W_dec {W_dec.shape}, "
      f"{len(_read_outbox())} edits so far. help(apply_edit) for the rules.")
