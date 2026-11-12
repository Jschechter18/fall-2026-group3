"""The LLM-edit sandbox, end to end in a real container (mas_sae.llm_edit.sandbox).

Needs Docker access (in a shell started before joining the docker group: `sg docker -c "pytest ..."`)
and the sandbox inputs from scripts/sae/llm_edit_sandbox_inputs.py; skipped otherwise. Each container
starts in about 5 s; the out-of-memory test takes longer.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mas_sae.llm_edit.edit_rules import cosine_similarity
from mas_sae.llm_edit.sandbox import Sandbox, SandboxConfig, SandboxError

REPO = Path(__file__).resolve().parents[4]
INPUTS = REPO / "artifacts/llm_edit/sandbox_inputs"
SAE_DIR = REPO / "results/sae/pretrained/gemma-scope-2-4b-it/resid_post_all/layer_25_width_16k_l0_big"


def docker_usable() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not (docker_usable() and (INPUTS / "rules.json").exists()),
                                reason="needs Docker access and artifacts/llm_edit/sandbox_inputs")


@pytest.fixture(scope="module")
def original():
    from safetensors.numpy import load_file
    w_dec = load_file(str(SAE_DIR / "params.safetensors"))["w_dec"]
    return w_dec, json.loads((INPUTS / "rules.json").read_text())["selected"]


def new_sandbox(original, outbox: Path) -> Sandbox:
    w_dec, selected = original
    return Sandbox(SandboxConfig(INPUTS, SAE_DIR, outbox), w_dec, selected)


@pytest.fixture(scope="module")
def sandbox(original, tmp_path_factory):
    with new_sandbox(original, tmp_path_factory.mktemp("outbox")) as running:
        yield running


def run_ok(sandbox, code: str) -> str:
    result = sandbox.run(code)
    assert result.error is None, result.error
    return result.output


# ---- isolation ------------------------------------------------------

def test_no_network(sandbox):
    output = run_ok(sandbox, "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
                             "    print('CONNECTED')\nexcept OSError as e:\n    print('blocked', type(e).__name__)")
    assert "blocked" in output and "CONNECTED" not in output


def test_no_repository_and_no_mas_sae(sandbox):
    output = run_ok(sandbox, f"from pathlib import Path\nprint(Path({str(REPO)!r}).exists())\n"
                             "try:\n    import mas_sae\n    print('IMPORTED')\nexcept ImportError:\n    print('no mas_sae')")
    assert "False" in output and "no mas_sae" in output


def test_inputs_and_parameters_are_read_only(sandbox):
    output = run_ok(sandbox, "results = []\n"
                             "for attempt in (lambda: open('/inputs/x', 'w'), lambda: W_dec.__setitem__((0, 0), 1.0),\n"
                             "                lambda: X_train.__setitem__((0, 0), 1.0)):\n"
                             "    try:\n        attempt(); results.append('WRITABLE')\n"
                             "    except (OSError, ValueError):\n        results.append('read-only')\n"
                             "print(results)")
    assert output.count("read-only") == 3


# ---- no training ----------------------------------------------------

@pytest.mark.parametrize("module", ["torch.optim", "scipy.optimize", "sklearn", "sklearn.linear_model"])
def test_optimizer_and_fitting_imports_are_blocked(sandbox, module):
    result = sandbox.run(f"import {module}")
    assert result.error is not None and "blocked in this sandbox" in result.error


def test_torch_optim_attribute_is_gone(sandbox):
    assert "False" in run_ok(sandbox, "import torch\nprint(hasattr(torch, 'optim'))")


@pytest.mark.parametrize("code", [
    "import torch\n(torch.ones(2, requires_grad=True) * 2).sum().backward()",
    "import torch\ntorch.set_grad_enabled(True)",
    "import torch\nwith torch.enable_grad():\n    pass",
    "import torch\nx = torch.ones(2, requires_grad=True)\ntorch.autograd.grad((x * 2).sum(), x)",
])
def test_gradients_cannot_be_turned_on(sandbox, code):
    assert sandbox.run(code).error is not None


# ---- a persistent kernel with a time limit ---------------------------

def test_variables_persist_and_survive_a_timeout(sandbox):
    run_ok(sandbox, "kept = 41")
    timed_out = sandbox.run("while True:\n    pass")
    assert timed_out.timed_out and not timed_out.restarted
    assert "42" in run_ok(sandbox, "print(kept + 1)")


# ---- helpers agree with mas_sae on the host --------------------------

def test_encode_matches_the_host_encoder(sandbox):
    # Z_train was computed on the host with mas_sae's JumpReLUSAE; the sandbox's encode must agree.
    output = run_ok(sandbox, "z = encode(X_train[:256]); ref = Z_train[:256].toarray()\n"
                             "print(float(np.abs(z - ref).max() / np.abs(ref).max()), int(((z > 0) != (ref > 0)).sum()))")
    relative_error, support_differences = output.split()
    assert float(relative_error) < 1e-4 and int(support_differences) == 0


def test_decode_matches_the_host_decoder(sandbox):
    import torch
    from mas_sae.sae.pretrained import load_gemma_scope_sae
    host_sae, _ = load_gemma_scope_sae(SAE_DIR)
    codes = np.load(INPUTS / "X_train.npy", mmap_mode="r")[:8]
    with torch.no_grad():
        expected = host_sae.decoder(host_sae.encoder(torch.from_numpy(np.array(codes)))).numpy()
    # A 64-column slice keeps the printed output far below the driver's output cap.
    output = run_ok(sandbox, "import json\nprint(json.dumps(decode(Z_train[:8])[:, :64].tolist()))")
    expected = expected[:, :64]
    assert np.abs(np.array(json.loads(output)) - expected).max() < 1e-3 * np.abs(expected).max()


# ---- edits ----------------------------------------------------------

ROTATE = """
def rotated(feature, cosine, seed):
    r = W_dec[feature].astype(np.float64); u = r / np.linalg.norm(r)
    v = np.random.default_rng(seed).normal(size=u.shape); p = v - (v @ u) * u; p /= np.linalg.norm(p)
    return np.linalg.norm(r) * (cosine * u + np.sqrt(1 - cosine ** 2) * p)
"""


def test_edits_are_checked_in_the_kernel_and_again_on_the_host(original, sandbox):
    w_dec, selected = original
    feature, outside = selected["adoption"][0], 0
    run_ok(sandbox, ROTATE)

    too_far = sandbox.run(f"apply_edit({feature}, rotated({feature}, 0.90, 1), 'too far')")
    assert "below the cap" in too_far.error and too_far.accepted_edits == []
    not_selected = sandbox.run(f"apply_edit({outside}, W_dec[{outside}], 'not mine')")
    assert "not one of the 20" in not_selected.error

    valid = sandbox.run(f"report = apply_edit({feature}, rotated({feature}, 0.97, 2), 'a small turn')")
    assert valid.error is None and [e.feature for e in valid.accepted_edits] == [feature]
    assert abs(cosine_similarity(valid.accepted_edits[0].row, w_dec[feature]) - 0.97) < 1e-4
    assert "0.97" in run_ok(sandbox, f"print(round(float(W_dec[{feature}] @ {w_dec[feature].tolist()} / "
                                     f"(np.linalg.norm(W_dec[{feature}]) * {float(np.linalg.norm(w_dec[feature]))})), 2))")


def test_the_host_rejects_an_edit_written_around_apply_edit(original, tmp_path):
    w_dec, selected = original
    feature = selected["retention"][0]
    with new_sandbox(original, tmp_path) as sandbox:
        run_ok(sandbox, ROTATE)
        forged = sandbox.run("import json\nwith open('/outbox/edits.jsonl', 'a') as f:\n"
                             f"    f.write(json.dumps({{'edit': 1, 'feature': {feature}, "
                             f"'row': list(rotated({feature}, 0.5, 3)), 'why': 'forged'}}) + '\\n')")
        assert forged.accepted_edits == [] and "below the cap" in forged.rejected_edits[0]

        with pytest.raises(SandboxError, match="rewritten"):        # rewriting the record stops the run
            sandbox.run("open('/outbox/edits.jsonl', 'w').close()")


def test_out_of_memory_restarts_the_kernel_and_keeps_the_edits(original, tmp_path):
    w_dec, selected = original
    feature = selected["adoption"][1]
    with new_sandbox(original, tmp_path) as sandbox:
        run_ok(sandbox, ROTATE + f"apply_edit({feature}, rotated({feature}, 0.98, 5), 'kept across restarts')\nlost = 1")
        blown = sandbox.run("big = np.ones(int(12e9) // 8)")     # 12 GB > the 8 GB container limit
        assert blown.restarted
        output = run_ok(sandbox, f"print('lost' in dir(), round(float(np.dot(W_dec[{feature}], "
                                 f"{w_dec[feature].tolist()})), 2))")
        assert output.split() == ["False", "0.98"]
