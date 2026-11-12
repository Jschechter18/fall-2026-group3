"""Host side of the LLM-edit sandbox: Claude's snippets run in a Docker container with no network.

The container (decided in Phase 0 of cookbooks/LLM_EDITED_SAE.md):
  - the stock ubuntu image with the host's conda env mounted read-only at its host path, so package
    versions match the host's (the env's editable mas_sae install points at the repository, which is
    not mounted, so mas_sae cannot be imported inside);
  - no network, read-only root filesystem, a /tmp tmpfs, every capability dropped, no new privileges,
    a pid limit, the host user's uid (not root), and CPU and memory caps;
  - mounts, all read-only except the outbox: the sandbox inputs (/inputs), the original SAE folder
    (/sae), the in-container code (/sandbox, this package's container/ folder), and /outbox, where
    the kernel's apply_edit records edits.

The in-container driver (container/driver.py) keeps one Jupyter kernel and answers one JSON request
per line over the container's stdin/stdout. After every snippet the host reads the new edits from
the outbox and re-checks each against the ORIGINAL SAE with edit_rules.edit_problems: kernel state
can never change the result, only edits accepted here (and replayed from edits.py) do.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mas_sae.llm_edit.edit_rules import Edit, edit_problems

IMAGE = "ubuntu:26.04"
CONTAINER_CODE = Path(__file__).parent / "container"
HOST_TIMEOUT_MARGIN_S = 120     # the driver enforces the snippet limit; this only catches a hung container


@dataclass(frozen=True)
class SandboxConfig:
    inputs_dir: Path               # written by scripts/sae/llm_edit_sandbox_inputs.py
    sae_dir: Path                  # the original SAE folder (params.safetensors)
    outbox_dir: Path               # writable; the kernel appends edits.jsonl here
    image: str = IMAGE
    cpus: float = 3
    memory: str = "8g"             # 15.4 GiB machine, no swap, Gemma loaded in a host process meanwhile
    pids_limit: int = 256
    snippet_timeout_s: float = 30
    setup_timeout_s: float = 900
    conda_env: Path = field(default_factory=lambda: Path(sys.prefix))


@dataclass
class SnippetResult:
    output: str
    error: str | None
    timed_out: bool
    restarted: bool
    seconds: float
    accepted_edits: list[Edit]     # new edits that passed the host's check
    rejected_edits: list[str]      # why any new outbox entry was refused


class SandboxError(RuntimeError):
    pass


class Sandbox:
    """Start with `with Sandbox(config, original_w_dec, selected) as sandbox:`, then `sandbox.run(code)`."""

    def __init__(self, config: SandboxConfig, original_w_dec: np.ndarray, selected: dict[str, list[int]]):
        self.config = config
        self.original_w_dec = original_w_dec
        self.selected = selected
        self.container_name = f"llm-edit-sandbox-{uuid.uuid4().hex[:12]}"
        self.edits: list[Edit] = []         # every accepted edit, in order
        self._outbox_seen = b""             # outbox/edits.jsonl as last read; it may only grow
        self._process: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self.setup_output = ""

    # ---- lifecycle ---------------------------------------------------

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        self.config.outbox_dir.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(self.docker_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=open(self.config.outbox_dir / "container_stderr.log", "a"),
                                         text=True, bufsize=1)
        threading.Thread(target=self._read_replies, daemon=True).start()
        setup = self._receive(self.config.setup_timeout_s)
        self.setup_output = setup["output"]

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._send({"op": "shutdown"})
            self._receive(30)
            self._process.wait(timeout=30)
        except (SandboxError, OSError, subprocess.TimeoutExpired):
            subprocess.run(["docker", "kill", self.container_name], capture_output=True)
        self._process = None

    # ---- snippets ----------------------------------------------------

    def run(self, code: str) -> SnippetResult:
        self._send({"op": "run", "code": code, "timeout_s": self.config.snippet_timeout_s})
        reply = self._receive(self.config.snippet_timeout_s + HOST_TIMEOUT_MARGIN_S)
        accepted, rejected = self._collect_new_edits()
        return SnippetResult(reply["output"], reply["error"], reply["timed_out"], reply["restarted"],
                             reply["seconds"], accepted, rejected)

    def _collect_new_edits(self) -> tuple[list[Edit], list[str]]:
        """Re-check every edit the kernel recorded since the last snippet; keep only valid ones."""
        path = self.config.outbox_dir / "edits.jsonl"
        if not path.exists():
            return [], []
        content = path.read_bytes()
        if not content.startswith(self._outbox_seen):
            raise SandboxError("outbox/edits.jsonl was rewritten, not appended to: the edit record was tampered with.")
        new_lines = content[len(self._outbox_seen):].decode().splitlines()
        self._outbox_seen = content

        accepted, rejected = [], []
        for line in filter(str.strip, new_lines):
            try:
                entry = json.loads(line)
                edit = Edit(int(entry["edit"]), int(entry["feature"]),
                            np.asarray(entry["row"], dtype=np.float32), str(entry.get("why", "")))
            except (ValueError, KeyError, TypeError) as e:
                rejected.append(f"unreadable outbox entry ({e})")
                continue
            expected_number = len(self.edits) + len(accepted) + 1
            problems = edit_problems(self.original_w_dec, self.selected, edit.feature, edit.row)
            if edit.number != expected_number:
                problems.append(f"edit number {edit.number}, expected {expected_number}")
            if problems:
                rejected.append(f"edit {edit.number} (feature {edit.feature}): " + "; ".join(problems))
            else:
                accepted.append(edit)
        self.edits += accepted
        return accepted, rejected

    # ---- the container -----------------------------------------------

    def docker_command(self) -> list[str]:
        c = self.config
        env = str(c.conda_env)
        def mount(host, container, mode):
            options = ",readonly" if mode == "readonly" else ""     # --mount is writable unless readonly
            return ["--mount", f"type=bind,src={Path(host).resolve()},dst={container}{options}"]

        return ["docker", "run", "-i", "--rm", "--name", self.container_name,
                "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=2g",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", str(c.pids_limit), "--cpus", str(c.cpus),
                "--memory", c.memory, "--memory-swap", c.memory,
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--env", "HOME=/tmp", "--env", "JUPYTER_RUNTIME_DIR=/tmp/jupyter",
                "--env", "JUPYTER_DATA_DIR=/tmp/jupyter", "--env", "IPYTHONDIR=/tmp/ipython",
                "--env", "MPLCONFIGDIR=/tmp/matplotlib", "--env", "OMP_NUM_THREADS=3",
                "--workdir", "/tmp",
                *mount(env, env, "readonly"),
                *mount(c.inputs_dir, "/inputs", "readonly"),
                *mount(c.sae_dir, "/sae", "readonly"),
                *mount(CONTAINER_CODE, "/sandbox", "readonly"),
                *mount(c.outbox_dir, "/outbox", "rw"),
                c.image, f"{env}/bin/python", "/sandbox/driver.py"]

    def provenance(self) -> dict:
        """What run.json records about the sandbox."""
        digest = subprocess.run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", self.config.image],
                                capture_output=True, text=True).stdout.strip()
        return {"image": self.config.image, "image_repo_digests": json.loads(digest or "[]"),
                "cpus": self.config.cpus, "memory": self.config.memory, "pids_limit": self.config.pids_limit,
                "snippet_timeout_s": self.config.snippet_timeout_s, "conda_env": str(self.config.conda_env),
                "docker_command": self.docker_command()}

    def _send(self, request: dict) -> None:
        if self._process is None or self._process.poll() is not None:
            raise SandboxError("the sandbox container is not running")
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

    def _read_replies(self) -> None:
        for line in self._process.stdout:
            self._replies.put(json.loads(line))
        self._replies.put(None)             # the container exited

    def _receive(self, timeout_s: float) -> dict:
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            raise SandboxError(f"no reply from the sandbox within {timeout_s:.0f} s") from None
        if reply is None:
            log = self.config.outbox_dir / "container_stderr.log"
            raise SandboxError(f"the sandbox container exited; see {log}")
        return reply


def environment_record(conda_env: Path | None = None) -> dict:
    """The sandbox's package lists (conda list --explicit and pip freeze), for run.json."""
    env = Path(conda_env or sys.prefix)
    conda = subprocess.run(["conda", "list", "--explicit", "--prefix", str(env)], capture_output=True, text=True)
    pip = subprocess.run([str(env / "bin" / "python"), "-m", "pip", "freeze"], capture_output=True, text=True)
    return {"conda_explicit": conda.stdout.splitlines(), "pip_freeze": pip.stdout.splitlines()}
