"""The sandbox's in-container driver: one persistent Jupyter kernel, driven over stdin/stdout.

The container has no network (--network none), so the host cannot reach a kernel's ports; the kernel
and its jupyter_client manager both live in here, on the container's loopback. The host
(mas_sae.llm_edit.sandbox) sends one JSON request per line on stdin and reads one JSON reply per
line on stdout. Nothing else is written to stdout.

    {"op": "run", "code": "...", "timeout_s": 30}  ->  {"output": "...", "error": null | "...",
                                                        "timed_out": bool, "restarted": bool, "seconds": float}
    {"op": "shutdown"}                             ->  {"ok": true}

A snippet that runs past its time limit is interrupted (its variables survive). If the kernel does
not stop, or dies (e.g. out of memory), it is restarted and kernel_setup.py runs again; the decoder
is restored from the recorded edits, other variables are lost, and the reply says restarted: true.
"""

import json
import queue
import re
import sys
import time
from pathlib import Path

from jupyter_client import KernelManager

SETUP_FILE = Path(__file__).with_name("kernel_setup.py")
SETUP_TIMEOUT_S = 600
INTERRUPT_GRACE_S = 10
MAX_OUTPUT_CHARS = 200_000     # the host truncates further before anything reaches the LLM
ANSI_ESCAPES = re.compile(r"\x1b\[[0-9;]*m")


def main():
    kernel_manager, client = start_kernel()
    reply(run_setup(kernel_manager, client))
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "shutdown":
            reply({"ok": True})
            break
        result = execute(kernel_manager, client, request["code"], request.get("timeout_s", 30))
        if result["timed_out"] or not kernel_manager.is_alive():
            result = recover(kernel_manager, client, result)
        reply(result)
    client.stop_channels()
    kernel_manager.shutdown_kernel(now=True)


def start_kernel():
    # IPC transport: kernel and manager talk over Unix sockets in /tmp, not TCP ports.
    kernel_manager = KernelManager(kernel_name="python3", transport="ipc", ip="/tmp/kernel")
    kernel_manager.start_kernel()
    client = kernel_manager.client()
    client.start_channels()
    client.wait_for_ready(timeout=120)
    return kernel_manager, client


def run_setup(kernel_manager, client) -> dict:
    result = execute(kernel_manager, client, SETUP_FILE.read_text(), SETUP_TIMEOUT_S)
    if result["error"] or result["timed_out"]:
        raise SystemExit(f"Kernel setup failed: {result}")
    return {"setup": True, **result}


def recover(kernel_manager, client, result: dict) -> dict:
    """The snippet overran its limit: interrupt it; if the kernel still won't stop (or died), restart it."""
    if kernel_manager.is_alive():
        kernel_manager.interrupt_kernel()
        if wait_for_idle(client, INTERRUPT_GRACE_S):
            return result
    kernel_manager.restart_kernel(now=True)
    client.wait_for_ready(timeout=120)
    run_setup(kernel_manager, client)
    note = ("\n[The kernel was restarted: all variables were lost except the edits already applied, "
            "which were restored. The preloaded inputs and helpers are back.]")
    return {**result, "restarted": True, "output": result["output"] + note}


def execute(kernel_manager, client, code: str, timeout_s: float) -> dict:
    """Run code; collect its printed output, results and error until the kernel is idle or time runs out."""
    started = time.monotonic()
    message_id = client.execute(code, store_history=True)
    output, error, timed_out = [], None, False
    while True:
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            timed_out = True
            break
        try:
            message = client.get_iopub_msg(timeout=min(remaining, 1.0))
        except queue.Empty:
            if not kernel_manager.is_alive():
                error = "The kernel died (probably out of memory)."
                break
            continue
        if message["parent_header"].get("msg_id") != message_id:
            continue
        kind, content = message["msg_type"], message["content"]
        if kind == "stream":
            output.append(content["text"])
        elif kind in ("execute_result", "display_data"):
            output.append(content["data"].get("text/plain", "") + "\n")
        elif kind == "error":
            error = ANSI_ESCAPES.sub("", "\n".join(content["traceback"]))
        elif kind == "status" and content["execution_state"] == "idle":
            break
    text = "".join(output)
    if timed_out:
        text += f"\n[Stopped: the snippet ran past its {timeout_s:.0f} s limit.]"
    return {"output": text[:MAX_OUTPUT_CHARS], "error": error, "timed_out": timed_out, "restarted": False,
            "seconds": round(time.monotonic() - started, 3)}


def wait_for_idle(client, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            message = client.get_iopub_msg(timeout=0.5)
        except queue.Empty:
            continue
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            return True
    return False


def reply(value: dict) -> None:
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
