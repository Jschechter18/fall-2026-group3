"""Phase 5 of the LLM-edited SAE experiment (cookbooks/LLM_EDITED_SAE.md): the audit.

For each edited SAE (LLM, mechanical, random):
  1. Replay edits.py onto the original Pretrained SAE, save it the same way the run did, and check
     that the params.safetensors hash matches the saved one (and that every tensor is bit-identical).
  2. Check that only the 20 selected decoder rows differ from the original, and that the encoder,
     threshold and decoder bias did not change at all.
  3. Check the rules on every accepted edit (not only the final rows): length kept, cosine >= the cap
     with the original row.
  4. Report each set direction's cosine with d (host-only; the LLM never saw d).

For the LLM run only:
  5. Token use and cost, recomputed from the transcript's per-turn usage and checked against run.json.
  6. Counts of turns, snippets, steer_test calls, protocol errors, snippet errors, timeouts and restarts.
  7. Which steer_test call, if any, tested exactly the final decoder.
  8. A pattern scan of every snippet for blocked imports, gradient use, optimizers, loops around
     apply_edit, process/network/file escape attempts, and any host-only text (d) sent to the LLM.
     Hits are listed by snippet number for a human to read; a hit is not by itself a violation.

Writes results/sae/llm_edit_audit/<run_id>/audit.json and prints a report. Reads only; never changes
the audited run folders. CPU only, no Bedrock.

    PYTHONPATH=src python scripts/sae/audit_llm_edit.py
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic
from mas_sae.intervention.experiment.config import load_config
from mas_sae.intervention.run_files.layout import new_run_id
from mas_sae.llm_edit.directions import direction_report, mean_difference_direction
from mas_sae.llm_edit.edit_rules import COSINE_CAP, apply_edits, cosine_similarity, edit_problems, read_edits_py
from mas_sae.llm_edit.edited_sae import SAE_DIR, original_sae, save_edited_sae

STUDY_CONFIG = Path("configs/intervention/intervention.yaml")
SANDBOX_INPUTS = Path("artifacts/llm_edit/sandbox_inputs")
OUTPUT_ROOT = Path("results/sae/llm_edit_audit")
DEFAULT_RUNS = {
    "llm": Path("results/sae/llm_edited/20261111T213251406629Z-c53c08f04ab9"),
    "mechanical": Path("results/sae/mechanical_edit/20261111T211535990035Z-6a61cf6c6b31"),
    "random": Path("results/sae/random_edit/20261111T234022791132Z-c53c08f04ab9"),
}
UNCHANGED_TENSORS = ("w_enc", "b_enc", "threshold", "b_dec")

# Snippet patterns worth a human look. Names match the guards in container/kernel_setup.py.
SCAN_PATTERNS = {
    "blocked import (attempted)": r"\b(?:import|from)\s+(?:torch\.optim|scipy\.optimize|sklearn)\b|__import__|importlib",
    "gradient use": r"\.backward\s*\(|autograd|requires_grad|enable_grad|set_grad_enabled|\.grad\b",
    "optimizer or fitting": r"\boptim\b|minimize|curve_fit|lstsq|\.fit\s*\(|LogisticRegression|gradient",
    "while loop": r"^\s*while\b",
    "apply_edit called in a loop": None,   # checked structurally in scan_snippet
    "process, network or file access": r"subprocess|os\.system|\bsocket\b|urllib|requests|\bopen\s*\(|/outbox|ctypes|sys\.modules",
}
GUARD_MESSAGES = {
    "blocked import (raised)": "is blocked in this sandbox",
    "gradient guard (raised)": "Gradients are disabled in this sandbox",
}
HOST_ONLY_TEXT = ("cosine_with_d", "host_diagnostic", "d_sha256", "mean_difference_direction")


def run(runs: dict[str, Path]) -> Path:
    study = load_config(STUDY_CONFIG)
    original_w_dec, sae_config, selected = original_sae(study.inputs.feature_sets)
    original_params = load_file(str(SAE_DIR / "params.safetensors"))
    d = mean_difference_direction(SANDBOX_INPUTS)

    report = {"run_id": new_run_id(), "command": get_run_command(), "git_commit": get_git_commit(),
              "cosine_cap": COSINE_CAP, "runs": {name: str(path) for name, path in runs.items()}, "groups": {}}
    for name, run_dir in runs.items():
        report["groups"][name] = audit_group(run_dir, original_w_dec, original_params, sae_config, selected, d)
    report["llm_run"] = audit_llm_run(runs["llm"], original_w_dec)

    out_dir = OUTPUT_ROOT / report["run_id"]
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(out_dir / "audit.json", report)
    print_report(report)
    print(f"\nWritten to {out_dir / 'audit.json'}")
    return out_dir


# =====================================================================
# EVERY GROUP: replay, changed rows, rules, set directions
# =====================================================================

def audit_group(run_dir, original_w_dec, original_params, sae_config, selected, d) -> dict:
    edits = read_edits_py(run_dir)   # checks every row file against the sha256 in edits.py
    replayed = apply_edits(original_w_dec, edits)
    saved_params = load_file(str(run_dir / "sae" / "params.safetensors"))

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = save_edited_sae(replayed, sae_config, Path(tmp) / "sae", {"audit": "replay"})
        rebuilt_sha = sha256_file(rebuilt / "params.safetensors")
    saved_sha = sha256_file(run_dir / "sae" / "params.safetensors")

    allowed = set(selected["adoption"]) | set(selected["retention"])
    changed_rows = sorted(int(i) for i in np.flatnonzero(np.any(saved_params["w_dec"] != original_w_dec, axis=1)))
    rule_problems = [f"edit {e.number} (feature {e.feature}): {p}"
                     for e in edits for p in edit_problems(original_w_dec, selected, e.feature, e.row)]
    final_rows = {f: saved_params["w_dec"][f] for f in changed_rows}

    return {
        "edits": len(edits),
        "params_sha256": {"saved": saved_sha, "replayed": rebuilt_sha, "match": saved_sha == rebuilt_sha},
        "replay_bit_identical": bool(np.array_equal(replayed, saved_params["w_dec"])),
        "other_tensors_unchanged": {t: bool(np.array_equal(saved_params[t], original_params[t])) for t in UNCHANGED_TENSORS},
        "changed_rows": changed_rows,
        "only_selected_rows_changed": set(changed_rows) <= allowed,
        "all_selected_rows_changed": set(changed_rows) == allowed,
        "rule_problems": rule_problems,
        "final_cosine_with_original": summary_stats([cosine_similarity(r, original_w_dec[f]) for f, r in final_rows.items()]),
        "final_length_ratio": summary_stats([float(np.linalg.norm(r.astype(np.float64)) /
                                                   np.linalg.norm(original_w_dec[f].astype(np.float64)))
                                             for f, r in final_rows.items()]),
        "set_directions": direction_report(saved_params["w_dec"], original_w_dec, selected, d),
    }


def summary_stats(values: list[float]) -> dict:
    return {"min": float(min(values)), "max": float(max(values))} if values else {}


# =====================================================================
# LLM RUN ONLY: tokens, counts, tested states, transcript scan
# =====================================================================

def audit_llm_run(run_dir: Path, original_w_dec: np.ndarray) -> dict:
    entries = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()]
    run_record = json.loads((run_dir / "run.json").read_text())
    by_kind = lambda kind: [e for e in entries if e["kind"] == kind]
    snippets = by_kind("snippet")

    return {
        "outcome": run_record["outcome"],
        "tokens_and_cost": tokens_and_cost(by_kind("assistant"), run_record),
        "counts": {
            "turns": len(by_kind("assistant")), "snippets": len(snippets), "steer_test_calls": len(by_kind("steer_test")),
            "protocol_errors": len(by_kind("protocol_error")),
            "snippet_errors": [s["number"] for s in snippets if s["error"]],
            "snippet_timeouts": [s["number"] for s in snippets if s["timed_out"]],
            "kernel_restarts": [s["number"] for s in snippets if s["restarted"]],
            "rejected_edits": sum(len(s["rejected_edits"]) for s in snippets),
            "longest_snippet_seconds": max((s["seconds"] for s in snippets), default=0),
        },
        "steer_tested_states": steer_tested_states(run_dir, by_kind("steer_test"), original_w_dec),
        "scan": {name: hits for name, hits in scan_transcript(snippets).items()},
        "host_only_text_sent_to_llm": [i for i, e in enumerate(entries) if e["kind"] == "user"
                                       and any(text in e["content"] for text in HOST_ONLY_TEXT)],
    }


def tokens_and_cost(assistant_turns: list[dict], run_record: dict) -> dict:
    """Recomputed at list price; cache reads and writes are counted as plain input (an upper bound)."""
    price = run_record["price_per_million_tokens_usd"]
    input_tokens = sum(t["usage"]["input_tokens"] + t["usage"].get("cache_creation_input_tokens", 0)
                       + t["usage"].get("cache_read_input_tokens", 0) for t in assistant_turns)
    output_tokens = sum(t["usage"]["output_tokens"] for t in assistant_turns)
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd_at_list_price": round(cost, 4),
            "matches_run_json": (input_tokens, output_tokens) == (run_record["input_tokens"], run_record["output_tokens"])}


def steer_tested_states(run_dir: Path, steer_calls: list[dict], original_w_dec: np.ndarray) -> list[dict]:
    """For each steer_test call, whether the decoder it tested is exactly the final saved decoder."""
    edits = read_edits_py(run_dir)
    final = apply_edits(original_w_dec, edits)
    return [{"call": call["number"], "edits_so_far": call["edits_so_far"],
             "flips": call["summary"]["edited"]["flips"], "unclear": call["summary"]["edited"]["unclear"],
             "is_final_decoder": bool(np.array_equal(apply_edits(original_w_dec, edits[:call["edits_so_far"]]), final))}
            for call in steer_calls]


def scan_transcript(snippets: list[dict]) -> dict[str, list[dict]]:
    hits = {name: [] for name in [*SCAN_PATTERNS, *GUARD_MESSAGES]}
    for snippet in snippets:
        for name, found in scan_snippet(snippet).items():
            hits[name] += [{"snippet": snippet["number"], "match": text} for text in found]
    return hits


def scan_snippet(snippet: dict) -> dict[str, list[str]]:
    code, output = snippet["code"], f"{snippet['output']}\n{snippet['error'] or ''}"
    found = {}
    for name, pattern in SCAN_PATTERNS.items():
        if pattern is not None:
            found[name] = [line.strip() for line in code.splitlines() if re.search(pattern, line)]
    found["apply_edit called in a loop"] = apply_edit_in_loop(code)
    for name, message in GUARD_MESSAGES.items():
        found[name] = [line.strip() for line in output.splitlines() if message in line]
    return {name: lines for name, lines in found.items() if lines}


def apply_edit_in_loop(code: str) -> list[str]:
    """apply_edit lines indented under a for/while loop: a loop that edits, then measures, then edits
    again would be a search. Looping once over the 20 features to apply them is expected and also listed."""
    lines, hits, loop_indents = code.splitlines(), [], []
    for line in lines:
        indent = len(line) - len(line.lstrip())
        loop_indents = [i for i in loop_indents if i < indent or not line.strip()]
        is_loop = re.match(r"\s*(for|while)\b", line)
        if "apply_edit(" in line and (loop_indents or is_loop):   # also `for i in A: apply_edit(...)`
            hits.append(line.strip())
        if is_loop:
            loop_indents.append(indent)
    return hits


# =====================================================================
# REPORT
# =====================================================================

def print_report(report: dict) -> None:
    print("Edited SAEs")
    for name, g in report["groups"].items():
        cos, length = g["final_cosine_with_original"], g["final_length_ratio"]
        print(f"  {name}: {g['edits']} edits; params hash matches replay: {g['params_sha256']['match']}; "
              f"w_dec bit-identical: {g['replay_bit_identical']}; other tensors unchanged: {all(g['other_tensors_unchanged'].values())}")
        print(f"    changed rows: {len(g['changed_rows'])} (only selected: {g['only_selected_rows_changed']}, "
              f"all selected: {g['all_selected_rows_changed']}); rule problems: {len(g['rule_problems'])}")
        print(f"    final cosine with original {cos['min']:.6f}–{cos['max']:.6f}; length ratio {length['min']:.7f}–{length['max']:.7f}")
        for arm, s in g["set_directions"].items():
            print(f"    {arm:9s} cos with original {s['cosine_with_original']:+.3f}, with d {s['cosine_with_d']:+.3f} "
                  f"(before {s['original_cosine_with_d']:+.3f})")

    llm = report["llm_run"]
    tc, c = llm["tokens_and_cost"], llm["counts"]
    print(f"\nLLM run: ended by {llm['outcome']['ended_by']}")
    print(f"  tokens: {tc['input_tokens']:,} in, {tc['output_tokens']:,} out; ${tc['cost_usd_at_list_price']:.2f} at list price "
          f"(matches run.json: {tc['matches_run_json']})")
    print(f"  {c['turns']} turns, {c['snippets']} snippets, {c['steer_test_calls']} steer_test calls, "
          f"{c['protocol_errors']} protocol errors, {c['rejected_edits']} rejected edits")
    print(f"  snippet errors {c['snippet_errors']}, timeouts {c['snippet_timeouts']}, restarts {c['kernel_restarts']}, "
          f"longest snippet {c['longest_snippet_seconds']:.1f} s")
    for s in llm["steer_tested_states"]:
        print(f"  steer_test #{s['call']} after {s['edits_so_far']} edits: {s['flips']} flips, {s['unclear']} unclear; "
              f"tested the final decoder: {s['is_final_decoder']}")
    print(f"  host-only text in messages to the LLM: {llm['host_only_text_sent_to_llm'] or 'none'}")
    print("  transcript scan (hits to read, not violations by themselves):")
    for name, hits in llm["scan"].items():
        print(f"    {name}: {len(hits)}" + "".join(f"\n      #{h['snippet']}: {h['match'][:110]}" for h in hits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, path in DEFAULT_RUNS.items():
        parser.add_argument(f"--{name}-run", type=Path, default=path, help=f"default {path}")
    args = parser.parse_args()
    run({name: getattr(args, f"{name}_run") for name in DEFAULT_RUNS})


if __name__ == "__main__":
    main()
