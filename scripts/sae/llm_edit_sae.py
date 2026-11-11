"""Phase 3 of the LLM-edited SAE experiment (cookbooks/LLM_EDITED_SAE_PROMPT.md): the harness.

Claude (on Bedrock) edits the decoder rows of the Pretrained SAE's 20 selected features by hand, in
a multi-turn conversation. Each turn it replies with exactly one of:

    a fenced python block   run in the Docker sandbox (no network, no training; mas_sae.llm_edit.sandbox),
                            output truncated and sent back
    steer_test()            the stage-6 "both" arm at r = 0.2 on fixed TRAIN episodes, run here on the
                            GPU with the current edited decoder vs. the unedited SAE
    DONE                    with a summary of what it changed and why

The saved SAE is always the original plus the edits the host accepted (edits.py): the sandbox's
kernel state never counts. Budgets (snippets, steer_test calls, a dollar cap on the conversation)
are enforced here and reported to Claude after every turn.

Decided in Phase 0 (see the cookbook): Claude Opus 5 at effort xhigh (Fable 5.2 and Opus 5.5 are not
callable on the school account); 40 snippets, 4 steer_test calls × 200 episodes; a dollar cap only as a
runaway guard. Claude is never shown validation, test or stage-6 results, the mean-difference direction
d, or the controls. steer_test is mas_sae.llm_edit.steer_test (shared with the controls).

Everything goes to results/sae/llm_edited/<run_id>/:
    transcript.jsonl   every message (assistant content blocks unchanged), snippet, output, steer_test
                       result and host-side diagnostic
    edits.py, rows/    the accepted edits (mas_sae.llm_edit.edit_rules)
    sae/               the edited SAE (params.safetensors + config.json, kind: gemma_scope)
    steer_test/        every steer_test generation (ResultsFile rows, as stage 6 writes them)
    outbox/            the sandbox's edit outbox and the container's stderr
    run.json           model, effort, budgets, brief, rules, git commit, input hashes, sandbox and
                       package provenance, cost, and how the run ended

    # Docker access in a shell started before joining the docker group: wrap in sg docker -c "..."
    PYTHONPATH=src python scripts/sae/llm_edit_sae.py --smoke     # 3 turns, 1 steer_test with n = 2
    PYTHONPATH=src python scripts/sae/llm_edit_sae.py             # the real run (ask before: it costs money)
"""

import argparse
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import boto3
from botocore.config import Config

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file, write_json_atomic
from mas_sae.experiments.records import append_jsonl
from mas_sae.intervention.experiment.config import load_config
from mas_sae.intervention.run_files.layout import new_run_id
from mas_sae.llm_edit.conversation import ConversationClient
from mas_sae.llm_edit.directions import direction_report, mean_difference_direction
from mas_sae.llm_edit.edit_rules import apply_edits, rules_record, write_edits_py
from mas_sae.llm_edit.edited_sae import SAE_DIR, original_sae, save_edited_sae
from mas_sae.llm_edit.sandbox import Sandbox, SandboxConfig, environment_record
from mas_sae.llm_edit.steer_test import TRAIN_CONTEXTS, TRAIN_INTERACTIONS, TRAIN_LABELS, Steerer

MODEL_ID = "us.anthropic.claude-opus-5"
EFFORT = "xhigh"
MAX_TOKENS_PER_TURN = 32_000          # covers thinking and the reply
# Anthropic list price for Claude Opus 5, $ per million tokens. Bedrock bills separately; the cap uses these.
PRICE_PER_MILLION = {"input": 5.0, "output": 25.0}
WRAP_UP_AT = 0.8                      # share of the dollar cap at which Claude is told to wrap up
OUTPUT_CHARS = 4_000                  # snippet output sent back to Claude: the first 3,000 and last 1,000

STUDY_CONFIG = Path("configs/intervention/intervention.yaml")   # read-only: solver, layer, seed, r, feature sets
SANDBOX_INPUTS = Path("artifacts/llm_edit/sandbox_inputs")      # Phase 2
OUTPUT_ROOT = Path("results/sae/llm_edited")


@dataclass(frozen=True)
class Budget:
    snippets: int
    steer_calls: int
    steer_episodes: int
    cost_cap_usd: float
    max_turns: int | None = None      # smoke runs only
    max_protocol_errors: int = 5      # replies that are not exactly one action


# The dollar cap only guards against a runaway loop (raised from $60 after Phase 3: money is not a
# constraint here); the snippet and steer_test budgets are what shape the run.
FULL_RUN = Budget(snippets=40, steer_calls=4, steer_episodes=200, cost_cap_usd=5000.0)
SMOKE_RUN = Budget(snippets=2, steer_calls=1, steer_episodes=2, cost_cap_usd=5.0, max_turns=3)

# Approved in Phase 0. Claude is not told about the controls and sees no stage-6 numbers.
BRIEF = """You are editing a sparse autoencoder (SAE) by hand. Setting: a Solver model answers a question, a \
Critic disagrees, and the Solver writes a second answer that either adopts the Critic's answer or keeps its own. \
The test your edit will face: at layer 25, once, at the final prompt token before the second answer, we replace \
the Solver's activation x with x' = x + 0.2·|x|·u. Here u is built from two feature sets of this SAE. The \
adoption set A and the retention set R each have one direction: the normalized sum of their features' unit \
decoder rows (u_A, u_R). In the "both" arm, an episode the Solver originally adopted gets u = normalize(u_R − u_A), \
and one it originally retained gets u = normalize(u_A − u_R). We count how often the second answer flips to the \
opposite behavior, compared with steering along random features of similar firing frequency.

Your goal: edit the decoder rows of the 10 adoption features and the 10 retention features in `selected`, so \
that this steering flips more answers in both directions.

Rules:
- Change only those 20 rows, and only through `apply_edit(...)`.
- Keep each row's length.
- Keep cosine ≥ 0.95 with its original row; `apply_edit` reports each row's cosine, and how the set directions \
changed.
- Measure anything you like on the train data.
- Do not fit models or optimize: no gradients, no regression, no automated search over values. Every number in \
an edit must be one you choose and can explain.

What you have:
- `X_train`, `y_train` (1 = adoption, 0 = retention), `meta_train`, and the SAE.
- `Z_train`, the train codes. They're fixed, since the encoder never changes.
- `encode`, `decode`, `separation(i)` and `top_examples(i, k)`.
- `hints`: auto-generated descriptions of the 20 features, from train examples and the decoder's logit lens. \
They may be wrong; check them. Many contrast "fictional vs. real" entities, which largely tracks the dataset \
source (synthetic vs. MuSiQue), not behavior.

`steer_test()` runs the real test on {steer_episodes} fixed train episodes. It returns flips and the mean change \
in the candidate margin, log p(the other answer's first token) − log p(the original answer's first token), on the \
episodes whose two answers start with different tokens, for your current edit vs. the unedited SAE. Flips are \
noisy at this size; weigh the margin too.

Budget: {snippets} code snippets, {steer_calls} `steer_test` calls and a token cap; you'll be told what's left \
after each turn. Each turn, reply with exactly one of: one fenced `python` block, `steer_test()`, or `DONE`. You \
will never see validation or test results.

Explain your reasoning before each edit. Finish with DONE and a summary: what you changed, why, and what you'd \
expect."""


def run(budget: Budget, smoke: bool) -> Path:
    study = load_config(STUDY_CONFIG)
    run_dir = OUTPUT_ROOT / (new_run_id() + ("-smoke" if smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=False)
    original_w_dec, sae_config, selected = original_sae(study.inputs.feature_sets)
    d = mean_difference_direction(SANDBOX_INPUTS)          # host-only diagnostic, never shown to Claude
    steerer = Steerer(study, budget.steer_episodes, run_dir / "steer_test")
    brief = BRIEF.format(**asdict(budget))
    client = ConversationClient(boto3.Session(profile_name=study.labeling.aws_profile), model_id=MODEL_ID,
                                region=study.labeling.region, effort=EFFORT,
                                config=Config(read_timeout=1800, retries={"max_attempts": 4, "mode": "standard"}))

    sandbox_config = SandboxConfig(SANDBOX_INPUTS, SAE_DIR, run_dir / "outbox")
    with Sandbox(sandbox_config, original_w_dec, selected) as sandbox:
        record = start_record(run_dir, budget, brief, selected, sandbox)
        conversation = Conversation(client, brief, budget, run_dir)
        outcome = {"ended_by": "interrupted"}      # replaced below unless the run is killed (e.g. Ctrl-C)
        try:
            outcome = conversation.run(sandbox, steerer, original_w_dec, selected, d)
        except Exception as error:      # keep what was accepted so far, then fail loudly
            outcome = {"ended_by": "error", "error": repr(error)}
            raise
        finally:
            finish_record(run_dir, record, outcome, conversation, sandbox, original_w_dec, sae_config, selected, d)
    print(f"Run {run_dir}: ended by {outcome['ended_by']}, {len(sandbox.edits)} accepted edits, "
          f"about ${conversation.cost_usd:.2f}.")
    return run_dir


# =====================================================================
# THE CONVERSATION
# =====================================================================

CODE_BLOCK = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)
STEER_CALL = re.compile(r"^\s*`?steer_test\(\s*\)`?\s*$", re.MULTILINE)
DONE_LINE = re.compile(r"^\s*\**DONE\b", re.MULTILINE)


class Conversation:
    """Append-only: every assistant response is kept with its content blocks unchanged."""

    def __init__(self, client: ConversationClient, brief: str, budget: Budget, run_dir: Path):
        self.client, self.brief, self.budget, self.run_dir = client, brief, budget, run_dir
        self.messages: list[dict] = []
        self.turns = self.snippets = self.steer_calls = self.protocol_errors = 0
        self.input_tokens = self.output_tokens = 0
        self.summary = None

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * PRICE_PER_MILLION["input"] + self.output_tokens * PRICE_PER_MILLION["output"]) / 1e6

    def run(self, sandbox: Sandbox, steerer: "Steerer", original_w_dec, selected, d) -> dict:
        self.say(first_message(sandbox.setup_output) + self.budget_note())
        while True:
            response = self.ask()
            if response["stop_reason"] == "refusal":
                return {"ended_by": "refusal", "stop_details": response.get("stop_details")}
            kind, payload = parse_action(response_text(response))
            if kind == "done":
                self.summary = payload
                return {"ended_by": "DONE"}
            reply = self.act(kind, payload, response, sandbox, steerer, original_w_dec, selected, d)
            if (stop := self.stop_reason()) is not None:
                self.log({"kind": "stopped", "reason": stop, "unsent_reply": reply})
                return {"ended_by": stop}
            self.say(reply + self.budget_note())

    def act(self, kind, payload, response, sandbox, steerer, original_w_dec, selected, d) -> str:
        """Carry out one action and return the message that reports it back."""
        cut_off = response["stop_reason"] == "max_tokens"
        if kind == "python" and not cut_off and self.snippets < self.budget.snippets:
            self.snippets += 1
            result = sandbox.run(payload)
            self.log({"kind": "snippet", "number": self.snippets, "code": payload, "output": result.output,
                      "error": result.error, "timed_out": result.timed_out, "restarted": result.restarted,
                      "seconds": result.seconds, "accepted_edits": [e.number for e in result.accepted_edits],
                      "rejected_edits": result.rejected_edits})
            if result.accepted_edits:
                write_edits_py(sandbox.edits, self.run_dir)
                edited = apply_edits(original_w_dec, sandbox.edits)
                self.log({"kind": "host_diagnostic", "after_edit": sandbox.edits[-1].number,
                          "set_directions": direction_report(edited, original_w_dec, selected, d)})
            return snippet_message(result)
        if kind == "steer" and self.steer_calls < self.budget.steer_calls:
            self.steer_calls += 1
            summary, text = steerer.test(apply_edits(original_w_dec, sandbox.edits), self.steer_calls)
            self.log({"kind": "steer_test", "number": self.steer_calls, "edits_so_far": len(sandbox.edits),
                      "summary": summary})
            return text
        self.protocol_errors += 1
        problem = ("Your reply was cut off at the token limit, so nothing was run." if cut_off
                   else "No code snippets are left." if kind == "python"
                   else "No steer_test calls are left." if kind == "steer" else payload)
        self.log({"kind": "protocol_error", "problem": problem})
        return f"[Not run] {problem} Reply with exactly one of: one fenced python block, steer_test(), or DONE."

    def stop_reason(self) -> str | None:
        b = self.budget
        if self.cost_usd >= b.cost_cap_usd:
            return "cost cap"
        if b.max_turns is not None and self.turns >= b.max_turns:
            return "turn limit (smoke run)"
        if self.protocol_errors >= b.max_protocol_errors:
            return "too many replies that were not exactly one action"
        return None

    def budget_note(self) -> str:
        b = self.budget
        note = (f"\n\n---\nBudget left: {b.snippets - self.snippets} code snippets, "
                f"{b.steer_calls - self.steer_calls} steer_test calls, "
                f"about {100 * (1 - self.cost_usd / b.cost_cap_usd):.0f}% of the token cap.")
        if self.snippets >= b.snippets and self.steer_calls >= b.steer_calls:
            note += " Everything is used: reply DONE with your summary."
        elif self.cost_usd >= WRAP_UP_AT * b.cost_cap_usd:
            note += " The token cap is close: make your final edits and reply DONE with your summary soon."
        if b.max_turns is not None:
            note += f" (Smoke test: {b.max_turns - self.turns} turns left.)"
        return note

    def say(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.log({"kind": "user", "content": text})

    def ask(self) -> dict:
        response = self.client.send(self.messages, system=self.brief, max_tokens=MAX_TOKENS_PER_TURN)
        self.turns += 1
        usage = response["usage"]
        self.input_tokens += usage["input_tokens"] + usage.get("cache_creation_input_tokens", 0) + \
            usage.get("cache_read_input_tokens", 0)
        self.output_tokens += usage["output_tokens"]
        self.messages.append({"role": "assistant", "content": response["content"]})   # unchanged
        self.log({"kind": "assistant", "turn": self.turns, "content": response["content"], "usage": usage,
                  "stop_reason": response["stop_reason"], "cost_usd_so_far": round(self.cost_usd, 4)})
        return response

    def log(self, entry: dict) -> None:
        append_jsonl(self.run_dir / "transcript.jsonl", entry)


def parse_action(text: str) -> tuple[str, str]:
    """("python", code) | ("steer", "") | ("done", summary) | ("error", what was wrong)."""
    blocks = CODE_BLOCK.findall(text)
    prose = CODE_BLOCK.sub("", text)
    actions = [kind for kind, present in (("python", bool(blocks)), ("steer", bool(STEER_CALL.search(prose))),
                                          ("done", bool(DONE_LINE.search(prose)))) if present]
    if len(blocks) > 1:
        return "error", f"Your reply had {len(blocks)} python blocks; send one per turn."
    if len(actions) != 1:
        return "error", ("Your reply had no action." if not actions
                         else f"Your reply had more than one action ({', '.join(actions)}).")
    if actions[0] == "python":
        return "python", blocks[0]
    return actions[0], (prose[DONE_LINE.search(prose).start():].strip() if actions[0] == "done" else "")


def response_text(response: dict) -> str:
    return "".join(block["text"] for block in response["content"] if block["type"] == "text")


def first_message(setup_output: str) -> str:
    return ("The sandbox is ready. Its startup output:\n\n" + setup_output.strip() +
            "\n\nThe preloaded variables and helpers are described in your instructions; help(name) shows each "
            "helper's details. `hints` holds the 20 features' descriptions. "
            "`apply_edit(feature, new_row, why)` takes one feature id, a 2560-number row and a reason. "
            "Send one action, then stop and wait: the output comes in the next message. Never write an output "
            "yourself. Your first turn.")


def snippet_message(result) -> str:
    parts = [truncate(result.output) if result.output.strip() else "(no output)"]
    if result.error:
        parts.append("Error:\n" + truncate(result.error))
    if result.accepted_edits:
        parts.append("The host accepted edit(s) " + ", ".join(f"{e.number} (feature {e.feature})"
                                                              for e in result.accepted_edits) + ".")
    if result.rejected_edits:
        parts.append("The host REJECTED: " + "; ".join(result.rejected_edits))
    return "\n\n".join(parts)


def truncate(text: str) -> str:
    if len(text) <= OUTPUT_CHARS:
        return text
    head, tail = text[:3 * OUTPUT_CHARS // 4], text[-OUTPUT_CHARS // 4:]
    return f"{head}\n[... {len(text) - len(head) - len(tail)} characters cut ...]\n{tail}"


# =====================================================================
# THE RUN RECORD
# =====================================================================

def start_record(run_dir: Path, budget: Budget, brief: str, selected: dict, sandbox: Sandbox) -> dict:
    inputs = [SAE_DIR / "params.safetensors", SAE_DIR / "config.json", SANDBOX_INPUTS / "inputs.json",
              TRAIN_INTERACTIONS, TRAIN_CONTEXTS, TRAIN_LABELS / "run_settings.json", STUDY_CONFIG]
    record = {
        "run_id": run_dir.name, "command": get_run_command(), "git_commit": get_git_commit(),
        "git_status": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.splitlines(),
        "model": MODEL_ID, "effort": EFFORT, "max_tokens_per_turn": MAX_TOKENS_PER_TURN,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "price_per_million_tokens_usd": PRICE_PER_MILLION, "budget": asdict(budget), "brief": brief,
        "rules": rules_record(selected), "output_chars": OUTPUT_CHARS,
        "inputs_sha256": {str(p): sha256_file(p) for p in inputs},
        "sandbox": sandbox.provenance(), "sandbox_environment": environment_record(),
    }
    write_json_atomic(run_dir / "run.json", record)
    return record


def finish_record(run_dir, record, outcome, conversation, sandbox, original_w_dec, sae_config, selected, d):
    if sandbox.edits:
        write_edits_py(sandbox.edits, run_dir)
    edited = apply_edits(original_w_dec, sandbox.edits)
    sae_dir = save_edited_sae(edited, sae_config, run_dir / "sae",
                              {"edit_source": "LLM (scripts/sae/llm_edit_sae.py)", "edits": str(run_dir / "edits.py")})
    write_json_atomic(run_dir / "run.json", {
        **record, "outcome": outcome, "turns": conversation.turns, "snippets": conversation.snippets,
        "steer_calls": conversation.steer_calls, "protocol_errors": conversation.protocol_errors,
        "input_tokens": conversation.input_tokens, "output_tokens": conversation.output_tokens,
        "cost_usd_at_list_price": round(conversation.cost_usd, 4), "accepted_edits": len(sandbox.edits),
        "edited_rows": sorted({e.feature for e in sandbox.edits}), "claude_summary": conversation.summary,
        "edited_sae_sha256": {f.name: sha256_file(f) for f in sorted(sae_dir.iterdir())},
        "set_directions": direction_report(edited, original_w_dec, selected, d),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="3 turns, 1 steer_test with n = 2, $5 cap")
    args = parser.parse_args()
    run(SMOKE_RUN if args.smoke else FULL_RUN, args.smoke)


if __name__ == "__main__":
    main()
