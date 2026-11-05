"""Labeling the regenerated A2s with the exact Claude labeler that produced the stored labels.

The labeler is Claude on AWS Bedrock with the settings recorded in the stored-label run's
``run_settings.json`` (model, prompt version and hash, temperature, max_tokens, attempts). They are
read from that file, and labeling refuses to start if the current prompt's sha256 differs: a
different labeler would make some "flips" labeler disagreement. The returned model version must be
one the stored-label run saw.

Each request is one A2, built by ``claude_validator.build_labeling_prompt`` from the stored question,
A1 and C1, the candidates display, and the regenerated A2. Claude is never told the condition, that
the answer came from an intervention, the stored label or the stored A2 (blind labeling).

  - An A2 identical to the stored A2 reuses the stored label.
  - Identical answers to the same episode share one request.
  - Every response is kept, with Claude's reason, under ``labeling/attempts/``; a rerun reuses them.
  - A failed request is retried once more; if it still fails, its episode is dropped from every
    condition, and why is recorded. A failure is never counted as retention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mas_sae.evaluation import claude_validator as cv
from mas_sae.experiments.artifacts import sha256_json
from mas_sae.experiments.records import read_jsonl
from mas_sae.utils.versioning import create_versioned_run_dir

PRESENTATION = "candidates"
LABEL_ROUNDS = 2            # the first attempt and one retry of every failed request


@dataclass(frozen=True)
class LabelerProtocol:
    model_id: str
    client: str
    max_tokens: int
    temperature: float
    prompt_version: str
    prompt_sha256: str
    max_attempts: int
    models_returned: tuple      # model versions the stored-label run got back


def read_protocol(labels_dir: Path) -> LabelerProtocol:
    settings = json.loads((Path(labels_dir) / "run_settings.json").read_text())
    summary = json.loads((Path(labels_dir) / "summary.json").read_text())
    return LabelerProtocol(
        model_id=settings["model_id"], client=settings["client"], max_tokens=settings["max_tokens"],
        temperature=settings["temperature"], prompt_version=settings["prompt_version"],
        prompt_sha256=settings["prompt_sha256"], max_attempts=settings["max_attempts"],
        models_returned=tuple(summary.get("models_returned", [])))


def protocol_problems(protocol: LabelerProtocol) -> list[str]:
    """Why the current labeler would not be the one that produced the stored labels."""
    problems = []
    if protocol.prompt_version != cv.prompt_version(True, PRESENTATION):
        problems.append(f"stored labels used prompt {protocol.prompt_version!r}, "
                        f"the current prompt is {cv.prompt_version(True, PRESENTATION)!r}")
    if protocol.prompt_sha256 != cv.prompt_sha256(True, PRESENTATION):
        problems.append("the current labeling prompt's sha256 differs from the stored labels' "
                        f"({cv.prompt_sha256(True, PRESENTATION)[:12]}… vs {protocol.prompt_sha256[:12]}…)")
    if protocol.client != "ClaudeClient":
        problems.append(f"stored labels used client {protocol.client!r}; only ClaudeClient is supported")
    if not protocol.models_returned:
        problems.append("the stored-label run recorded no returned model version")
    return problems


def create_client(protocol: LabelerProtocol, labeling_config):
    import boto3
    from botocore.config import Config
    from mas_sae.models.claude import ClaudeClient
    session = boto3.Session(profile_name=labeling_config.aws_profile)
    return ClaudeClient(session=session, model_id=protocol.model_id, region=labeling_config.region,
                        config=Config(max_pool_connections=labeling_config.concurrency,
                                      retries={"mode": "adaptive", "max_attempts": 8}))


def labeling_request(episode, answer: str) -> dict:
    """What Claude sees: the stored question, A1 and C1, the candidates display, and the new A2. Nothing else."""
    return {"question_id": episode.question_id, "question": episode.question,
            "solver_attempt_1": episode.solver_attempt_1, "critic_advocated_answer": episode.critic_advocated_answer,
            "solver_attempt_2": answer, "a2_presentation": PRESENTATION, "candidate_order": episode.candidate_order,
            "critic_noncommittal": False}


def request_key(protocol: LabelerProtocol, request: dict) -> str:
    """Same protocol and same inputs: same key, so identical answers share one request."""
    return sha256_json({"protocol": protocol.__dict__, "request": request})


def read_cached_labels(attempts_dir: Path) -> dict[str, dict]:
    """Every saved response, by request key; a later successful attempt replaces an earlier failure."""
    cached = {}
    for path in sorted(Path(attempts_dir).glob("*/raw_results.jsonl")):
        for row in read_jsonl(path):
            if row["episode_id"] not in cached or row.get("behavior_label") in cv.VALID_LABELS:
                cached[row["episode_id"]] = row
    return cached


def label_rows(rows: list[dict], episodes: dict, protocol: LabelerProtocol, work_dir: Path,
               client_factory, concurrency: int) -> tuple[list[dict], dict]:
    """(rows with labels, {episode_id: why it was excluded}). Rows of excluded episodes are dropped."""
    requests, row_requests = {}, {}
    for row in rows:
        episode = episodes[row["episode_id"]]
        if row["answer"] == episode.stored_solver_attempt_2:
            continue
        request = labeling_request(episode, row["answer"])
        key = request_key(protocol, request)
        requests[key] = {**request, "episode_id": key}
        row_requests[(row["episode_id"], row["condition"])] = key

    attempts_dir = Path(work_dir) / "attempts"
    cached = read_cached_labels(attempts_dir)
    client = None
    for _ in range(LABEL_ROUNDS):
        pending = [r for k, r in requests.items() if cached.get(k, {}).get("behavior_label") not in cv.VALID_LABELS]
        if not pending:
            break
        client = client or client_factory()
        attempt = create_versioned_run_dir(attempts_dir)
        result = cv.validate_batch(client, pending, output_path=attempt / "raw_results.jsonl",
                                   max_tokens=protocol.max_tokens, temperature=protocol.temperature,
                                   max_attempts=protocol.max_attempts, include_question=True,
                                   concurrency=concurrency)
        cached.update({r["episode_id"]: r for r in result["results"]})
    if client is not None:
        client.close()

    labeled, excluded = [], {}
    for row in rows:
        episode = episodes[row["episode_id"]]
        key = row_requests.get((row["episode_id"], row["condition"]))
        if key is None:
            labeled.append({**row, "label": episode.stored_label, "label_reason": episode.stored_reason,
                            "label_source": "stored_answer", "label_request": None})
            continue
        response = cached[key]
        if response.get("behavior_label") not in cv.VALID_LABELS:
            excluded.setdefault(row["episode_id"], f"labeling failed: {response.get('exclusion_reason')}")
            continue
        if response.get("claude_model") not in protocol.models_returned:
            raise ValueError(f"Claude returned {response.get('claude_model')!r}, not a model version the "
                             f"stored labels came from {list(protocol.models_returned)}.")
        labeled.append({**row, "label": response["behavior_label"], "label_reason": response.get("claude_reason"),
                        "label_source": "claude", "label_request": key})
    return [r for r in labeled if r["episode_id"] not in excluded], excluded


class StubClaudeClient:
    """For smoke runs only: answers like Claude without calling it, so merge can be tested offline.

    Labels an answer "retention" if it contains A1's text, "adoption" if it contains C1's, else "out".
    Results produced with it are meaningless and are marked ``labels_stubbed`` in the summary."""

    def __init__(self, model: str):
        self.model = model

    def close(self):
        pass

    def generate(self, prompt: str, **kwargs):
        from mas_sae.models.claude import ClaudeResult
        payload = json.loads(prompt[prompt.rindex("\n{") + 1:])
        a2 = payload["A2"].lower()
        label = ("retention" if payload["A1"].lower() in a2 else "adoption" if payload["C1"].lower() in a2
                 else "out")
        return ClaudeResult(text=json.dumps({"label": label, "reason": "stub"}), model=self.model,
                            input_tokens=0, output_tokens=0, task="labeling", stop_reason="end_turn")
