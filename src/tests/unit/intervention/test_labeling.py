"""Labeling: stored answers reused, identical answers shared, blind requests, failures excluded, resume."""

import json

import pytest

from . import fakes
from mas_sae.intervention.analysis import labeling
from mas_sae.models.claude import ClaudeResult

MODEL = "claude-sonnet-4-5-20250929"
PROTOCOL = labeling.LabelerProtocol(model_id="m", client="ClaudeClient", max_tokens=512, temperature=0.0,
                                    prompt_version="v", prompt_sha256="s", max_attempts=1, models_returned=(MODEL,))


class FakeClaude:
    """Labels every answer "adoption"; ``fail`` answers get a truncated response instead."""

    def __init__(self, fail=(), model=MODEL):
        self.prompts, self.fail, self.model = [], set(fail), model

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        a2 = json.loads(prompt[prompt.rindex("\n{") + 1:])["A2"]
        stop = "max_tokens" if a2 in self.fail else "end_turn"
        return ClaudeResult(text=json.dumps({"label": "adoption", "reason": f"why {a2}"}), model=self.model,
                            input_tokens=1, output_tokens=1, task="labeling", stop_reason=stop)

    def close(self):
        pass


def _rows(answers_by_episode):
    return [{"episode_id": e, "condition": f"c{k}", "answer": a}
            for e, answers in answers_by_episode.items() for k, a in enumerate(answers)]


def _episodes():
    return {"ep0": fakes.episode(0, "retention", stored_a2="Paris"), "ep1": fakes.episode(1, "retention", stored_a2="Paris")}


def test_stored_answers_reuse_the_stored_label_and_identical_answers_share_a_request(tmp_path):
    client = FakeClaude()
    rows = _rows({"ep0": ["Paris", "Rome", "Rome"], "ep1": ["Rome"]})
    labeled, excluded = labeling.label_rows(rows, _episodes(), PROTOCOL, tmp_path, lambda: client, concurrency=2)
    assert excluded == {} and len(client.prompts) == 2              # one request per (episode, distinct answer)
    by = {(r["episode_id"], r["condition"]): r for r in labeled}
    assert by[("ep0", "c0")]["label"] == "retention" and by[("ep0", "c0")]["label_source"] == "stored_answer"
    assert by[("ep0", "c1")]["label"] == "adoption" and by[("ep0", "c1")]["label_reason"] == "why Rome"
    assert by[("ep0", "c1")]["label_request"] == by[("ep0", "c2")]["label_request"]


def test_requests_are_blind_to_the_condition_and_the_stored_answer(tmp_path):
    client = FakeClaude()
    rows = [{"episode_id": "ep0", "condition": "edit|L1 SAE|both|targeted||0.05", "answer": "Rome"}]
    labeling.label_rows(rows, {"ep0": fakes.episode(0, "retention", stored_a2="Stored answer text")},
                        PROTOCOL, tmp_path, lambda: client, concurrency=1)
    sent = json.loads(client.prompts[0][client.prompts[0].rindex("\n{") + 1:])
    assert set(sent) == {"question", "A1", "C1", "A2", "candidates_shown_to_solver"}
    assert "Stored answer text" not in client.prompts[0] and "L1 SAE" not in client.prompts[0]
    assert "stored reason" not in client.prompts[0]


def test_a_request_that_keeps_failing_drops_its_whole_episode(tmp_path):
    client = FakeClaude(fail={"Bad"})
    rows = _rows({"ep0": ["Rome", "Bad"], "ep1": ["Rome"]})
    labeled, excluded = labeling.label_rows(rows, _episodes(), PROTOCOL, tmp_path, lambda: client, concurrency=1)
    assert list(excluded) == ["ep0"] and "incomplete_response" in excluded["ep0"]
    assert {r["episode_id"] for r in labeled} == {"ep1"}             # ep0 is gone from every condition
    assert sum('"Bad"' in p for p in client.prompts) == labeling.LABEL_ROUNDS   # retried once


def test_a_rerun_reuses_every_saved_response(tmp_path):
    rows = _rows({"ep0": ["Rome", "Berlin"]})
    labeling.label_rows(rows, _episodes(), PROTOCOL, tmp_path, lambda: FakeClaude(), concurrency=1)
    second = FakeClaude()
    labeled, _ = labeling.label_rows(rows, _episodes(), PROTOCOL, tmp_path, lambda: second, concurrency=1)
    assert second.prompts == [] and len(labeled) == 2


def test_a_different_returned_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a model version the stored labels came from"):
        labeling.label_rows(_rows({"ep0": ["Rome"]}), _episodes(), PROTOCOL, tmp_path,
                            lambda: FakeClaude(model="claude-other"), concurrency=1)


def test_the_stub_client_labels_by_containment(tmp_path):
    rows = _rows({"ep0": ["I say Paris", "Rome!", "Lyon"]})
    labeled, _ = labeling.label_rows(rows, {"ep0": fakes.episode(0, "retention", stored_a2="x")}, PROTOCOL,
                                     tmp_path, lambda: labeling.StubClaudeClient(MODEL), concurrency=1)
    assert [r["label"] for r in labeled] == ["retention", "adoption", "out"]
