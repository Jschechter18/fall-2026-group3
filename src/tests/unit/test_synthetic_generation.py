import json
import random
import re

import pytest

from mas_sae.data.synthetic import validate_candidate
from mas_sae.data.synthetic.generation import (
    ParagraphPool, assemble_candidate, build_prompt, generate_one, parse_response)
from mas_sae.data.synthetic.generation_cli import main
from mas_sae.models.claude import ClaudeResult
from mas_sae.evaluation.scoring import answer_matches


def paragraph(title):
    return {"title": title, "paragraph_text": f"{title} is a fictional entity used for testing."}


RAW_2HOP = {
    "question": "When was the town containing Luma Observatory founded?",
    "answer": "1842",
    "answer_aliases": [],
    "supporting_paragraphs": [paragraph("Luma Observatory"), paragraph("Selin")],
    "question_decomposition": [
        {"question": "In which town is Luma Observatory located?", "answer": "Selin", "support": 0},
        {"question": "When was #1 founded?", "answer": "1842", "support": 1},
    ],
    "distractors": [paragraph("Selin River"), paragraph("Varro Valley")],
}

# 4hop3: 1 -> 2, 3 independent, 2 and 3 -> 4. Supports listed out of step order on purpose.
RAW_4HOP3 = {
    "question": "Which nation sent most migrants to the port city from the sea bordering the island with Kell as capital?",
    "answer": "Ostria",
    "answer_aliases": ["Republic of Ostria"],
    "supporting_paragraphs": [paragraph("Kell"), paragraph("Marrow Island"),
                              paragraph("Port Dane"), paragraph("Dane Migration")],
    "question_decomposition": [
        {"question": "Kell >> capital of", "answer": "Marrow Island", "support": 0},
        {"question": "#1 >> located in sea", "answer": "Grey Sea", "support": 1},
        {"question": "Which port city is the largest in the region?", "answer": "Port Dane", "support": 2},
        {"question": "Which nation in the #2 sent most migrants to #3 ?", "answer": "Ostria", "support": 3},
    ],
    "distractors": [paragraph("Grey Sea Fisheries")],
}


@pytest.fixture
def pool():
    return ParagraphPool([paragraph(f"Filler {i}") for i in range(40)])


class FakeClient:
    model_id = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []

    def generate(self, prompt, *, system="", max_tokens=4096, temperature=0.0, task):
        self.calls += 1
        self.prompts.append(prompt)
        text, stop_reason = self.responses.pop(0)
        return ClaudeResult(text=text, model="fake-model", input_tokens=10, output_tokens=20,
                            task=task, stop_reason=stop_reason)

    def close(self):
        pass


@pytest.mark.parametrize("raw,hop_type", [(RAW_2HOP, "2hop"), (RAW_4HOP3, "4hop3")])
def test_assembled_candidate_is_valid(raw, hop_type, pool):
    candidate = assemble_candidate(raw, hop_type, pool, random.Random(0))
    assert validate_candidate(candidate, expected_paragraphs=20).valid, validate_candidate(candidate).errors
    assert candidate["id"].startswith(f"{hop_type}__synthetic_")
    for step, raw_step in zip(candidate["question_decomposition"], raw["question_decomposition"]):
        support = candidate["paragraphs"][step["paragraph_support_idx"]]
        assert support["is_supporting"]
        assert support["title"] == raw["supporting_paragraphs"][raw_step["support"]]["title"]


def test_fillers_skip_used_titles_and_answer_mentions():
    pool = ParagraphPool([paragraph("Selin"), {"title": "Leak", "paragraph_text": "Founded in 1842."}]
                         + [paragraph(f"Filler {i}") for i in range(16)])
    candidate = assemble_candidate(RAW_2HOP, "2hop", pool, random.Random(0))
    titles = [p["title"] for p in candidate["paragraphs"]]
    assert titles.count("Selin") == 1 and "Leak" not in titles


# ---- related fillers (#156): the pool paragraphs most similar to the question

def text(title, body):
    return {"title": title, "paragraph_text": body}


TOPICAL = [text(f"Observatory {i}", f"The observatory {i} houses a refracting telescope founded by astronomers "
                                    f"in a hill town; its dome and telescope survey comets.") for i in range(30)]
UNRELATED = [text(f"Album {i}", f"Album {i} is a rock record with guitar songs released by a band on tour.")
             for i in range(30)]


def test_random_filler_draws_exactly_as_before():
    paragraphs = [paragraph(f"Filler {i}") for i in range(40)]
    pool = ParagraphPool(paragraphs)
    assert pool.filler == "random"
    expected = ParagraphPool(paragraphs).sample(16, random.Random(7), exclude_titles={"Selin"}, exclude_text="1842")
    assert pool.fillers(16, random.Random(7), raw=RAW_2HOP, exclude_titles={"Selin"}) == expected


def test_related_fillers_prefer_paragraphs_about_the_question():
    pool = ParagraphPool(UNRELATED + TOPICAL, filler="related", related_top=20)
    fillers = pool.fillers(16, random.Random(0), raw=RAW_2HOP, exclude_titles=set())
    assert all(p["title"].startswith("Observatory") for p in fillers)


@pytest.mark.parametrize("leak", ["Selin was its earlier name.", "It opened in 1842.", "Its site, SELIN, is old."])
def test_related_fillers_never_mention_a_hop_answer(leak):
    """A topical paragraph naming any step answer (normalized) could contradict the evidence."""
    raw = dict(RAW_2HOP, answer_aliases=["the year 1842"])
    tainted = text("Observatory leak", f"The observatory telescope dome survey founded astronomers. {leak}")
    pool = ParagraphPool([tainted] + TOPICAL + UNRELATED, filler="related", related_top=40)
    fillers = pool.fillers(16, random.Random(0), raw=raw, exclude_titles=set())
    assert "Observatory leak" not in [p["title"] for p in fillers]


def test_related_fillers_repeat_for_the_same_seed():
    pool = ParagraphPool(UNRELATED + TOPICAL, filler="related", related_top=25)
    draw = lambda seed: pool.fillers(16, random.Random(seed), raw=RAW_2HOP, exclude_titles=set())
    assert draw(3) == draw(3)


def test_related_candidate_is_valid_and_keeps_used_titles_out():
    pool = ParagraphPool([text("Selin River", "An observatory telescope near the river.")] + UNRELATED + TOPICAL,
                         filler="related")
    candidate = assemble_candidate(RAW_2HOP, "2hop", pool, random.Random(0))
    assert validate_candidate(candidate, expected_paragraphs=20).valid
    assert [p["title"] for p in candidate["paragraphs"]].count("Selin River") == 1   # the generated distractor


def test_unknown_filler_and_small_related_pool_are_refused():
    with pytest.raises(ValueError, match="filler must be one of"):
        ParagraphPool([], filler="nearest")
    with pytest.raises(ValueError, match="pool too small"):
        ParagraphPool(TOPICAL[:5], filler="related").fillers(16, random.Random(0), raw=RAW_2HOP, exclude_titles=set())


def test_small_pool_raises():
    with pytest.raises(ValueError):
        assemble_candidate(RAW_2HOP, "2hop", ParagraphPool([]), random.Random(0))


def test_parse_response_fenced_and_plain():
    text = json.dumps(RAW_2HOP)
    assert parse_response(text) == RAW_2HOP
    assert parse_response(f"```json\n{text}\n```") == RAW_2HOP
    with pytest.raises(ValueError):
        parse_response("[1, 2]")
    with pytest.raises(ValueError):
        parse_response("Sure! Here is the JSON:")


def test_build_prompt_rejects_unknown_hop_type():
    assert "#1 and #2" in build_prompt("3hop2")
    with pytest.raises(ValueError):
        build_prompt("5hop")


def test_generate_one_retries_then_succeeds(pool):
    client = FakeClient([("not json", "end_turn"), (json.dumps(RAW_2HOP), "end_turn")])
    candidate, provenance = generate_one(client, "2hop", pool, random.Random(0))
    assert candidate is not None and client.calls == 2
    assert provenance["record_id"] == candidate["id"]
    assert provenance["attempts"][0]["error"].startswith("JSONDecodeError")
    assert provenance["attempts"][1]["precheck_errors"] == []


def test_chain_rejected_for_branched_hop_type(pool):
    chain = dict(RAW_4HOP3, question_decomposition=[
        dict(step, question=q) for step, q in zip(RAW_4HOP3["question_decomposition"], [
            "Kell >> capital of", "#1 >> located in sea", "Largest port on #2?", "Who sent migrants to #3?"])])
    candidate = assemble_candidate(chain, "4hop3", pool, random.Random(0))
    result = validate_candidate(candidate)
    assert not result.valid
    assert {i.field for i in result.errors if i.code == "hop_shape"} == {
        "question_decomposition[2].question", "question_decomposition[3].question"}
    assert validate_candidate(
        assemble_candidate(RAW_4HOP3, "4hop3", pool, random.Random(0))
    ).valid


def test_alias_equal_to_answer_is_dropped(pool):
    raw = dict(RAW_2HOP, answer_aliases=["1842 ", "eighteen forty-two"])
    assert assemble_candidate(raw, "2hop", pool, random.Random(0))["answer_aliases"] == ["eighteen forty-two"]


def test_generate_one_gives_up_on_truncation_and_invalid(pool):
    broken = dict(RAW_2HOP, answer="1843")  # final step answer mismatch
    client = FakeClient([("{", "max_tokens"), (json.dumps(broken), "end_turn")])
    candidate, provenance = generate_one(client, "2hop", pool, random.Random(0))
    assert candidate is None
    assert provenance["attempts"][0]["error"] == "stop_reason=max_tokens"
    assert {e["code"] for e in provenance["attempts"][1]["precheck_errors"]} == {"final_answer"}


def test_cli_writes_outputs(tmp_path):
    original = tmp_path / "train.json"
    original.write_text(json.dumps({"paragraphs": [paragraph(f"Filler {i}") for i in range(30)]}) + "\n")
    client = FakeClient([(json.dumps(RAW_2HOP), "end_turn"), ("nope", "end_turn"), ("nope", "end_turn")])
    out = tmp_path / "run"
    assert main(["--no-dataset-audit", "--n", "2", "--hop-types", "2hop", "--seed", "1", "--original", str(original),
                 "--output-dir", str(out)], client=client) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["generated"] == 1 and summary["failed"] == 1 and summary["api_calls"] == 3
    assert len((out / "candidates.jsonl").read_text().splitlines()) == 1
    assert len((out / "provenance.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("support", [0.0, True, False, -1, 2, "0", None, [], {}])
def test_invalid_raw_support_is_rejected(pool, support):
    raw = json.loads(json.dumps(RAW_2HOP))
    raw["question_decomposition"][0]["support"] = support
    with pytest.raises(ValueError, match="support"):
        assemble_candidate(raw, "2hop", pool, random.Random(0))


@pytest.mark.parametrize("error", [AttributeError("broken code"), TimeoutError("API timeout")])
def test_cli_preserves_attempts_and_closes_client_on_fatal_error(tmp_path, monkeypatch, error):
    from mas_sae.data.synthetic import generation_cli

    class BrokenClient(FakeClient):
        closed = False

        def generate(self, *args, **kwargs):
            if self.calls == 1:
                raise error
            return super().generate(*args, **kwargs)

        def close(self):
            self.closed = True

    client = BrokenClient([("not json", "end_turn")])
    monkeypatch.setattr(generation_cli, "make_client", lambda model_id, **kwargs: client)
    original = tmp_path / "train.json"
    original.write_text(json.dumps({"paragraphs": [paragraph(f"Filler {i}") for i in range(30)]}) + "\n")
    out = tmp_path / "run"
    with pytest.raises(type(error), match=str(error)):
        main(["--no-dataset-audit", "--n", "2", "--hop-types", "2hop", "--original", str(original),
              "--output-dir", str(out)])

    provenance = json.loads((out / "provenance.jsonl").read_text())
    assert len(provenance["attempts"]) == 1
    assert provenance["attempts"][0]["raw_text"] == "not json"
    assert provenance["error"] == f"{type(error).__name__}: {error}"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failed"] == 1
    assert summary["api_calls"] == 2
    assert summary["completed_responses"] == 1
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 20
    assert client.closed
    assert (out / "candidates.jsonl").read_text() == ""


def test_cli_records_client_initialization_failure(tmp_path, monkeypatch):
    from mas_sae.data.synthetic import generation_cli

    def fail(model_id, **kwargs):
        raise RuntimeError("client initialization failed")

    monkeypatch.setattr(generation_cli, "make_client", fail)
    original = tmp_path / "train.json"
    original.write_text(json.dumps({"paragraphs": [paragraph("Filler")]}) + "\n")
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="client initialization failed"):
        main(["--no-dataset-audit", "--original", str(original), "--output-dir", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["api_calls"] == 0


def test_cli_marks_output_failure_before_starting_a_client(tmp_path, monkeypatch):
    from pathlib import Path
    from mas_sae.data.synthetic import generation_cli

    class TrackingClient(FakeClient):
        closed = False

        def close(self):
            self.closed = True

    client = TrackingClient([])
    monkeypatch.setattr(generation_cli, "make_client", lambda model_id, **kwargs: client)
    original = tmp_path / "train.json"
    original.write_text(json.dumps({"paragraphs": [paragraph("Filler")]}) + "\n")
    out = tmp_path / "run"
    original_open = Path.open

    def fail_candidate_open(path, *args, **kwargs):
        if path == out / "candidates.jsonl":
            raise OSError("cannot create candidates")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_candidate_open)
    with pytest.raises(OSError, match="cannot create candidates"):
        main(["--no-dataset-audit", "--original", str(original), "--output-dir", str(out)])

    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["fatal_error"] == "OSError: cannot create candidates"
    assert summary["api_calls"] == 0
    assert not client.closed
    assert client.calls == 0


@pytest.mark.parametrize("hop_type", ["2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"])
def test_prompt_example_matches_hop_subtype(pool, hop_type):
    from mas_sae.data.synthetic.prompt_examples import build_example

    example = build_example(hop_type)
    candidate = assemble_candidate(example, hop_type, pool, random.Random(0))
    result = validate_candidate(candidate)
    assert result.valid, result.errors
    prompt = build_prompt(hop_type)
    assert json.dumps(example, indent=2) in prompt
    assert prompt.rstrip().endswith("Return JSON only.")


@pytest.mark.parametrize("failure", ["parse", "truncation", "support", "validation"])
def test_retry_contains_rejected_response_and_specific_errors(pool, failure):
    broken = json.loads(json.dumps(RAW_2HOP))
    stop_reason = "end_turn"
    if failure == "parse":
        rejected_text = "not json"
        expected_error = "JSONDecodeError"
    elif failure == "truncation":
        rejected_text = '{"question":'
        stop_reason = "max_tokens"
        expected_error = "stop_reason=max_tokens"
    elif failure == "support":
        broken["question_decomposition"][0]["support"] = False
        rejected_text = json.dumps(broken)
        expected_error = "support must be an integer"
    else:
        broken["answer"] = "1843"
        rejected_text = json.dumps(broken)
        expected_error = "final_answer"

    client = FakeClient([
        (rejected_text, stop_reason),
        (json.dumps(RAW_2HOP), "end_turn"),
    ])
    candidate, provenance = generate_one(client, "2hop", pool, random.Random(0))
    assert candidate is not None
    assert client.calls == 2
    assert client.prompts[1].startswith(client.prompts[0])
    assert json.dumps(rejected_text) in client.prompts[1]
    assert expected_error in client.prompts[1]
    assert client.prompts[1].count("Your previous response was rejected.") == 1
    assert provenance["prompt_version"] == "v3"
    assert "topic" not in provenance
    assert provenance["attempts"][0]["raw_text"] == rejected_text


def test_repair_uses_latest_failure_without_nesting_prompts(pool):
    client = FakeClient([
        ("first invalid response", "end_turn"),
        ("second invalid response", "end_turn"),
        (json.dumps(RAW_2HOP), "end_turn"),
    ])
    candidate, provenance = generate_one(
        client, "2hop", pool, random.Random(0), max_attempts=3
    )
    assert candidate is not None
    assert len(provenance["attempts"]) == 3
    assert "second invalid response" in client.prompts[2]
    assert "first invalid response" not in client.prompts[2]
    assert client.prompts[2].count("Your previous response was rejected.") == 1


@pytest.mark.parametrize("field,value,error_field", [
    *(('answer', value, 'answer') for value in [1842, 18.42, None, True, "", "  ", [], {}]),
    *(('answer_aliases', value, 'answer_aliases') for value in
      ["MDCCC", "", None, 1842, False, {}, [""], ["  "], [1], [None], [True], ["MDCCC", 1]]),
    *(('step_answer', value, 'question_decomposition[0].answer') for value in
      [1842, 18.42, None, False, "", "  ", [], {}]),
])
def test_malformed_answer_fields_are_rejected_before_assembly_and_repaired(pool, field, value, error_field):
    raw = json.loads(json.dumps(RAW_2HOP))
    if field == "step_answer":
        raw["question_decomposition"][0]["answer"] = value
    else:
        raw[field] = value
    rejected_text = json.dumps(raw)

    # A malformed field must be reported before the empty filler pool is sampled.
    with pytest.raises(ValueError, match=re.escape(error_field)):
        assemble_candidate(raw, "2hop", ParagraphPool([]), random.Random(0))

    client = FakeClient([(rejected_text, "end_turn"), (json.dumps(RAW_2HOP), "end_turn")])
    candidate, provenance = generate_one(client, "2hop", pool, random.Random(0))
    assert candidate is not None
    assert validate_candidate(candidate).valid
    assert client.calls == provenance["api_calls"] == 2
    assert provenance["attempts"][0]["error"].startswith("ValueError:")
    assert error_field in provenance["attempts"][0]["error"]
    assert provenance["attempts"][0]["raw_text"] == rejected_text
    assert error_field in client.prompts[1]
    assert json.dumps(rejected_text) in client.prompts[1]


@pytest.mark.parametrize("field,value", [("answer", 1842), ("answer_aliases", "MDCCC")])
def test_malformed_answer_exhaustion_is_record_failure_not_batch_failure(tmp_path, field, value):
    raw = dict(RAW_2HOP, **{field: value})
    client = FakeClient([(json.dumps(raw), "end_turn")] * 2)
    original = tmp_path / "train.json"
    original.write_text(json.dumps({"paragraphs": [paragraph(f"Filler {i}") for i in range(30)]}) + "\n")
    output = tmp_path / "run"
    assert main(["--no-dataset-audit", "--n", "1", "--hop-types", "2hop", "--original", str(original),
                 "--output-dir", str(output)], client=client) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["failed"] == 1 and summary["api_calls"] == 2
    assert summary["generated"] == summary["accepted"] == 0
    assert (output / "accepted.jsonl").read_text() == ""
    assert (output / "candidates.jsonl").read_text() == ""
    provenance = json.loads((output / "provenance.jsonl").read_text())
    assert all(attempt["error"].startswith("ValueError:") for attempt in provenance["attempts"])


@pytest.mark.parametrize("aliases,expected", [([], []), (["MDCCC"], ["MDCCC"]),
                                               (["1842 ", "MDCCC"], ["MDCCC"])])
def test_valid_alias_lists_preserve_scoring_and_do_not_mutate_raw(pool, aliases, expected):
    raw = dict(RAW_2HOP, answer_aliases=aliases)
    before = json.dumps(raw)
    candidate = assemble_candidate(raw, "2hop", pool, random.Random(0))
    assert candidate["answer_aliases"] == expected
    assert not answer_matches("C", candidate["answer"], candidate["answer_aliases"])
    assert answer_matches("1842", candidate["answer"], candidate["answer_aliases"])
    if "MDCCC" in expected:
        assert answer_matches("MDCCC", candidate["answer"], candidate["answer_aliases"])
    assert json.dumps(raw) == before


def test_omitted_aliases_still_default_to_empty_list(pool):
    raw = dict(RAW_2HOP)
    del raw["answer_aliases"]
    candidate = assemble_candidate(raw, "2hop", pool, random.Random(0))
    assert candidate["answer_aliases"] == []
    assert "answer_aliases" not in raw


@pytest.mark.parametrize("steps,error_field", [(None, "question_decomposition"),
                                               ([None], "question_decomposition[0]"),
                                               ([{}], "question_decomposition[0].answer")])
def test_malformed_step_containers_produce_field_errors(steps, error_field):
    raw = dict(RAW_2HOP, question_decomposition=steps)
    with pytest.raises(ValueError, match=re.escape(error_field)):
        assemble_candidate(raw, "2hop", ParagraphPool([]), random.Random(0))
