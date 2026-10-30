import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mas_sae.data.synthetic import DuplicateIndex, normalize_question, validate_candidate
from mas_sae.data.synthetic.cli import main


@pytest.fixture
def candidate():
    path = Path(__file__).parents[1] / "fixtures/synthetic_musique/valid.json"
    return json.loads(path.read_text())


def codes(candidate):
    return {issue.code for issue in validate_candidate(candidate).errors}


def test_valid_fixture_is_not_mutated(candidate):
    before = json.dumps(candidate)
    assert validate_candidate(candidate).valid
    assert json.dumps(candidate) == before


@pytest.mark.parametrize("value", [None, [], "text", 42])
def test_non_object(value):
    assert codes(value) == {"object"}


@pytest.mark.parametrize("field", ["id", "question", "answer", "answer_aliases", "answerable",
                                   "paragraphs", "question_decomposition"])
def test_missing_fields(candidate, field):
    del candidate[field]
    assert not validate_candidate(candidate).valid


def test_invalid_indices_and_boolean_integer(candidate):
    candidate["paragraphs"][1]["idx"] = True
    candidate["question_decomposition"][0]["paragraph_support_idx"] = False
    assert {"paragraph_index", "support_reference"} <= codes(candidate)


def test_decomposition_references(candidate):
    candidate["question_decomposition"][1]["question"] = "When was #2 founded?"
    assert {"step_reference", "unused_step"} <= codes(candidate)
    candidate["question_decomposition"][1]["paragraph_support_idx"] = 19
    assert {"support_reference", "unused_support"} <= codes(candidate)


def test_answer_and_support_consistency(candidate):
    candidate["answer"] = "different"
    candidate["paragraphs"][2]["is_supporting"] = True
    assert {"final_answer", "support_count", "unused_support"} <= codes(candidate)


def test_malformed_nested_values(candidate):
    candidate["paragraphs"][0] = []
    candidate["question_decomposition"][0] = None
    assert "object" in codes(candidate)


def test_normalization():
    assert normalize_question("  ＷＨＥＲＥ\t is\nStraße? ") == "where is strasse?"
    assert normalize_question("Where?") != normalize_question("Where")
    with pytest.raises(ValueError):
        normalize_question(" \n")


def test_duplicates_persist_and_report_source(tmp_path):
    path = tmp_path / "index.sqlite"
    with DuplicateIndex(path) as index:
        assert index.register_question("original-1", "Where is Selin?", source="train") is None
    with DuplicateIndex(path) as index:
        match = index.register_question("new-1", " WHERE  is Selin? ")
        assert match.record_id == "original-1"
        assert match.source == "train"
        assert index.find_duplicate("New question?") is None


def test_concurrent_registration_has_one_winner(tmp_path):
    path = tmp_path / "index.sqlite"
    with DuplicateIndex(path):
        pass

    def register(number):
        with DuplicateIndex(path) as index:
            return index.register_question(str(number), "Same question?")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(register, range(8)))
    assert sum(result is None for result in results) == 1
    assert len({result.record_id for result in results if result}) == 1


def test_import_is_atomic_and_idempotent(tmp_path):
    source = tmp_path / "original.jsonl"
    source.write_text('{"id":"o1","question":"Original?"}\nnot json\n')
    with DuplicateIndex(tmp_path / "index.sqlite") as index:
        with pytest.raises(ValueError, match=":2:"):
            index.import_jsonl(source, source="train")
        assert index.find_duplicate("Original?") is None
        source.write_text('{"id":"o1","question":"Original?"}\n')
        assert index.import_jsonl(source, source="train") == 1
        assert index.import_jsonl(source, source="train") == 0


def test_cli_filters_original_and_batch_duplicates(tmp_path, candidate, capsys):
    source = tmp_path / "original.jsonl"
    source.write_text('{"id":"o1","question":"Original?"}\n')
    incoming = tmp_path / "candidates.jsonl"
    original_duplicate = dict(candidate, id="2hop__old", question=" ORIGINAL? ")
    incoming.write_text('\n'.join([json.dumps(original_duplicate), json.dumps(candidate),
                                    json.dumps(candidate), '{bad json'])+'\n')
    output = tmp_path / "run"
    args = [str(incoming), "--original", str(source), "--index", str(tmp_path / "index.sqlite"),
            "--output-dir", str(output)]
    assert main(args) == 0
    assert len((output / "accepted.jsonl").read_text().splitlines()) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["accepted"] == 1
    assert summary["rejected"] == 3
    assert summary["rejections_by_code"] == {"duplicate": 2, "json": 1}
    with pytest.raises(SystemExit):
        main(args)


def test_paragraph_count_profile(candidate):
    candidate["paragraphs"] = candidate["paragraphs"][:19]
    assert validate_candidate(candidate).valid
    assert not validate_candidate(candidate, expected_paragraphs=20).valid
    with pytest.raises(ValueError):
        validate_candidate(candidate, expected_paragraphs=0)


def test_branched_decomposition(candidate):
    candidate["id"] = "3hop2__fixture"
    candidate["paragraphs"][2]["is_supporting"] = True
    candidate["question_decomposition"][1]["question"] = "Which region contains Selin?"
    candidate["question_decomposition"][1]["answer"] = "Region A"
    candidate["question_decomposition"].append({
        "id": 3, "question": "When was #1 in #2 founded?", "answer": "1842", "paragraph_support_idx": 2
    })
    assert validate_candidate(candidate).valid


@pytest.mark.parametrize("source", ["", "   ", None])
def test_import_and_registration_reject_invalid_source(tmp_path, source):
    original = tmp_path / "original.jsonl"
    original.write_text('{"id":"o1","question":"Original?"}\n')
    with DuplicateIndex(tmp_path / "index.sqlite") as index:
        with pytest.raises(ValueError):
            index.register_question("o1", "Original?", source=source)
        with pytest.raises(ValueError, match=":1:"):
            index.import_jsonl(original, source=source)
        assert index.find_duplicate("Original?") is None


@pytest.mark.parametrize("hop_type,questions", [
    ("2hop", ["First?", "Uses #1?"]),
    ("3hop1", ["First?", "Uses #1?", "Uses #2?"]),
    ("3hop2", ["First?", "Second?", "Uses #1 and #2?"]),
    ("4hop1", ["First?", "Uses #1?", "Uses #2?", "Uses #3?"]),
    ("4hop2", ["First?", "Second?", "Uses #1 and #2?", "Uses #3?"]),
    ("4hop3", ["First?", "Uses #1?", "Third?", "Uses #2 and #3?"]),
])
def test_hop_topology_matches_subtype(candidate, hop_type, questions):
    candidate["id"] = f"{hop_type}__fixture"
    for i, paragraph in enumerate(candidate["paragraphs"]):
        paragraph["is_supporting"] = i < len(questions)
    candidate["question_decomposition"] = [
        {"id": i + 1, "question": question, "answer": candidate["answer"],
         "paragraph_support_idx": i}
        for i, question in enumerate(questions)
    ]
    assert validate_candidate(candidate).valid
    candidate["question_decomposition"][-1]["question"] = "Missing dependencies?"
    assert "hop_shape" in codes(candidate)


def test_cli_rejects_chain_labeled_as_branch(tmp_path, candidate):
    candidate["id"] = "3hop2__chain"
    candidate["paragraphs"][2]["is_supporting"] = True
    candidate["question_decomposition"].append({
        "id": 3, "question": "What follows #2?", "answer": candidate["answer"],
        "paragraph_support_idx": 2,
    })
    source = tmp_path / "original.jsonl"
    source.write_text('{"id":"o1","question":"Original?"}\n')
    incoming = tmp_path / "candidates.jsonl"
    incoming.write_text(json.dumps(candidate) + "\n")
    output = tmp_path / "validation"
    assert main([str(incoming), "--original", str(source), "--index", str(tmp_path / "index.sqlite"),
                 "--output-dir", str(output)]) == 0
    assert (output / "accepted.jsonl").read_text() == ""
    rejection = json.loads((output / "rejected.jsonl").read_text())
    assert "hop_shape" in {error["code"] for error in rejection["errors"]}
