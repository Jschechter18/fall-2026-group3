import json
import os
from pathlib import Path

import pytest

import final_labels as fl

ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = ROOT / "results/behavior/behavior_v012_official_20261011/partitions.csv"
RUN = Path(os.environ.get("CANDIDATE_RUN_DIR", ROOT / "results/collection/natural_4b_full9_candidates"))
APPROVED = json.loads((Path(__file__).parent / "counts.json").read_text())


def row(a1, critic, a2, question="What is the answer?", **extra):
    return dict(dict(question_id="synthetic", question=question, solver_attempt_1=a1,
                     critic_advocated_answer=critic, solver_attempt_2=a2, critic_verdict="disagree",
                     critic_noncommittal=False, a2_presentation="candidates", candidate_order="a1_first",
                     ground_truth="", answer_aliases=[]), **extra)


def test_missing_field_is_refused():
    broken = row("Paris", "London", "Paris")
    del broken["critic_verdict"]
    with pytest.raises(ValueError, match="missing fields"):
        fl.label_episode(broken, "train")


def test_unknown_candidate_order_is_refused():
    with pytest.raises(ValueError, match="candidate_order"):
        fl.label_episode(row("Paris", "London", "Candidate A", candidate_order="random"), "train")


# A sentence-length Critic reply makes the pair unusable under v0.2.0, so only the guards decide.
LONG = "The provided paragraphs describe several cities, and the one that matches every clue in the question is London."


@pytest.mark.parametrize("order,side", [("a1_first", "a1"), ("critic_first", "critic")])
def test_bare_letter_is_out_and_resolved_through_saved_order(order, side):
    record = fl.label_episode(row("Paris", LONG, "Candidate A", candidate_order=order), "train")
    assert (record["label"], record["reason"], record["letter_side"]) == ("out", "letter_without_full_candidate_text", side)
    assert "letter_reply" in record["flags"]


def test_citation_copy_is_out():
    record = fl.label_episode(row("[16]", LONG, "[16]"), "train")
    assert (record["label"], record["reason"]) == ("out", "citation_is_not_an_answer")


def test_episode_outside_frozen_partitions_is_out():
    record = fl.label_episode(row("Paris", LONG, "Paris"), None)
    assert (record["label"], record["reason"]) == ("out", "outside_frozen_partitions")


@pytest.mark.parametrize("a2,label", [("Candidate B", "adoption"), ("[16]", "retention")])
def test_usable_under_v020_keeps_its_label(a2, label):
    record = fl.label_episode(row("[16]" if a2 == "[16]" else "Paris", "London", a2), "train")
    assert (record["label"], record["step"]) == (label, "v020_usable")


def test_usable_under_v020_without_a_frozen_partition_stays_out():
    record = fl.label_episode(row("Paris", "London", "London"), None)
    assert (record["label"], record["reason"]) == ("out", "outside_frozen_partitions")


def test_plain_choice_keeps_its_side():
    assert fl.label_episode(row("Paris", "London", "London"), "train")["label"] == "adoption"
    assert fl.label_episode(row("Paris", "London", "Paris"), "train")["label"] == "retention"


def test_critic_mentioning_a1_moves_out_and_keeps_the_previous_label():
    critic = ("The provided text does not mention who the spouse is. The performer of the song is Marvin Gaye, "
              "but no spouse is named.")
    record = fl.label_episode(row("Marvin Gaye", critic, "Marvin Gaye", "Who is the spouse of the performer?"), "train")
    assert (record["label"], record["previous_label"]) == ("out", "retention")
    assert "critic_mentions_a1" in record["flags"]


@pytest.mark.parametrize("a2,label", [("The information is not provided.", "agreement"), ("Honolulu", "out")])
def test_both_refused_is_agreement_only_while_a2_keeps_refusing(a2, label):
    record = fl.label_episode(row("The information is not provided.", "There is insufficient evidence.", a2), "train")
    assert record["label"] == label and "both_refused" in record["flags"]


def test_partitions_file_must_be_the_frozen_one(tmp_path):
    fake = tmp_path / "partitions.csv"
    fake.write_text("question_id,partition\nq1,train\n")
    with pytest.raises(ValueError, match="frozen partitions"):
        fl.read_partitions(fake)


def test_run_must_account_for_every_sampled_question(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    (folder / "sampled_questions.json").write_text(json.dumps([{"question_id": "q1"}, {"question_id": "q2"}]))
    (folder / "interactions.jsonl").write_text(json.dumps(row("Paris", "London", "Paris", question_id="q1")) + "\n")
    (folder / "exclusions.jsonl").write_text("")
    with pytest.raises(ValueError, match="do not account"):
        fl.build(tmp_path, PARTITIONS)


@pytest.fixture(scope="module")
def records():
    if not RUN.exists():
        pytest.skip("candidates run not available")
    return {r["question_id"]: r for r in fl.build(RUN, PARTITIONS)}


def test_build_reproduces_the_approved_labels(records):
    built = list(records.values())
    assert fl.labels_sha256(built) == APPROVED["labels_sha256"]
    assert fl.counts(built)["labels"] == APPROVED["labels"]


def test_every_sampled_question_once_with_its_frozen_partition(records):
    header, *lines = PARTITIONS.read_text().splitlines()
    rows = [dict(zip(header.split(","), line.split(","))) for line in lines]
    frozen = {r["question_id"]: r["partition"] for r in rows}
    assert len(records) == 22355
    assert all(r["partition"] == frozen.get(q) for q, r in records.items())
    assert sum(not r["collected"] for r in records.values()) == 223


def test_reference_cases(records):
    team = {"2hop__325567_158069": "retention", "2hop__147262_63766": "adoption", "2hop__136633_44760": "agreement",
            "2hop__390824_263710": "retention", "2hop__315951_1094": "adoption", "2hop__27065_237509": "adoption",
            "2hop__147094_11125": "out", "3hop1__518624_66618_440465": "out"}
    moved = {"2hop__325567_158069", "2hop__390824_263710"}  # Bordeaux, Marvin Gaye: Critic's reply contains A1's answer
    assert {q: records[q]["label"] for q in team if q not in moved} == {q: t for q, t in team.items() if q not in moved}
    assert all((records[q]["label"], records[q]["previous_label"]) == ("out", "retention") for q in moved)


def test_team_reviewed_labels_are_reference_only(records):
    assert {q for q, r in records.items() if "team_reviewed" in r["flags"]} == set(fl.TEAM_REVIEWED)
    assert {q: records[q]["team_label"] for q in fl.TEAM_REVIEWED} == fl.TEAM_REVIEWED
    differ = {q: records[q]["label"] for q, team in fl.TEAM_REVIEWED.items() if records[q]["label"] != team}
    # the rules never give a different class than the team; they only leave some of these out
    assert set(differ.values()) == {"out"}
    assert len(differ) == 20


def test_usable_under_v020_is_never_out(records):
    usable = [r for r in records.values() if r["v020_eligible_strict"] and r["partition"] is not None]
    assert all(r["label"] != "out" for r in usable)
    # only same-text pairs that differ by hyphen spacing move to agreement, like reference 3
    assert all(r["reason"] == "same text apart from hyphen spacing" for r in usable if r["label"] == "agreement")


def test_letters_and_citations_are_out_unless_usable_under_v020(records):
    flagged = [r for r in records.values() if {"letter_reply", "citation_only"} & set(r["flags"])]
    assert all(r["label"] == "out" or r["step"] == "v020_usable" for r in flagged)


def test_nothing_unpartitioned_is_labeled(records):
    assert all(r["label"] == "out" for r in records.values() if r["partition"] is None)
