"""Episode selection, the condition list, and which feature sets each arm steers along."""

import json

import pytest
import torch

import intervention_fakes as fakes
from mas_sae.intervention.experiment.conditions import edited_features, pilot_conditions, sets_for, study_conditions
from mas_sae.intervention.experiment.config import parse_config
from mas_sae.intervention.experiment.episodes import eligible_episodes, seeded_order, study_and_pilot_episodes
from mas_sae.intervention.experiment.representations import frozen_feature_sets


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return fakes.write_workspace(tmp_path, n_episodes=12)


def _eligible(**kw):
    return eligible_episodes("in/interactions.jsonl", "in/contexts.jsonl", "in/labels", "intervention", **kw)


def test_only_adoption_and_retention_episodes_are_eligible(workspace):
    episodes = _eligible()
    assert [e.stored_label for e in episodes] == ["adoption", "retention"] * 3          # 12 labels cycle 4 kinds
    assert [e.question_id for e in episodes] == sorted(e.question_id for e in episodes)
    assert episodes[0].target_label == "retention" and episodes[0].stored_prompt_sha256 == "sha0"


def test_study_and_pilot_are_disjoint_positions_of_one_seeded_order(workspace):
    config = parse_config({**workspace, "experiment": {**workspace["experiment"], "number_of_episodes": 4,
                                                       "number_of_pilot_episodes": 2}})
    study, pilot = study_and_pilot_episodes(config)
    order = seeded_order(_eligible(), 42)
    assert [e.episode_id for e in study + pilot] == [e.episode_id for e in order]
    assert seeded_order(_eligible(), 42) == order and seeded_order(_eligible(), 7) != order
    too_many = parse_config({**workspace, "experiment": {**workspace["experiment"], "number_of_episodes": 6}})
    with pytest.raises(ValueError, match="Only 6 eligible episodes; the study and pilot need 7"):
        study_and_pilot_episodes(too_many)


def test_other_presentations_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fakes.write_workspace(tmp_path, presentation="feedback")
    with pytest.raises(ValueError, match="only candidates is supported"):
        _eligible()


def test_a_context_for_another_question_is_refused(workspace, tmp_path):
    path = tmp_path / "in/contexts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["payload"]["question"] = "Something else?"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="mismatched collected context"):
        _eligible()


def test_condition_counts():
    reps = list(fakes.representations().values())
    conditions = study_conditions(reps, 0.05, fakes.N_DRAWS)
    # 1 baseline + 2 SAE reconstructions + 3 representations × 3 arms × (1 targeted + 2 draws)
    assert len(conditions) == 1 + 2 + 3 * 3 * 3 and len({c.key for c in conditions}) == len(conditions)
    assert len(pilot_conditions(reps[1:], [0.02, 0.05], fakes.N_DRAWS)) == 1 + 2 * 2 * 3 * 3


def test_each_arm_steers_along_the_right_sets():
    rep = fakes.representations()["SAE"]
    edit = lambda arm, source="targeted", draw=None: [c for c in study_conditions([rep], 0.05, fakes.N_DRAWS)
                                                      if (c.arm, c.source, c.draw) == (arm, source, draw)][0]
    toward, away, edited = sets_for(edit("decrease"), rep, "adoption")
    assert (toward.label, away.label, edited) == ("retention", "adoption", "adoption")
    assert edited_features(edit("decrease"), toward, away) == [0, 1, 2]
    toward, away, edited = sets_for(edit("increase"), rep, "retention")
    assert (toward.label, away.label, edited) == ("adoption", "retention", "adoption")
    toward, away, edited = sets_for(edit("both", "random", 1), rep, "adoption")
    assert (toward.features, away.features, edited) == ([15, 7, 9], [9, 10, 11], "both")    # draw k with draw k


def test_raw_targeted_neurons_are_signed_toward_their_label_and_random_ones_are_not():
    raw = fakes.representations()["Raw"]
    assert torch.all(raw.targeted["adoption"].direction[[0, 1, 2]] > 0)
    assert torch.all(raw.targeted["retention"].direction[[3, 4, 5]] > 0)    # coefficient toward adoption < 0
    assert torch.all(raw.random["adoption"][0].direction[[6, 7, 8]] > 0)
    frozen = frozen_feature_sets([raw])["Raw"]
    assert raw.set_cosine() == 0 and frozen["adoption_random"][1] == [9, 10, 11]
