"""Each preflight failure, the config checks, and every problem reported in one error."""

import json

import pytest

from . import fakes
from mas_sae.intervention.checks import preflight
from mas_sae.intervention.experiment.config import parse_config


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A complete set of inputs in the cwd, and its config mapping."""
    monkeypatch.chdir(tmp_path)
    return fakes.write_workspace(tmp_path)


def _problems(raw, check):
    return check(parse_config(raw))


def test_the_workspace_passes_every_check_that_needs_no_gpu_or_aws(workspace):
    config = parse_config(workspace)
    assert preflight.input_file_problems(config) == []
    assert preflight.representation_problems(config) == []
    assert preflight.output_problems(config) == []
    assert preflight.labeler_problems(config) == []


def test_config_problems_are_reported_together(workspace):
    raw = {**workspace, "experiment": {**workspace["experiment"], "edit_size": 2, "number_of_random_draws": 0},
           "solver": {**workspace["solver"], "generation": {"do_sample": True}}}
    with pytest.raises(ValueError) as error:
        parse_config(raw)
    for text in ("edit_size must be a number in (0, 1)", "number_of_random_draws must be a positive integer",
                 "do_sample must be false"):
        assert text in str(error.value)


def test_config_refuses_more_draws_per_question_than_the_pool_holds(workspace):
    raw = {**workspace, "experiment": {**workspace["experiment"], "number_of_random_draws": 5,
                                       "random_draw_pool_size": 4}}
    with pytest.raises(ValueError, match="number_of_random_draws must be at most experiment.random_draw_pool_size"):
        parse_config(raw)


def test_config_refuses_unknown_and_missing_keys(workspace):
    with pytest.raises(ValueError, match="Unknown keys in experiment"):
        parse_config({**workspace, "experiment": {**workspace["experiment"], "typo": 1}})
    with pytest.raises(ValueError, match="missing required fields: \\['output_root'\\]"):
        parse_config({k: v for k, v in workspace.items() if k != "output_root"})
    with pytest.raises(ValueError, match="missing required fields: \\['edit_size'\\]"):
        parse_config({**workspace, "experiment": {k: v for k, v in workspace["experiment"].items() if k != "edit_size"}})


def test_config_requires_the_primary_family_to_be_the_primary_representations(workspace):
    raw = {**workspace, "analysis": {**workspace["analysis"], "primary_representations": ["Raw"]}}
    with pytest.raises(ValueError, match="primary_representations must be exactly"):
        parse_config(raw)


def test_missing_input_files(workspace, tmp_path):
    (tmp_path / "in/contexts.jsonl").unlink()
    (tmp_path / "in/labels/summary.json").unlink()
    problems = _problems(workspace, preflight.input_file_problems)
    assert problems == ["missing input file in/contexts.jsonl", "missing input file in/labels/summary.json"]


def test_stored_labels_of_other_interactions(workspace, tmp_path):
    with (tmp_path / "in/interactions.jsonl").open("a") as f:
        f.write("\n")
    assert "labeled a different interactions file" in _problems(workspace, preflight.input_file_problems)[0]


def test_a_changed_or_missing_checkpoint(workspace, tmp_path):
    (tmp_path / "in/center.npy").write_bytes(b"changed")
    (tmp_path / "in/sae/params.safetensors").unlink()
    problems = _problems(workspace, preflight.representation_problems)
    assert problems == ["Raw: checkpoint files differ from the ones feature selection used",
                        "SAE: missing checkpoint files ['in/sae/params.safetensors']"]


def test_a_representation_path_other_than_feature_selections(workspace):
    raw = {**workspace, "representations": {**workspace["representations"],
                                            "SAE": {"kind": "gemma_scope", "path": "elsewhere", "role": "primary"}}}
    assert "is not the one feature selection used" in _problems(raw, preflight.representation_problems)[0]


def test_invalid_feature_sets(workspace, tmp_path):
    path = tmp_path / "in/feature_sets.json"
    sets = json.loads(path.read_text())
    sets["Raw"]["retention_features"][0]["feature"] = 0              # overlaps adoption
    sets["Raw"]["random_draws_adoption"][1] = [99, 98, 97]            # outside the width
    sets["SAE"]["random_draws_retention"][0] = [3, 30, 31]            # contains a targeted feature
    sets["SAE"]["random_draws_adoption"] = sets["SAE"]["random_draws_adoption"][:1]   # too few draws
    path.write_text(json.dumps(sets))
    problems = _problems(workspace, preflight.representation_problems)
    assert "Raw: the adoption and retention sets overlap" in problems
    assert "Raw: random adoption draw 1 must be 3 unique ints in [0, 16)" in problems
    assert "SAE: random retention draw 0 shares a feature with a targeted set" in problems
    assert "SAE: 1 random adoption draws, 4 needed" in problems                          # the pool size
    assert "SAE: missing" not in " ".join(problems)


def test_a_changed_labeling_prompt(workspace, tmp_path):
    path = tmp_path / "in/labels/run_settings.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "prompt_sha256": "0" * 64}))
    assert "labeling prompt's sha256 differs" in _problems(workspace, preflight.labeler_problems)[0]


def test_uncached_solver_weights(workspace, tmp_path):
    assert "are not cached" in _problems(workspace, preflight.solver_weight_problems)[0]
    snapshot = tmp_path / "cache/models--org--model/snapshots/abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("")
    assert _problems(workspace, preflight.solver_weight_problems) == []


def test_no_cuda(workspace, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _problems(workspace, preflight.machine_problems) == ["CUDA is not available"]


def test_unwritable_output_and_too_little_disk(workspace, tmp_path):
    (tmp_path / "out").write_text("a file, not a directory")
    assert "is not writable" in _problems(workspace, preflight.output_problems)[0]
    raw = {**workspace, "output_root": "out2", "machine": {"min_free_disk_gb": 1e9}}
    assert "GB free in out2" in _problems(raw, preflight.output_problems)[0]


def test_only_choose_r_may_start_a_run_and_every_stage_needs_an_existing_run_folder(workspace, tmp_path):
    new = parse_config(workspace)                                              # run_id: null
    assert preflight.run_problems(new, "choose-r") == []
    for stage in ("prepare", "run", "merge"):
        assert "run_id is not set" in preflight.run_problems(new, stage)[0]
    named = parse_config({**workspace, "run_id": "20261106T000000000000Z-abc"})
    for stage in ("choose-r", "prepare", "run", "merge"):                      # a typo'd or deleted run
        assert "no run folder out/20261106T000000000000Z-abc" in preflight.run_problems(named, stage)[0]
    (tmp_path / "out/20261106T000000000000Z-abc").mkdir(parents=True)
    assert preflight.run_problems(named, "prepare") == []


def test_config_refuses_the_old_output_dir_and_a_run_id_that_is_a_path(workspace):
    with pytest.raises(ValueError, match="output_dir was replaced by output_root"):
        parse_config({**workspace, "output_dir": "out"})
    with pytest.raises(ValueError, match="run_id must be the name of one folder"):
        parse_config({**workspace, "run_id": "../elsewhere"})


def test_aws_credentials_that_do_not_work(workspace, monkeypatch):
    import boto3

    class Broken:
        def __init__(self, **kwargs):
            pass

        def client(self, *args, **kwargs):
            raise RuntimeError("expired token")

    monkeypatch.setattr(boto3, "Session", Broken)
    problem = _problems(workspace, preflight.aws_credential_problems)[0]
    assert "do not work" in problem and "aws sso login --profile school" in problem


def test_stage_requirements(workspace, tmp_path):
    workspace = {**workspace, "run_id": "run"}
    (tmp_path / "out/run").mkdir(parents=True)
    raw = {**workspace, "experiment": {**workspace["experiment"], "edit_size": None}}
    prepare = preflight.stage_problems(parse_config(raw), "prepare", None)
    assert any("edit_size is not set" in p for p in prepare) and any("no choose-r report" in p for p in prepare)
    run = preflight.stage_problems(parse_config(workspace), "run", 0)
    assert len(run) == 2 and all("run prepare first" in p for p in run)


def test_one_error_lists_every_problem(workspace, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (tmp_path / "in/contexts.jsonl").unlink()
    with pytest.raises(preflight.PreflightError) as error:
        preflight.run_preflight(parse_config(workspace), "choose-r", labels_stubbed=True)
    message = str(error.value)
    assert "(3 problems)" in message
    for text in ("missing input file in/contexts.jsonl", "are not cached", "CUDA is not available"):
        assert text in message
