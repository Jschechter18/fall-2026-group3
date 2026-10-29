import copy
import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from mas_sae.sae import training, tuning
from mas_sae.sae.dataloader import create_sae_dataloader
from mas_sae.sae.hyperparamters import Hyperparameters


@pytest.fixture
def setup(tmp_path, monkeypatch):
    location = tmp_path / "data/activations/fixture/layer_33"
    location.mkdir(parents=True)
    generator = torch.Generator().manual_seed(7)
    for split, size in (("train", 7), ("validation", 4), ("test", 3)):
        torch.save(torch.randn(size, 4, generator=generator), location / f"{split}.pt")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(training, "get_git_provenance", lambda: {})
    monkeypatch.setattr(tuning, "get_git_provenance", lambda: {})

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return create_sae_dataloader(*args, **kwargs)

    monkeypatch.setattr(training, "create_sae_dataloader", loader)
    return dict(
        activation_location=location, activation_run_name="fixture", layer=33,
        results_root=tmp_path / "results/sae/musique", project_root=tmp_path,
    )


@pytest.fixture
def config():
    return {
        "activation_run_name": "fixture", "layer": 33,
        "defaults": {"epochs": 1, "batch_size": 3, "latent_dim": 6},
        "groups": [
            {"name": "l1", "fixed": {"sparsity_mode": "l1"},
             "grid": {"sparsity_coefficient": [0.01, 0.1]}},
            {"name": "topk", "fixed": {"sparsity_mode": "topk"},
             "variants": [{"top_k": 1}, {"top_k": 2}]},
        ],
        "selection": {"max_mean_active_features": 6},
    }


def test_expansion_preserves_pairs_and_defaults(config):
    config["groups"] = [{"name": "paired", "fixed": {"sparsity_mode": "topk"},
                         "grid": {"lr": [0.001, 0.002]},
                         "variants": [{"latent_dim": 4, "top_k": 1},
                                      {"latent_dim": 8, "top_k": 2}]}]
    original = copy.deepcopy(config)
    candidates = tuning.expand_search_space(config)
    assert len(candidates) == 4
    assert {(hp.latent_dim, hp.top_k) for _, hp in candidates} == {(4, 1), (8, 2)}
    assert all(hp.epochs == 1 for _, hp in candidates)
    assert config == original


def test_training_skips_existing_test_set_by_default(setup, monkeypatch):
    original_load = torch.load

    def load(path, *args, **kwargs):
        assert Path(path).name != "test.pt", "Default training must not read the test set."
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    result = training.train_sae(
        Hyperparameters(epochs=1, latent_dim=6, batch_size=3), **setup,
    )
    assert "test" not in result.metrics
    history = json.loads((result.run_directory / "history.json").read_text())
    assert history["test_loss"] is None
    manifest = json.loads((result.run_directory / "manifest.json").read_text())
    assert "test" not in manifest["activation_files"]


@pytest.mark.parametrize("group", [
    {"name": "bad", "grid": {"lr": []}},
    {"name": "bad", "fixed": {"lr": 0.1}, "grid": {"lr": [0.2]}},
    {"name": "bad", "fixed": {"unknown": 1}},
    {"name": "bad", "fixed": {"sparsity_mode": "l1", "top_k": 2}},
    {"name": "bad", "fixed": {"sparsity_mode": "topk", "sparsity_coefficient": 0.1}},
    {"name": "bad", "fixed": {"sparsity_mode": "topk", "top_k": 99}},
    {"name": "bad", "grid": {"lr": [0.1, 0.1]}},
    {"name": "bad", "variants": []},
    {"name": "bad", "fixed": {"batch_size": True}},
])
def test_invalid_search_rejected_before_training(config, group, tmp_path):
    config["groups"] = [group]
    with pytest.raises(ValueError):
        tuning.run_tuning(config, tmp_path)
    assert not (tmp_path / "results").exists()


def test_selection_uses_common_metrics_not_total_loss():
    trials = [dict(status="completed", validation=dict(rec_loss=rec,
              mean_active_features=activity, loss=loss))
              for rec, activity, loss in [(0.1, 8, 0.1), (0.2, 2, 9), (0.3, 3, 0.3)]]
    winner = tuning.compare_trials(trials, {"max_mean_active_features": 3})
    assert winner is trials[1]
    assert [trial["pareto"] for trial in trials] == [True, True, False]
    assert tuning.compare_trials(trials, {"max_mean_active_features": 0}) is None
    assert tuning.compare_trials(trials, {}) is None


@pytest.mark.parametrize("settings", [
    {"aux_coefficient": -1}, {"aux_coefficient": float("nan")},
    {"aux_coefficient": float("inf")}, {"aux_coefficient": True},
    {"aux_k": 0}, {"aux_k": True}, {"aux_dead_window_examples": 0},
    {"aux_dead_window_examples": 1.5},
    {"aux_coefficient": 0.25, "aux_k": 7},
    {"sparsity_mode": "l1", "aux_coefficient": 0.25, "aux_k": 2},
])
def test_invalid_auxk_search_rejected_before_outputs(config, tmp_path, settings):
    config["groups"] = [{"name": "auxk", "fixed": {
        "sparsity_mode": "topk", **settings,
    }}]
    config["defaults"]["top_k"] = 2
    with pytest.raises(ValueError):
        tuning.run_tuning(config, tmp_path)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("mode, aux_coefficient", [("l1", 0.0), ("topk", 0.0), ("topk", 0.25)])
def test_continuation_matches_uninterrupted_training(setup, mode, aux_coefficient):
    hp = Hyperparameters(epochs=3, latent_dim=6, batch_size=3, top_k=2,
                         sparsity_mode=mode, inactivity_window_examples=8,
                         aux_coefficient=aux_coefficient, aux_k=2, aux_dead_window_examples=2,
                         early_stopping_min_delta=0.0001)
    full = training.train_sae(hp, **setup, evaluate_test=True)
    trial = training.train_sae(replace(hp, epochs=1), **setup,
                               evaluate_test=False, register_version=False)
    assert "test" not in trial.metrics
    resumed = training.finish_run(trial.run_directory, 3, setup["project_root"])
    assert resumed.run_directory == trial.run_directory
    assert resumed.metrics == full.metrics
    full_history = json.loads((full.run_directory / "history.json").read_text())
    resumed_history = json.loads((resumed.run_directory / "history.json").read_text())
    assert full_history == resumed_history
    if aux_coefficient:
        assert any(row["train_aux_loss"] > 0 for row in full_history["history"])
        assert all(row["val_loss"] == row["val_rec_loss"] for row in full_history["history"])
    full_state = torch.load(full.checkpoint_path, weights_only=True)["model_state_dict"]
    resumed_state = torch.load(resumed.checkpoint_path, weights_only=True)["model_state_dict"]
    assert all(torch.equal(value, resumed_state[key]) for key, value in full_state.items())
    with pytest.raises(ValueError, match="unfinalized"):
        training.finish_run(trial.run_directory, 4, setup["project_root"])


@pytest.mark.parametrize("remove_tied", [False, True])
def test_finish_legacy_config_without_auxk_fields(setup, remove_tied):
    hp = Hyperparameters(epochs=1, latent_dim=6, batch_size=3, top_k=2,
                         sparsity_mode="topk", tied_initialization=not remove_tied)
    trial = training.train_sae(hp, **setup, evaluate_test=False, register_version=False)
    path = trial.run_directory / "config.json"
    config = json.loads(path.read_text())
    for key in ("aux_coefficient", "aux_k", "aux_dead_window_examples"):
        del config[key]
    del config["early_stopping_min_delta"]
    if remove_tied:
        del config["tied_initialization"]
    path.write_text(json.dumps(config))
    result = training.finish_run(trial.run_directory, 2, setup["project_root"])
    saved = json.loads((result.run_directory / "config.json").read_text())
    assert saved["aux_coefficient"] == 0.0
    assert saved["early_stopping_min_delta"] == 0.0
    assert saved["tied_initialization"] == (not remove_tied)
    history = json.loads((result.run_directory / "history.json").read_text())
    assert len(history["history"]) == 2
    assert all(row["train_aux_loss"] == 0 for row in history["history"])


@pytest.mark.parametrize("change", [
    {"aux_coefficient": 0.5}, {"aux_k": 3}, {"aux_dead_window_examples": 4},
])
def test_continuation_rejects_auxk_setting_changes(setup, change):
    hp = Hyperparameters(epochs=1, latent_dim=6, batch_size=3, top_k=2,
                         sparsity_mode="topk", aux_coefficient=0.25, aux_k=2,
                         aux_dead_window_examples=2)
    trial = training.train_sae(hp, **setup, evaluate_test=False, register_version=False)
    with pytest.raises(ValueError, match="Cannot change aux_"):
        training.train_sae(replace(hp, epochs=2, **change), **setup,
                           evaluate_test=False, register_version=False,
                           resume_directory=trial.run_directory)


def test_auxk_study_records_settings_and_training_objective(setup, config, capsys):
    config["groups"] = [{"name": "auxk", "fixed": {
        "sparsity_mode": "topk", "top_k": 1, "aux_k": 2,
        "aux_dead_window_examples": 2,
    }, "grid": {"aux_coefficient": [0.0, 0.25]}}]
    study = tuning.run_tuning(config, setup["project_root"])
    with (study / "trials.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert {float(row["aux_coefficient"]) for row in rows} == {0.0, 0.25}
    for path in (study / "trials").glob("*/config.json"):
        saved = json.loads(path.read_text())
        assert saved["aux_k"] == 2
        assert saved["aux_dead_window_examples"] == 2
        assert ("aux_coefficient * aux_loss" in saved["total_loss"]) == (saved["aux_coefficient"] > 0)
        history = json.loads((path.parent / "history.json").read_text())
        row = history["history"][0]
        assert ("train_auxiliary" in row) == (saved["aux_coefficient"] > 0)
        if saved["aux_coefficient"] > 0:
            assert row["train_auxiliary"]["measured_examples"] > 0
        assert "val_auxiliary" not in row
    assert "Auxiliary: eligible=" in capsys.readouterr().out


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.0001"])
def test_invalid_stopping_threshold_rejected_before_outputs(config, tmp_path, value):
    config["defaults"]["early_stopping_min_delta"] = value
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        tuning.run_tuning(config, tmp_path)
    assert not (tmp_path / "results").exists()


def test_training_uses_stopping_threshold_and_saves_lowest_loss(setup, monkeypatch):
    original = training.ModelRunner.val_epoch
    losses = iter([0.5, 0.49997, 0.49994])

    def controlled_validation(runner, loader):
        metrics = original(runner, loader)
        loss = next(losses, None)
        if loss is not None:
            metrics["loss"] = metrics["rec_loss"] = loss
        return metrics

    monkeypatch.setattr(training.ModelRunner, "val_epoch", controlled_validation)
    hp = Hyperparameters(epochs=10, patience=2, latent_dim=6, top_k=2,
                         sparsity_mode="topk", early_stopping_min_delta=0.0001)
    result = training.train_sae(hp, **setup, evaluate_test=False, register_version=False)
    history = json.loads((result.run_directory / "history.json").read_text())
    assert history["summary"]["stop_reason"] == "early_stopping"
    assert len(history["history"]) == 3
    assert result.best_epoch == 3
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["val_loss"] == pytest.approx(0.49994)
    saved = json.loads((result.run_directory / "config.json").read_text())
    assert saved["early_stopping_min_delta"] == 0.0001
    with pytest.raises(ValueError, match="Cannot change early_stopping_min_delta"):
        training.train_sae(replace(hp, early_stopping_min_delta=0), **setup,
                           evaluate_test=False, register_version=False,
                           resume_directory=result.run_directory)


@pytest.mark.parametrize("study_trial", [False, True])
def test_finishing_after_project_relocation(setup, config, study_trial):
    project_root = setup["project_root"]
    if study_trial:
        study = tuning.run_tuning(config, project_root)
        selection = json.loads((study / "selection.json").read_text())
        run = project_root / selection["run_directory"]
    else:
        trial = training.train_sae(
            Hyperparameters(epochs=1, latent_dim=6, batch_size=3), **setup,
            evaluate_test=False, register_version=False,
        )
        run = trial.run_directory
    relative_run = run.relative_to(project_root)
    original_manifest = json.loads((run / "manifest.json").read_text())
    relocated_root = project_root / "relocated"
    relocated_root.mkdir()
    for directory in ("data", "results"):
        (project_root / directory).rename(relocated_root / directory)
    assert not Path(original_manifest["activation_files"]["train"]["path"]).exists()

    result = training.finish_run(relocated_root / relative_run, 2, relocated_root)

    assert result.run_directory.is_relative_to(relocated_root)
    assert result.checkpoint_path.is_file()
    assert "test" in result.metrics
    manifest = json.loads((result.run_directory / "manifest.json").read_text())
    assert manifest["finalized"] is True


def test_study_artifacts_and_finishing_selected_run(setup, config):
    # Corrupt test data proves neither loading nor hashing it is needed for tuning.
    test_path = setup["activation_location"] / "test.pt"
    test_path.write_bytes(b"not a tensor")
    study = tuning.run_tuning(config, setup["project_root"])
    assert study.parent == setup["results_root"] / "tuning/layer_33"
    assert "T" in study.name and "Z-" in study.name
    trials_text = (study / "trials.json").read_text()
    trials = json.loads(trials_text)["trials"]
    assert len(trials) == 4
    assert {trial["group"] for trial in trials} == {"l1", "topk"}
    assert not (setup["results_root"] / "runs").exists()
    assert len(list((study / "trials").iterdir())) == 4
    with (study / "trials.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert "val_rec_loss" in rows[0] and "activation_run_name" in rows[0]
    selection = json.loads((study / "selection.json").read_text())
    run = setup["project_root"] / selection["run_directory"]
    original_files = {str(path.relative_to(run)): path.read_bytes()
                      for path in run.rglob("*") if path.is_file()}
    torch.save(torch.randn(3, 4), test_path)
    result = training.finish_run(run, 2, setup["project_root"])
    assert result.run_directory.parent == setup["results_root"] / "runs/layer_33"
    assert result.checkpoint_path != setup["project_root"] / selection["checkpoint_path"]
    assert result.checkpoint_path.is_file()
    assert original_files == {str(path.relative_to(run)): path.read_bytes()
                              for path in run.rglob("*") if path.is_file()}
    assert len(list(result.run_directory.parent.iterdir())) == 2  # Run plus registry.
    assert "test" in result.metrics
    assert (study / "trials.json").read_text() == trials_text
    finalization = json.loads((study / "finalizations" / f"{run.name}.json").read_text())
    assert finalization["metrics"] == result.metrics
    assert finalization["trial_id"] == selection["trial_id"]
    manifest = json.loads((result.run_directory / "manifest.json").read_text())
    assert manifest["finalized"] is True
    assert setup["project_root"] / manifest["source_trial_directory"] == run
    with pytest.raises(ValueError, match="already been published"):
        training.finish_run(run, 3, setup["project_root"])
    with (result.run_directory.parent / "sae_versions.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 1


def test_continuation_rejects_changed_data(setup):
    trial = training.train_sae(Hyperparameters(epochs=1, latent_dim=6), **setup,
                               evaluate_test=False, register_version=False)
    torch.save(torch.randn(7, 4), setup["activation_location"] / "train.pt")
    with pytest.raises(ValueError, match="activations changed"):
        training.finish_run(trial.run_directory, 2, setup["project_root"])


def test_failure_is_recorded(setup, config, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("trial failed")
    monkeypatch.setattr(tuning, "train_sae", fail)
    with pytest.raises(RuntimeError, match="trial failed"):
        tuning.run_tuning(config, setup["project_root"])
    study = next((setup["results_root"] / "tuning/layer_33").iterdir())
    assert json.loads((study / "manifest.json").read_text())["status"] == "failed"
    trials = json.loads((study / "trials.json").read_text())["trials"]
    assert trials[0]["status"] == "failed"
    assert trials[1]["status"] == "pending"


@pytest.mark.parametrize("args", [
    ["--hp-tune"], ["--config", "config.yaml"], ["--dry-run"],
    ["--finish-run", "run"], ["--epochs", "3"],
    ["--hp-tune", "--config", "config.yaml", "--layer", "33"],
    ["--hp-tune", "--config", "config.yaml", "--finish-run", "run", "--epochs", "3"],
    ["--hp-tune", "--config", "config.yaml", "--evaluate-test"],
    ["--finish-run", "run", "--epochs", "3", "--evaluate-test"],
])
def test_cli_rejects_incompatible_modes(args, monkeypatch):
    script = Path(__file__).resolve().parents[3] / "scripts/train_sae.py"
    spec = importlib.util.spec_from_file_location("train_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), *args])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_finishing_respects_early_stopping(setup):
    hp = Hyperparameters(epochs=4, latent_dim=6, batch_size=3, lr=1e-30, patience=1)
    trial = training.train_sae(hp, **setup, evaluate_test=False, register_version=False)
    before = json.loads((trial.run_directory / "history.json").read_text())
    assert before["summary"]["stop_reason"] == "early_stopping"
    result = training.finish_run(trial.run_directory, 10, setup["project_root"])
    after = json.loads((result.run_directory / "history.json").read_text())
    assert after["history"] == before["history"]
    assert "test" in result.metrics


def test_cli_loads_yaml_and_delegates_tuning(config, tmp_path, monkeypatch):
    import yaml
    script = Path(__file__).resolve().parents[3] / "scripts/train_sae.py"
    spec = importlib.util.spec_from_file_location("tune_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config))
    calls = []

    def run(config, project_root):
        calls.append(config)
        return tmp_path

    monkeypatch.setattr(module, "run_tuning", run)
    monkeypatch.setattr(sys, "argv", [str(script), "--hp-tune", "--config", str(config_path)])
    module.main()
    assert calls == [config]
    calls.clear()
    monkeypatch.setattr(sys, "argv", [str(script), "--hp-tune", "--config", str(config_path), "--dry-run"])
    module.main()
    assert calls == []


def test_failed_finishing_stays_inside_study(setup, config):
    config["groups"] = [config["groups"][0]]
    config["groups"][0]["grid"] = {"sparsity_coefficient": [0.1]}
    study = tuning.run_tuning(config, setup["project_root"])
    trial = next((study / "trials").iterdir())
    (setup["activation_location"] / "test.pt").write_bytes(b"invalid tensor")
    with pytest.raises(Exception):
        training.finish_run(trial, 2, setup["project_root"])
    assert not (setup["results_root"] / "runs").exists()
    assert (study / "finishing" / trial.name).is_dir()
    torch.save(torch.randn(3, 4), setup["activation_location"] / "test.pt")
    result = training.finish_run(trial, 2, setup["project_root"])
    assert result.checkpoint_path.is_file()
    assert not (study / "finishing" / trial.name).exists()


def test_publication_retry_does_not_retrain(setup, config, monkeypatch):
    config["groups"] = [config["groups"][0]]
    config["groups"][0]["grid"] = {"sparsity_coefficient": [0.1]}
    study = tuning.run_tuning(config, setup["project_root"])
    trial = next((study / "trials").iterdir())
    register = training.append_sae_version

    def fail(*args, **kwargs):
        raise OSError("registry unavailable")

    monkeypatch.setattr(training, "append_sae_version", fail)
    with pytest.raises(OSError, match="registry unavailable"):
        training.finish_run(trial, 1, setup["project_root"])
    monkeypatch.setattr(training, "append_sae_version", register)
    monkeypatch.setattr(training, "train_sae", lambda **kwargs: pytest.fail("must not retrain"))
    result = training.finish_run(trial, 1, setup["project_root"])
    assert result.checkpoint_path.is_file()
    records = list((study / "finalizations").glob("*.json"))
    assert json.loads(records[0].read_text())["status"] == "completed"


@pytest.mark.parametrize("stopped,total_epochs", [(False, 1), (True, 3)])
def test_evaluation_only_finalization_skips_device_rng(setup, monkeypatch, stopped, total_epochs):
    trial = training.train_sae(
        Hyperparameters(epochs=1, latent_dim=6), **setup,
        evaluate_test=False, register_version=False,
    )
    path = trial.run_directory / "checkpoints/last_checkpoint.pt"
    state = torch.load(path, weights_only=True)
    # Simulate a checkpoint from another device without requiring GPU hardware.
    state["device_type"] = "cuda"
    state["stopped"] = stopped
    torch.save(state, path)
    monkeypatch.setattr(training, "restore_training_state",
                        lambda *args: pytest.fail("Must not restore training RNG for evaluation"))
    result = training.finish_run(trial.run_directory, total_epochs, setup["project_root"])
    history = json.loads((result.run_directory / "history.json").read_text())
    assert len(history["history"]) == 1
    assert "test" in result.metrics


def test_additional_training_still_requires_same_device(setup):
    trial = training.train_sae(
        Hyperparameters(epochs=1, latent_dim=6), **setup,
        evaluate_test=False, register_version=False,
    )
    path = trial.run_directory / "checkpoints/last_checkpoint.pt"
    state = torch.load(path, weights_only=True)
    state["device_type"] = "cuda"
    torch.save(state, path)
    with pytest.raises(ValueError, match="same device type"):
        training.finish_run(trial.run_directory, 2, setup["project_root"])
