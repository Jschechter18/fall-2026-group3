import importlib.util
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

from mas_sae.sae.hyperparamters import Hyperparameters
from mas_sae.sae.dataloader import create_sae_dataloader
from mas_sae.sae import training


@pytest.mark.parametrize("mode", ["l1", "topk"])
@pytest.mark.parametrize("include_test", [False, True])
def test_training_writes_diagnostics_and_frozen_feature_artifact(
    tmp_path, monkeypatch, mode, include_test
):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    torch.manual_seed(4)
    splits = {"train": 7, "validation": 4}
    if include_test:
        splits["test"] = 3
    for split, n in splits.items():
        torch.save(torch.randn(n, 4), activation_dir / f"{split}.pt")

    hp = Hyperparameters(
        epochs=2, batch_size=3, latent_dim=6, top_k=2,
        sparsity_mode=mode, inactivity_window_examples=8,
    )
    original_hp = asdict(hp)
    monkeypatch.setattr(training, "get_git_provenance", lambda: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return create_sae_dataloader(*args, **kwargs)

    monkeypatch.setattr(training, "create_sae_dataloader", loader)
    result = training.train_sae(
        hp=hp,
        activation_location=activation_dir,
        activation_run_name="fixture",
        layer=33,
        results_root=tmp_path / "results",
        project_root=tmp_path,
        evaluate_test=True,
    )
    run = result.run_directory
    assert asdict(hp) == original_hp
    assert run.parent == tmp_path / "results" / "runs" / "layer_33"
    assert result.checkpoint_path.is_file()
    config = json.loads((run / "config.json").read_text())
    assert config["input_dim"] == 4
    assert config["activation_run_name"] == "fixture"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert set(manifest["activation_files"]) == set(splits)
    history = json.loads((run / "history.json").read_text())
    assert history["history"][0]["train_persistent_inactive_feature_fraction"] is None
    assert history["history"][1]["train_persistent_inactive_feature_fraction"] is not None
    assert "val_variance_explained" in history["history"][0]
    frozen = history["summary"]["frozen_checkpoint_metrics"]
    artifact = json.loads((run / "feature_frequencies.json").read_text())
    for split in splits:
        frequencies = artifact["frequencies"][split]
        assert len(frequencies) == 6
        assert sum(frequencies) == pytest.approx(frozen[split]["mean_active_features"])
        assert "variance_explained" in frozen[split]
        if mode == "topk":
            assert frozen[split]["loss"] == frozen[split]["rec_loss"]
            assert frozen[split]["mean_active_features"] <= 2
    assert result.metrics == frozen
    assert result.best_epoch == history["summary"]["best_epoch"]
    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["epoch"] + 1 == result.best_epoch
    assert checkpoint["val_loss"] == min(row["val_loss"] for row in history["history"])
    assert history["test_loss"] == (frozen["test"]["loss"] if include_test else None)
    assert ("test" in frozen) == include_test
    with (run.parent / "sae_versions.csv").open() as file:
        versions = list(csv.DictReader(file))
    assert len(versions) == 1
    assert tmp_path / versions[0]["checkpoint_path"] == result.checkpoint_path
    assert versions[0]["test_score"] == (
        str(frozen["test"]["loss"]) if include_test else "null"
    )


@pytest.mark.parametrize("evaluate_test", [False, True])
def test_training_cli_delegates_to_shared_trainer(tmp_path, monkeypatch, evaluate_test):
    script = Path(__file__).resolve().parents[3] / "scripts" / "train_sae.py"
    spec = importlib.util.spec_from_file_location("train_sae_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def train(**kwargs):
        calls.append(kwargs)
        return training.TrainingResult(tmp_path, best_epoch=1, metrics={})

    monkeypatch.setattr(module, "train_sae", train)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--run-name", "fixture", "--layer", "33",
        "--sparsity-mode", "topk", "--top-k", "2",
    ] + (["--evaluate-test"] if evaluate_test else []))
    module.main()
    assert len(calls) == 1
    args = calls[0]
    assert args["evaluate_test"] is evaluate_test
    assert args["hp"].sparsity_mode == "topk"
    assert args["hp"].top_k == 2
    assert args["activation_location"] == Path("data/activations/fixture/layer_33")
    assert args["activation_run_name"] == "fixture"
    assert args["layer"] == 33
    assert args["results_root"] == args["project_root"] / "results/sae/musique"
