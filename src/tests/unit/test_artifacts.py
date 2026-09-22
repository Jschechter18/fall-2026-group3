import json
from datetime import datetime, timezone
from types import SimpleNamespace

from mas_sae.experiments import artifacts


def test_create_run_directory_uses_reusable_run_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    git_commit = "abc123def456"
    timestamp = datetime(2026, 9, 16, 15, 30, 45, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{git_commit}\n"),
    )
    monkeypatch.setattr(artifacts, "utc_now", lambda: timestamp)
    monkeypatch.setattr(
        artifacts,
        "get_run_command",
        lambda: "python scripts/train_sae.py",
    )

    run_directory = artifacts.create_sae_run_directory(
        run_name="sae-l64",
        results_root=tmp_path,
        subdirectories=("checkpoints", "plots"),
    )

    manifest = json.loads((run_directory / "manifest.json").read_text())

    assert git_commit not in run_directory.name
    assert run_directory.name == "20260916T153045123456Z_sae-l64"
    assert manifest["git_commit"] == git_commit
    assert manifest["run_id"] == run_directory.name
    assert manifest["run_name"] == "sae-l64"
    assert manifest["created_at"] == timestamp.isoformat()
    assert manifest["command"] == "python scripts/train_sae.py"
    assert manifest["status"] == "running"
    assert manifest["error_message"] == ""
    assert (run_directory / "checkpoints").is_dir()
    assert (run_directory / "plots").is_dir()


def test_get_run_command_quotes_arguments(monkeypatch) -> None:
    monkeypatch.setattr(artifacts.sys, "executable", "/usr/bin/python")
    monkeypatch.setattr(
        artifacts.sys,
        "argv",
        ["scripts/train_sae.py", "--run-name", "test run"],
    )

    command = artifacts.get_run_command()

    assert command == "/usr/bin/python scripts/train_sae.py --run-name 'test run'"


def test_write_run_config(tmp_path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }

    artifacts.write_run_config(run_directory, config)

    saved_config = json.loads((run_directory / "config.json").read_text())
    assert saved_config == config


def test_update_run_manifest_records_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )
    run_directory = artifacts.create_sae_run_directory(
        run_name="sae-l64",
        results_root=tmp_path,
    )

    artifacts.update_run_manifest(run_directory, status="completed")

    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["error_message"] == ""


def test_update_run_manifest_records_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="abc123\n"),
    )
    run_directory = artifacts.create_sae_run_directory(
        run_name="sae-l64",
        results_root=tmp_path,
    )

    artifacts.update_run_manifest(
        run_directory,
        status="failed",
        error_message="RuntimeError: training failed",
    )

    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_message"] == "RuntimeError: training failed"
