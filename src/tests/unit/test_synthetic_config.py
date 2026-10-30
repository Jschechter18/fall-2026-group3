"""Config parsing is tested with a stub runner; no generation is performed."""
from pathlib import Path

import pytest
import yaml

from mas_sae.data.synthetic import generation_cli


@pytest.fixture
def setup(tmp_path, monkeypatch):
    original = tmp_path / "original.jsonl"
    original.write_text('{"id":"original","question":"Original?","paragraphs":[]}\n')
    captured = []

    def runner(output, **kwargs):
        captured.append((output, kwargs))
        return {"status": "completed"}

    monkeypatch.setattr(generation_cli, "run_batch", runner)
    return original, captured


def write_config(tmp_path, settings):
    path = tmp_path / "generation.yaml"
    path.write_text(yaml.safe_dump(settings))
    return path


def test_full_run_config_preserves_requested_settings(tmp_path, setup):
    original, captured = setup
    config = yaml.safe_load(Path("configs/synthetic/opus5_v3_full.yaml").read_text())
    config.update(original=str(original), dedup_original=[str(original)], output_dir=str(tmp_path / "run"))
    path = write_config(tmp_path, config)
    assert generation_cli.main(["--config", str(path)]) == 0
    output, args = captured[0]
    assert output == tmp_path / "run"
    assert args["n"] is None
    assert args["target_per_type"] == 13334
    assert args["max_requests_per_type"] == 16000
    assert args["concurrency"] == 256
    assert args["seed"] == 42
    assert args["model_id"] == "us.anthropic.claude-opus-5"
    assert args["dedup_originals"] == [original]
    assert args["transport_settings"] == {
        "connect_timeout": 10, "read_timeout": 300,
        "sdk_max_attempts": 3, "retry_mode": "standard",
    }


def test_cli_options_override_config_including_mode(tmp_path, setup):
    original, captured = setup
    path = write_config(tmp_path, dict(original=str(original), target_per_hop=20,
                                     max_requests_per_hop=30, concurrency=64,
                                     output_dir=str(tmp_path / "configured")))
    assert generation_cli.main(["--config", str(path), "--n", "2", "--concurrency", "4",
                                "--output-dir", str(tmp_path / "override")]) == 0
    output, args = captured[0]
    assert output == tmp_path / "override"
    assert args["concurrency"] == 4 and args["n"] == 2
    assert args["target_per_type"] is None and args["max_requests_per_type"] is None


def test_cli_can_switch_fixed_config_to_quota_mode(tmp_path, setup):
    original, captured = setup
    path = write_config(tmp_path, dict(original=str(original), n=5))
    assert generation_cli.main(["--config", str(path), "--target-per-hop", "1",
                                "--max-requests-per-hop", "2"]) == 0
    _, args = captured[0]
    assert args["n"] is None and args["target_per_type"] == 1


@pytest.mark.parametrize("settings", [
    [], {"typo": 1}, {"concurrency": True}, {"concurrency": "256"},
    {"hop_types": "2hop"}, {"hop_types": ["5hop"]}, {"hop_types": ["2hop", "2hop"]},
    {"read_timeout": "slow"}, {"read_timeout": float("inf")},
    {"model_id": None}, {"original": []}, {"dedup_original": []},
    {"n": 5, "target_per_hop": 1}, {"target_per_hop": 2, "max_requests_per_hop": 1},
])
def test_invalid_configs_fail_before_runner(tmp_path, setup, settings):
    _, captured = setup
    path = write_config(tmp_path, settings)
    with pytest.raises(SystemExit) as error:
        generation_cli.main(["--config", str(path)])
    assert error.value.code == 2
    assert not captured


def test_missing_config_fails_before_runner(tmp_path, setup):
    _, captured = setup
    with pytest.raises(SystemExit):
        generation_cli.main(["--config", str(tmp_path / "missing.yaml")])
    assert not captured
