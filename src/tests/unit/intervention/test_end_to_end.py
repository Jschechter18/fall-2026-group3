"""choose-r -> prepare -> run (2 shards) -> merge, through the script's stages, on the tiny CPU model with stub labels."""

import importlib.util
import json
from pathlib import Path

import pytest

from . import fakes
from mas_sae.experiments.artifacts import sha256_file
from mas_sae.intervention.run_files import layout
from mas_sae.intervention.experiment.config import parse_config
from mas_sae.intervention.experiment.episodes import eligible_episodes
from mas_sae.intervention.run_files.provenance import IDENTITY_KEYS
from mas_sae.intervention.generation.solver import generate_baseline

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/intervention/run_intervention.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_intervention", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_stored_answers_reproducible(tmp_path, solver_model):
    """Store the tiny model's own baseline answers as the stored A2s, so the baseline check can pass."""
    path = tmp_path / "in/interactions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    episodes = {e.episode_id: e for e in eligible_episodes("in/interactions.jsonl", "in/contexts.jsonl",
                                                           "in/labels", "intervention")}
    for r in rows:
        r.pop("generation_telemetry")
        if r["episode_id"] in episodes:
            r["solver_attempt_2"] = generate_baseline(solver_model, episodes[r["episode_id"]])[0].answer
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    settings = tmp_path / "in/labels/run_settings.json"
    settings.write_text(json.dumps({**json.loads(settings.read_text()), "episodes_sha256": sha256_file(path)}))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = fakes.write_workspace(tmp_path, n_episodes=12)
    raw["experiment"].update(number_of_episodes=2, number_of_choose_r_episodes=1, number_of_shards=2,
                             edit_size=0.5, candidate_edit_sizes=[0.5])
    solver_model = fakes.solver_model()
    _make_stored_answers_reproducible(tmp_path, solver_model)
    script = _load_script()
    monkeypatch.setattr(script, "load_solver", lambda config: solver_model)
    monkeypatch.setattr(script, "provenance", lambda config, device: {k: "same" for k in IDENTITY_KEYS})
    return script, raw


def test_every_stage_runs_and_resumes(experiment, tmp_path):
    script, raw = experiment
    config = parse_config(raw)
    run_dir = Path(config.output_dir)

    script.choose_r(config, labels_stubbed=True)
    assert layout.choose_r_report_md(run_dir).read_text().startswith("# Choose r: edit-size report")

    script.prepare(config)
    assert json.loads(layout.baseline_check(run_dir).read_text())["passed"]
    shards = json.loads(layout.shards(run_dir).read_text())["shards"]
    assert [len(s) for s in shards] == [1, 1]

    for shard in (0, 1):
        script.run(config, shard)
    shard_0 = (run_dir / "shard_0.jsonl").read_text()
    assert len(shard_0.splitlines()) == 1 + 1 + 2 * 3 * (1 + fakes.N_DRAWS)    # baseline, splice, 2 reps × arms
    frozen_draws = json.loads(layout.frozen_run(run_dir).read_text())["random_draws"]
    for line in shard_0.splitlines():                                          # each row used its frozen draws
        row = json.loads(line)
        if row["source"] == "random":
            assert row["random_pool_draw"] == frozen_draws[row["episode_id"]][row["draw"]]
    script.run(config, 0)                                                      # finished: nothing is redone
    assert (run_dir / "shard_0.jsonl").read_text() == shard_0

    script.merge(config, labels_stubbed=True)
    summary = json.loads((run_dir / "merge/summary.json").read_text())
    assert summary["labels_stubbed"] and summary["n_episodes_analysed"] == 2
    assert len(summary["primary"]["table_P"]) == 3                             # 1 primary SAE × 3 arms
    for name in ("table_P_primary", "table_1", "reference_rows", "table_D_directions", "table_3_diagnostics"):
        assert (run_dir / f"merge/{name}.csv").exists() and (run_dir / f"merge/{name}.md").exists()


def test_a_worker_with_a_different_experiment_refuses_to_run(experiment):
    script, raw = experiment
    config = parse_config(raw)
    script.choose_r(config, labels_stubbed=True)
    script.prepare(config)
    changed = parse_config({**raw, "experiment": {**raw["experiment"], "edit_size": 0.25}})
    with pytest.raises(RuntimeError, match="frozen with different"):
        script.run(changed, 0)


def test_a_worker_excludes_an_episode_whose_baseline_differs_and_merge_refuses_too_many(experiment):
    script, raw = experiment
    config = parse_config(raw)
    run_dir = Path(config.output_dir)
    script.choose_r(config, labels_stubbed=True)
    script.prepare(config)
    episode_id = json.loads(layout.shards(run_dir).read_text())["shards"][0][0]
    baseline = layout.baseline_rows(run_dir)
    rows = [json.loads(line) for line in baseline.read_text().splitlines()]
    baseline.write_text("".join(json.dumps({**r, "answer": "not what prepare saw"} if r["episode_id"] == episode_id
                                           else r) + "\n" for r in rows))

    script.run(config, 0)                                                      # does not stop
    assert list(json.loads((run_dir / "shard_0.excluded.json").read_text())) == [episode_id]
    assert (run_dir / "shard_0.jsonl").read_text() == ""                        # none of its rows written
    assert json.loads((run_dir / "shard_0.complete.json").read_text())["excluded"].keys() == {episode_id}
    script.run(config, 0)                                                      # rerun: not retried
    script.run(config, 1)
    with pytest.raises(RuntimeError, match="1 episodes were excluded"):        # 1 of 2 > the 2% allowed
        script.merge(config, labels_stubbed=True)


def test_prepare_stops_when_the_baseline_does_not_reproduce_the_stored_answers(experiment, tmp_path):
    script, raw = experiment
    path = tmp_path / "in/interactions.jsonl"
    rows = [{**json.loads(line), "solver_attempt_2": "something else"} for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    settings = tmp_path / "in/labels/run_settings.json"
    settings.write_text(json.dumps({**json.loads(settings.read_text()), "episodes_sha256": sha256_file(path)}))
    config = parse_config(raw)
    script.choose_r(config, labels_stubbed=True)
    with pytest.raises(RuntimeError, match="does not reproduce the stored answers"):
        script.prepare(config)
    assert not layout.shards(Path(config.output_dir)).exists()
