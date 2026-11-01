"""Synthetic-pilot paper figure (#174, #179): counts per cohort, collection failures as Out, refusals."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/evaluation/plot_synthetic_pilot.py"


@pytest.fixture
def plot():
    spec = importlib.util.spec_from_file_location("plot_synthetic_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


LABELED = [{"question_id": "q1", "behavior_label": "adoption"}, {"question_id": "q2", "behavior_label": "retention"},
           {"question_id": "q3", "behavior_label": "agreement"}]


def test_collection_failures_count_as_out(plot, tmp_path):
    counts = plot.cohort_counts(write(tmp_path / "l.jsonl", LABELED),
                                write(tmp_path / "x.jsonl", [{"question_id": "q4", "stage": "critic_recommendation"}]))
    assert counts == {"adoption": 1, "retention": 1, "agreement": 1, "out": 1, "n": 4, "collection_failures": 1}
    row = plot.table_row("c", counts)
    assert row["a_plus_r_share"] == 0.5 and row["adoption_share_of_a_plus_r"] == 0.5


def test_writes_vector_figure_table_and_input_hashes(plot, tmp_path):
    labels = write(tmp_path / "l.jsonl", LABELED)
    plot.main(["--cohort", "A", str(labels), "--figures", str(tmp_path / "f"), "--tables", str(tmp_path / "t")])
    assert (tmp_path / "f/synthetic_vs_musique_behavior.pdf").stat().st_size > 0
    assert (tmp_path / "f/synthetic_vs_musique_behavior.svg").read_text().lstrip().startswith("<?xml")
    assert json.loads((tmp_path / "t/synthetic_vs_musique_inputs.json").read_text())["A"][str(labels)]


def test_refuses_inputs_that_change_while_read(plot, tmp_path, monkeypatch):
    labels = write(tmp_path / "l.jsonl", LABELED)
    calls = iter(range(100))
    monkeypatch.setattr(plot, "file_sha256", lambda path: str(next(calls)))
    with pytest.raises(SystemExit, match="changed while it was read"):
        plot.main(["--cohort", "A", str(labels), "--figures", str(tmp_path / "f"), "--tables", str(tmp_path / "t")])
    assert not (tmp_path / "f").exists() and not (tmp_path / "t").exists()


@pytest.mark.parametrize("labels, exclusions, message", [
    (LABELED + LABELED[:1], [], "labeled twice"),
    (LABELED + [{"question_id": "q9", "behavior_label": None, "exclusion_reason": "parse_error"}], [], "retry it first"),
    (LABELED, [{"question_id": "q1"}], "also labeled"),
    (LABELED, [{"question_id": "q4"}, {"question_id": "q4"}], "repeat"),
])
def test_refuses_inconsistent_inputs(plot, tmp_path, labels, exclusions, message):
    with pytest.raises(SystemExit, match=message):
        plot.cohort_counts(write(tmp_path / "l.jsonl", labels), write(tmp_path / "x.jsonl", exclusions))
