"""Label rates per hop type for a PR #153 label build: counts, A1 accuracy, and refusals (#174)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/experiments/compare_label_rates.py"


@pytest.fixture
def rates():
    spec = importlib.util.spec_from_file_location("compare_label_rates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LABELS = [
    {"question_id": "musique:2hop__1_2", "label": "adoption", "collected": True, "partition": "train"},
    {"question_id": "synthetic:2hop__synthetic_aa", "label": "retention", "collected": True,
     "partition": "validation"},
    {"question_id": "synthetic:3hop1__synthetic_bb", "label": "agreement", "collected": True,
     "partition": "train"},
    {"question_id": "4hop2__3_4_5_6", "label": "out", "collected": False, "partition": None},
]
INTERACTIONS = [
    {"question_id": "musique:2hop__1_2", "solver_attempt_1_correct": False},
    {"question_id": "synthetic:2hop__synthetic_aa", "solver_attempt_1_correct": True},
    {"question_id": "synthetic:3hop1__synthetic_bb", "solver_attempt_1_correct": True},
]


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def run(rates, tmp_path, labels=LABELS, interactions=(INTERACTIONS,)):
    files = [write(tmp_path / f"interactions_{i}.jsonl", rows) for i, rows in enumerate(interactions)]
    output = tmp_path / "out"
    rates.main(["--labels", str(write(tmp_path / "labels.jsonl", labels)),
                "--interactions", *map(str, files), "--output", str(output)])
    return output


def scopes(output):
    rows = json.loads((output / "summary.json").read_text())["rows"]
    return [(row["partition"], row["hop_type"]) for row in rows], {
        (row["partition"], row["hop_type"]): row for row in rows}


def test_counts_all_labels_and_a1_over_collected_questions_only(rates, tmp_path):
    order, rows = scopes(run(rates, tmp_path))
    assert order[0] == ("all", "all")
    everything = rows["all", "all"]
    assert (everything["n"], everything["binary"], everything["out"]) == (4, 2, 1)
    assert everything["binary_share"] == 0.5 and everything["adoption_of_binary_share"] == 0.5
    assert (everything["collected"], everything["a1_correct"]) == (3, 2)
    two_hop = rows["all", "2hop"]  # musique: and synthetic: namespaces share the 2hop scope
    assert (two_hop["n"], two_hop["adoption"], two_hop["retention"]) == (2, 1, 1)
    unpartitioned = rows["unpartitioned", "all"]  # a null partition is reported, not dropped
    assert unpartitioned["out"] == 1 and unpartitioned["a1_accuracy_share"] is None


def test_reports_each_partition_and_partition_by_hop(rates, tmp_path):
    order, rows = scopes(run(rates, tmp_path))
    assert [p for p, hop in order if hop == "all"] == ["all", "train", "validation", "unpartitioned"]
    train, validation = rows["train", "all"], rows["validation", "all"]
    assert (train["n"], train["adoption_share"], train["agreement_share"]) == (2, 0.5, 0.5)
    assert (validation["n"], validation["retention_share"], validation["adoption_share"]) == (1, 1.0, 0.0)
    assert train["adoption_of_binary_share"] == 1.0 and validation["adoption_of_binary_share"] == 0.0
    assert (rows["train", "3hop1"]["n"], rows["train", "3hop1"]["agreement"]) == (1, 1)
    assert ("validation", "3hop1") not in rows                       # no empty scopes


def test_records_input_hashes_and_writes_the_table(rates, tmp_path):
    output = run(rates, tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["inputs"]) == 2 and all(len(value) == 64 for value in summary["inputs"].values())
    assert (output / "label_rates.csv").read_text().startswith("partition,hop_type,n,adoption,")


@pytest.mark.parametrize("labels, interactions, message", [
    (LABELS + [LABELS[0]], (INTERACTIONS,), "repeated question_id"),
    (LABELS, (INTERACTIONS, INTERACTIONS[:1]), "repeated question_id"),
    (LABELS[1:], (INTERACTIONS,), "interactions without a label"),
    (LABELS, (INTERACTIONS[1:],), "disagrees with the interactions"),
    (LABELS, (INTERACTIONS + [{"question_id": "4hop2__3_4_5_6", "solver_attempt_1_correct": False}],),
     "disagrees with the interactions"),
    ([{**LABELS[0], "label": "adopted"}, *LABELS[1:]], (INTERACTIONS,), "unknown labels"),
    ([{**LABELS[0], "partition": "dev"}, *LABELS[1:]], (INTERACTIONS,), "unknown partition"),
    ([{k: v for k, v in LABELS[0].items() if k != "partition"}, *LABELS[1:]], (INTERACTIONS,),
     "no partition field"),
    (LABELS, ([{**INTERACTIONS[0], "solver_attempt_1_correct": "True"}, *INTERACTIONS[1:]],),
     "not a boolean"),
])
def test_refuses_inconsistent_inputs(rates, tmp_path, labels, interactions, message):
    with pytest.raises(ValueError, match=message):
        run(rates, tmp_path, labels, interactions)
    assert not (tmp_path / "out").exists()


def test_refuses_an_existing_output(rates, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run(rates, tmp_path)


def test_refuses_inputs_that_change_while_reading(rates, tmp_path, monkeypatch):
    calls = iter(range(1000))
    monkeypatch.setattr(rates, "sha256_file", lambda path: f"{next(calls):064d}")
    with pytest.raises(RuntimeError, match="inputs changed"):
        run(rates, tmp_path)
    assert not (tmp_path / "out").exists()
