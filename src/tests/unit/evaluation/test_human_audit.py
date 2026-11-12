"""Tests for the human-audit packet (#237): sampling, assignment, blinding and the build's refusals."""

import csv
import hashlib
import json
from collections import Counter

import pytest

from mas_sae.evaluation.claude_validator import prompt_sha256
from mas_sae.evaluation.human_audit import (
    ANNOTATION_COLUMNS,
    assign_annotators,
    build_packet,
    index_unique,
    stratified_sample,
)
from mas_sae.experiments.artifacts import sha256_file

LABELS = ("adoption", "retention", "agreement", "out")
QUOTAS = {label: {"musique": 2, "synthetic": 1} for label in LABELS}  # 12 items


def label_row(i, label, source, partition="validation", label_source="claude"):
    return {"episode_id": f"{source}:q{i}__natural__candidates", "question_id": f"{source}:q{i}",
            "dataset_source": source, "partition": partition, "claude_label": label,
            "claude_reason": f"secret reason {i}", "label_source": label_source, "exclusion_reason": None}


def episode_row(row):
    return {"episode_id": row["episode_id"], "question_id": row["question_id"], "partition": row["partition"],
            "dataset_source": row["dataset_source"], "question": f"Q {row['question_id'].split(':')[1]}?",
            "solver_attempt_1": "A one", "critic_advocated_answer": "C one", "solver_attempt_2": "A two",
            "candidate_order": "a1_first" if row["question_id"].endswith(("0", "2", "4", "6", "8")) else "critic_first"}


def write_inputs(root, labels, episodes=None, settings=None):
    run = root / "run"
    run.mkdir()
    (run / "raw_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    episodes = [episode_row(r) for r in labels] if episodes is None else episodes
    (root / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    base = {"model_id": "m", "prompt_version": "claude_binary_v3_candidates", "a2_presentation": "candidates",
            "prompt_sha256": prompt_sha256(True, "candidates"),
            "episodes_sha256": sha256_file(root / "episodes.jsonl")}
    (run / "run_settings.json").write_text(json.dumps({**base, **(settings or {})}))
    return {"claude_run": "run", "interactions": "episodes.jsonl", "output": "out", "partition": "validation",
            "seed": 7, "quotas": QUOTAS, "annotators": {"israel": 4, "josh": 4, "raye": 4}}


def pool(n_per_stratum=5):
    rows, i = [], 0
    for label in LABELS:
        for source in ("musique", "synthetic"):
            for _ in range(n_per_stratum):
                rows.append(label_row(i, label, source))
                i += 1
    return rows


def stratum(row):
    return row["claude_label"], row["dataset_source"]


def test_stratified_sample_meets_quotas_and_ignores_file_order():
    rows = pool()
    first = stratified_sample(rows, stratum, QUOTAS, seed=3)
    again = stratified_sample(list(reversed(rows)), stratum, QUOTAS, seed=3)
    assert Counter(map(stratum, first)) == Counter({(l, "musique"): 2 for l in LABELS} | {(l, "synthetic"): 1 for l in LABELS})
    assert [r["episode_id"] for r in first] == [r["episode_id"] for r in again]
    assert [r["episode_id"] for r in stratified_sample(rows, stratum, QUOTAS, seed=4)] != [r["episode_id"] for r in first]


def test_stratified_sample_refuses_a_short_stratum():
    rows = [label_row(1, "out", "synthetic")]
    with pytest.raises(ValueError, match="quota is 2"):
        stratified_sample(rows, stratum, {"out": {"synthetic": 2}}, seed=1)


def test_assign_annotators_meets_sizes_and_mixes_strata():
    sample = stratified_sample(pool(), stratum, QUOTAS, seed=3)
    names = assign_annotators(sample, {"israel": 5, "josh": 4, "raye": 3})
    assert Counter(names) == {"israel": 5, "josh": 4, "raye": 3}
    assert len({stratum(r) for r, n in zip(sample, names) if n == "raye"}) == 3
    with pytest.raises(ValueError, match="sum"):
        assign_annotators(sample, {"israel": 5, "josh": 5})


def test_index_unique_refuses_repeated_ids():
    with pytest.raises(ValueError, match="repeats"):
        index_unique([{"episode_id": "a"}, {"episode_id": "a"}], "episode_id", "x")


def test_build_packet_writes_blind_files_and_a_key(tmp_path):
    config = write_inputs(tmp_path, pool())
    selection = build_packet(config, root=tmp_path)
    out = tmp_path / "out"
    rows = {}
    for name, size in config["annotators"].items():
        with (out / "annotations" / f"{name}.csv").open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ANNOTATION_COLUMNS
            rows[name] = list(reader)
        assert len(rows[name]) == size
    text = "".join((out / "annotations" / f"{n}.csv").read_text() for n in rows)
    for secret in ("secret reason", "musique:", "synthetic:", "adoption,", "retention,"):
        assert secret not in text
    assert all(r["human_label"] == "" for rs in rows.values() for r in rs)
    items = selection["items"]
    assert len({it["episode_id"] for it in items}) == 12 and len({it["item_id"] for it in items}) == 12
    assert {it["item_id"] for it in items} == {r["item_id"] for rs in rows.values() for r in rs}
    assert selection["output_sha256"]["annotations/israel.csv"] == sha256_file(out / "annotations" / "israel.csv")
    assert "adoption" in (out / "GUIDE.md").read_text() and "Two refusals are the same position" in (out / "GUIDE.md").read_text()


def test_recorded_hashes_survive_lf_normalization(tmp_path):
    # The repository stores text with LF endings (core.autocrlf=input), so a recorded hash only
    # matches the committed file if the file is written without carriage returns.
    selection = build_packet(write_inputs(tmp_path, pool()), root=tmp_path)
    for name in selection["output_sha256"]:
        data = (tmp_path / "out" / name).read_bytes()
        assert b"\r" not in data
        assert hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == selection["output_sha256"][name]


def test_build_packet_is_reproducible(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    first = build_packet(write_inputs(tmp_path / "a", pool()), root=tmp_path / "a")
    second = build_packet(write_inputs(tmp_path / "b", pool()), root=tmp_path / "b")
    assert first["items"] == second["items"]


def test_build_packet_never_overwrites(tmp_path):
    config = write_inputs(tmp_path, pool())
    build_packet(config, root=tmp_path)
    with pytest.raises(FileExistsError):
        build_packet(config, root=tmp_path)


def test_build_packet_refuses_an_episodes_file_claude_did_not_label(tmp_path):
    config = write_inputs(tmp_path, pool(), settings={"episodes_sha256": "0" * 64})
    with pytest.raises(ValueError, match="not the episodes file"):
        build_packet(config, root=tmp_path)
    assert not (tmp_path / "out").exists()


def test_build_packet_refuses_a_changed_prompt(tmp_path):
    config = write_inputs(tmp_path, pool(), settings={"prompt_sha256": "f" * 64})
    with pytest.raises(ValueError, match="prompt"):
        build_packet(config, root=tmp_path)


def test_build_packet_leaves_out_failures_and_rule_labels(tmp_path):
    rows = pool(n_per_stratum=1)
    rows += [label_row(200, None, "synthetic") | {"label_source": None, "exclusion_reason": "incomplete_response:refusal"},
             label_row(201, "out", "musique", label_source="rule:critic_noncommittal")]
    config = write_inputs(tmp_path, rows)
    config["quotas"] = {label: {"musique": 1, "synthetic": 1} for label in LABELS}
    config["annotators"] = {"israel": 3, "josh": 3, "raye": 2}
    selection = build_packet(config, root=tmp_path)
    assert selection["pool"] == 8
    assert selection["left_out_of_pool"] == {"None:incomplete_response:refusal": 1, "rule:critic_noncommittal:None": 1}
    assert "musique:q201__natural__candidates" not in {it["episode_id"] for it in selection["items"]}


@pytest.mark.parametrize("changed", ["run/raw_results.jsonl", "run/run_settings.json", "episodes.jsonl"])
def test_build_packet_refuses_an_input_that_changes_during_the_build(tmp_path, monkeypatch, changed):
    import mas_sae.evaluation.human_audit as audit

    config = write_inputs(tmp_path, pool())
    real_read = audit.read_jsonl

    def read_then_touch(path):
        rows = real_read(path)
        with (tmp_path / changed).open("a") as f:
            f.write(" ")
        return rows

    monkeypatch.setattr(audit, "read_jsonl", read_then_touch)
    with pytest.raises(ValueError, match="changed while the packet was built"):
        build_packet(config, root=tmp_path)
    assert not (tmp_path / "out").exists()


def test_build_packet_refuses_rows_outside_the_partition(tmp_path):
    rows = [label_row(i, r["claude_label"], r["dataset_source"], partition="test") for i, r in enumerate(pool())]
    config = write_inputs(tmp_path, rows)
    with pytest.raises(ValueError, match="partition"):
        build_packet(config, root=tmp_path)


def test_build_packet_refuses_mismatched_episode_sets(tmp_path):
    rows = pool()
    config = write_inputs(tmp_path, rows, episodes=[episode_row(r) for r in rows[:-1]])
    with pytest.raises(ValueError, match="different episodes"):
        build_packet(config, root=tmp_path)
