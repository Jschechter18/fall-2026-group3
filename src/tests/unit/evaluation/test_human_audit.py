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
    score_audit,
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


# ---------------------------------------------------------------- scoring the returned files

def fill_annotations(root, selection, claude_labels, override=None, *, excel=False, edit=None):
    """Write each annotator's file back with Claude's label, except the ``override`` item labels.

    ``excel`` re-saves the way Excel does (byte-order mark, CRLF, capitalized label with a space);
    ``edit`` maps a file name to a function that changes its rows before writing.
    """
    override = override or {}
    episode_of = {it["item_id"]: it["episode_id"] for it in selection["items"]}
    for name in selection["annotators"]:
        path = root / "out" / "annotations" / f"{name}.csv"
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            label = override.get(row["item_id"], claude_labels[episode_of[row["item_id"]]])
            row["human_label"] = f"{label.capitalize()} " if excel else label
        if edit and name in edit:
            rows = edit[name](rows)
        with path.open("w", newline="", encoding="utf-8-sig" if excel else "utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\r\n" if excel else "\n")
            writer.writeheader()
            writer.writerows(rows)


def packet(tmp_path, rows=None):
    rows = pool() if rows is None else rows
    config = write_inputs(tmp_path, rows)
    selection = build_packet(config, root=tmp_path)
    return config, selection, {r["episode_id"]: r["claude_label"] for r in rows}


def test_score_audit_full_agreement(tmp_path):
    config, selection, claude = packet(tmp_path)
    fill_annotations(tmp_path, selection, claude)
    record = score_audit(config, root=tmp_path)
    scores = record["scores"]
    assert (scores["agree"], scores["n"], scores["agreement"], scores["cohen_kappa"]) == (12, 12, 1.0, 1.0)
    assert all(m["f1"] == 1.0 for m in scores["per_class"].values())
    out = tmp_path / "out" / "scoring"
    assert not (out / "disagreements.csv").exists()
    assert record["output_sha256"] == {"report.md": sha256_file(out / "report.md")}
    assert json.loads((out / "scores.json").read_text())["scores"] == json.loads(json.dumps(scores))
    assert record["git_dirty"] in (True, False) and len(record["items"]) == 12
    report = (out / "report.md").read_text()
    assert "| Human–Claude agreement | 12/12 = 100.0%" in report
    assert "| adoption | 100.0% | 100.0% | 1.000 | 3 | 3 |" in report
    assert "israel" not in report  # the report scores Claude, not the annotators


def test_score_audit_scores_disagreements_with_the_human_as_reference(tmp_path):
    config, selection, claude = packet(tmp_path)
    label_of = {it["item_id"]: claude[it["episode_id"]] for it in selection["items"]}
    adoption = next(i for i, l in sorted(label_of.items()) if l == "adoption")
    retention = next(i for i, l in sorted(label_of.items()) if l == "retention")
    fill_annotations(tmp_path, selection, claude, {adoption: "out", retention: "adoption"})
    scores = score_audit(config, root=tmp_path)["scores"]
    assert (scores["agree"], scores["n"]) == (10, 12)
    matrix = scores["confusion_human_rows_claude_columns"]
    assert matrix["out"] == {"adoption": 1, "retention": 0, "agreement": 0, "out": 3}
    assert matrix["adoption"] == {"adoption": 2, "retention": 1, "agreement": 0, "out": 0}
    per = scores["per_class"]
    assert (per["adoption"]["precision"], per["adoption"]["recall"]) == pytest.approx((2 / 3, 2 / 3))
    assert (per["retention"]["precision"], per["retention"]["recall"]) == pytest.approx((2 / 3, 1.0))
    assert (per["out"]["precision"], per["out"]["recall"], per["out"]["f1"]) == pytest.approx((1.0, 0.75, 6 / 7))
    with (tmp_path / "out" / "scoring" / "disagreements.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["item_id"] for r in rows] == sorted([adoption, retention])
    assert all(r["claude_reason"].startswith("secret reason") for r in rows)
    assert "## Disagreements (2)" in (tmp_path / "out" / "scoring" / "report.md").read_text()


def test_score_audit_accepts_an_excel_resave(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    plain = packet(tmp_path / "a")
    fill_annotations(tmp_path / "a", plain[1], plain[2])
    excel = packet(tmp_path / "b")
    fill_annotations(tmp_path / "b", excel[1], excel[2], excel=True)
    record = score_audit(excel[0], root=tmp_path / "b")
    assert record["scores"] == score_audit(plain[0], root=tmp_path / "a")["scores"]
    path = tmp_path / "b" / "out" / "annotations" / "israel.csv"
    assert b"\r\n" in path.read_bytes()
    assert record["input_sha256"]["out/annotations/israel.csv"] == \
        hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_score_audit_refuses_blank_or_unknown_labels(tmp_path):
    config, selection, claude = packet(tmp_path)
    first, second = sorted(it["item_id"] for it in selection["items"] if it["annotator"] == "josh")[:2]
    fill_annotations(tmp_path, selection, claude, {first: "", second: "adopted"})
    with pytest.raises(ValueError, match=rf"blank or unknown labels: {first}='', {second}='adopted'"):
        score_audit(config, root=tmp_path)
    assert not (tmp_path / "out" / "scoring").exists()


def edit_a2(rows):
    rows[0]["A2"] += " (edited)"
    return rows


def drop_row(rows):
    return rows[1:]


def repeat_row(rows):
    return rows + rows[:1]


def rename_column(rows):
    return [{("label" if k == "human_label" else k): v for k, v in r.items()} for r in rows]


@pytest.mark.parametrize("edit, message", [(edit_a2, "no longer shows"), (drop_row, "items differ"),
                                           (repeat_row, "repeats"), (rename_column, "columns")])
def test_score_audit_refuses_a_changed_annotation_file(tmp_path, edit, message):
    config, selection, claude = packet(tmp_path)
    fill_annotations(tmp_path, selection, claude, edit={"raye": edit})
    with pytest.raises(ValueError, match=message):
        score_audit(config, root=tmp_path)


def test_score_audit_refuses_an_item_moved_to_another_file(tmp_path):
    config, selection, claude = packet(tmp_path)
    moved = []
    fill_annotations(tmp_path, selection, claude, edit={"israel": lambda rows: moved.extend(rows[:1]) or rows[1:]})
    fill_annotations(tmp_path, selection, claude, edit={"josh": lambda rows: rows + moved})
    with pytest.raises(ValueError, match="items differ"):
        score_audit(config, root=tmp_path)


def test_score_audit_refuses_labels_changed_since_the_packet(tmp_path):
    config, selection, claude = packet(tmp_path)
    fill_annotations(tmp_path, selection, claude)
    with (tmp_path / "run" / "raw_results.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="changed since the packet was built"):
        score_audit(config, root=tmp_path)


def test_score_audit_refuses_a_different_config(tmp_path):
    config, selection, claude = packet(tmp_path)
    fill_annotations(tmp_path, selection, claude)
    with pytest.raises(ValueError, match="different config"):
        score_audit({**config, "seed": config["seed"] + 1}, root=tmp_path)


def test_score_audit_never_overwrites(tmp_path):
    config, selection, claude = packet(tmp_path)
    fill_annotations(tmp_path, selection, claude)
    score_audit(config, root=tmp_path)
    with pytest.raises(FileExistsError):
        score_audit(config, root=tmp_path)


# israel.csv is read first, so touching it after each read changes a file already read.
@pytest.mark.parametrize("changed", ["run/raw_results.jsonl", "out/annotations/israel.csv"])
def test_score_audit_refuses_an_input_that_changes_while_scoring(tmp_path, monkeypatch, changed):
    import mas_sae.evaluation.human_audit as audit

    config, selection, claude = packet(tmp_path)
    fill_annotations(tmp_path, selection, claude)
    real_read = audit.read_annotations

    def read_then_touch(*args):
        answers = real_read(*args)
        with (tmp_path / changed).open("a") as f:
            f.write(" ")
        return answers

    monkeypatch.setattr(audit, "read_annotations", read_then_touch)
    with pytest.raises(ValueError, match="changed while scoring"):
        score_audit(config, root=tmp_path)
    assert not (tmp_path / "out" / "scoring").exists()
