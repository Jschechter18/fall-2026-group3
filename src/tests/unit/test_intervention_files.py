"""Resumable results files, shards, completion markers and merge checks."""

import json

import pytest

from mas_sae.intervention.run_files.results_file import ResultsFile
from mas_sae.intervention.run_files.shards import (
    mark_shard_complete, read_complete_shards, read_excluded, record_excluded, shard_marker_path, shard_rows_path,
    split_into_shards,
)

CONDITIONS = ["baseline", "edit|SAE|both|targeted||0.05"]


def _row(episode_id, condition):
    return {"episode_id": episode_id, "condition": condition, "answer": "a"}


def test_resume_after_a_truncated_last_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row("e1", "baseline")) + "\n" + '{"episode_id": "e2", "condi')
    results = ResultsFile(path)
    assert [r["episode_id"] for r in results.rows] == ["e1"] and path.read_text().endswith("\n")
    results.append(_row("e2", "baseline"))
    assert [r["episode_id"] for r in ResultsFile(path).rows] == ["e1", "e2"]


def test_a_corrupt_line_before_the_end_is_an_error(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("not json\n" + json.dumps(_row("e1", "baseline")) + "\n")
    with pytest.raises(ValueError, match="line 1 is not valid JSON"):
        ResultsFile(path)


def test_a_row_is_never_written_twice(tmp_path):
    results = ResultsFile(tmp_path / "rows.jsonl")
    results.append(_row("e1", "baseline"))
    assert results.has("e1", "baseline") and not results.has("e1", CONDITIONS[1])
    with pytest.raises(ValueError, match="already has"):
        results.append(_row("e1", "baseline"))


def test_shards_are_contiguous_and_balanced():
    shards = split_into_shards([f"e{i}" for i in range(10)], 3)
    assert shards == [["e0", "e1", "e2", "e3"], ["e4", "e5", "e6"], ["e7", "e8", "e9"]]
    with pytest.raises(ValueError):
        split_into_shards(["e0"], 2)


def _write_shard(run_dir, shard, episode_ids, conditions=CONDITIONS):
    results = ResultsFile(shard_rows_path(run_dir, shard))
    for e in episode_ids:
        for c in conditions:
            results.append(_row(e, c))


def test_an_incomplete_shard_is_not_marked_complete(tmp_path):
    _write_shard(tmp_path, 0, ["e1", "e2"], conditions=CONDITIONS[:1])
    with pytest.raises(RuntimeError, match="2 missing"):
        mark_shard_complete(tmp_path, 0, ["e1", "e2"], CONDITIONS, "frozen")
    assert not shard_marker_path(tmp_path, 0).exists()


def test_merge_reads_complete_shards(tmp_path):
    shards = [["e1", "e2"], ["e3"]]
    for k, ids in enumerate(shards):
        _write_shard(tmp_path, k, ids)
        assert mark_shard_complete(tmp_path, k, ids, CONDITIONS, "frozen")["row_count"] == 2 * len(ids)
    rows, excluded = read_complete_shards(tmp_path, shards, CONDITIONS, "frozen")
    assert len(rows) == 6 and excluded == {}


def test_an_excluded_episode_is_dropped_at_merge_even_with_leftover_rows(tmp_path):
    shards = [["e1", "e2"], ["e3"]]
    _write_shard(tmp_path, 0, ["e1"])
    _write_shard(tmp_path, 0, ["e2"], conditions=CONDITIONS[:1])         # e2: a partial earlier attempt
    excluded = record_excluded(tmp_path, 0, "e2", "baseline differs")
    assert read_excluded(tmp_path, 0) == {"e2": "baseline differs"}
    assert mark_shard_complete(tmp_path, 0, shards[0], CONDITIONS, "frozen", excluded)["row_count"] == 3
    _write_shard(tmp_path, 1, ["e3"])
    mark_shard_complete(tmp_path, 1, shards[1], CONDITIONS, "frozen")
    rows, excluded = read_complete_shards(tmp_path, shards, CONDITIONS, "frozen")
    assert {r["episode_id"] for r in rows} == {"e1", "e3"} and excluded == {"e2": "baseline differs"}


def test_a_shard_cannot_exclude_an_episode_it_does_not_have(tmp_path):
    _write_shard(tmp_path, 0, ["e1"])
    with pytest.raises(RuntimeError, match="not in it"):
        mark_shard_complete(tmp_path, 0, ["e1"], CONDITIONS, "frozen", {"e9": "baseline differs"})


def test_merge_lists_every_problem_with_the_shards(tmp_path):
    shards = [["e1"], ["e2"], ["e3"]]
    for k, ids in enumerate(shards):
        _write_shard(tmp_path, k, ids)
        mark_shard_complete(tmp_path, k, ids, CONDITIONS, "frozen")
    shard_marker_path(tmp_path, 0).unlink()                                  # never marked complete
    ResultsFile(shard_rows_path(tmp_path, 1)).append(_row("e2", "extra"))    # changed after marking
    marker = json.loads(shard_marker_path(tmp_path, 2).read_text())
    shard_marker_path(tmp_path, 2).write_text(json.dumps({**marker, "frozen_run_sha256": "other"}))
    with pytest.raises(RuntimeError) as error:
        read_complete_shards(tmp_path, shards, CONDITIONS, "frozen")
    message = str(error.value)
    assert "shard 0 has no completion marker" in message
    assert "shard 1's rows changed" in message and "shard 1 has 3 rows" in message
    assert "shard 2 was generated for a different frozen run" in message
    assert "2 conditions missing, 1 unexpected" in message
