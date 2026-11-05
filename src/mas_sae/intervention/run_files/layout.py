"""Where each stage writes its files, under the config's output_dir.

  pilot/episodes.json          the pilot's episode ids (positions N+1..N+P), refused by prepare
  pilot/rows.jsonl             pilot generations;  pilot/labeling/  its Claude responses
  pilot/pilot_report.{md,csv}  the report the edit size is chosen from
  frozen_run.json              everything a worker must reproduce, frozen by prepare
  baseline.jsonl               prepare's baseline generation of every study episode
  baseline_check.json          does the baseline reproduce the stored A2s?  (+ baseline_mismatches.csv)
  shards.json                  the episode ids of each shard, and the frozen run's sha256
  shard_K.jsonl                worker K's generations;  shard_K.complete.json  its completion marker
  merge/                       labels (labeling/, labeled_rows.jsonl), tables, summary.json
"""

from pathlib import Path


def pilot_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "pilot"


def pilot_episodes(run_dir: Path) -> Path:
    return pilot_dir(run_dir) / "episodes.json"


def pilot_rows(run_dir: Path) -> Path:
    return pilot_dir(run_dir) / "rows.jsonl"


def pilot_report_csv(run_dir: Path) -> Path:
    return pilot_dir(run_dir) / "pilot_report.csv"


def pilot_report_md(run_dir: Path) -> Path:
    return pilot_dir(run_dir) / "pilot_report.md"


def frozen_run(run_dir: Path) -> Path:
    return Path(run_dir) / "frozen_run.json"


def baseline_rows(run_dir: Path) -> Path:
    return Path(run_dir) / "baseline.jsonl"


def baseline_check(run_dir: Path) -> Path:
    return Path(run_dir) / "baseline_check.json"


def baseline_mismatches(run_dir: Path) -> Path:
    return Path(run_dir) / "baseline_mismatches.csv"


def shards(run_dir: Path) -> Path:
    return Path(run_dir) / "shards.json"


def merge_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "merge"
