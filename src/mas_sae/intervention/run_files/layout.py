"""Where each stage writes its files, under the config's output_dir.

  choose_r/episodes.json             choose-r's episode ids (positions N+1..N+P), refused by prepare
  choose_r/rows.jsonl                choose-r generations;  choose_r/labeling/  its Claude responses
  choose_r/choose_r_report.{md,csv}  the report the edit size is chosen from
  frozen_run.json                    everything a worker must reproduce, frozen by prepare
  baseline.jsonl                     prepare's baseline generation of every study episode
  baseline_check.json                does the baseline reproduce the stored A2s?  (+ baseline_mismatches.csv)
  shards.json                        the episode ids of each shard, and the frozen run's sha256
  shard_K.jsonl                      worker K's generations;  shard_K.complete.json  its completion marker
  merge/                             labels (labeling/, labeled_rows.jsonl), tables, summary.json
"""

from pathlib import Path


def choose_r_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "choose_r"


def choose_r_episodes(run_dir: Path) -> Path:
    return choose_r_dir(run_dir) / "episodes.json"


def choose_r_rows(run_dir: Path) -> Path:
    return choose_r_dir(run_dir) / "rows.jsonl"


def choose_r_report_csv(run_dir: Path) -> Path:
    return choose_r_dir(run_dir) / "choose_r_report.csv"


def choose_r_report_md(run_dir: Path) -> Path:
    return choose_r_dir(run_dir) / "choose_r_report.md"


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
