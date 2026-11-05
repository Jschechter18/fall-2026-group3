"""Append-only JSONL of generated rows: one line per (episode, condition), written as soon as it exists.

A process killed mid-write leaves an unfinished last line; it is cut off when the file is opened, so
the next append starts on a new line. Any other unreadable line means the file is corrupt, not
interrupted, and is an error. Rows already in the file are never generated again.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def row_key(row: dict) -> tuple[str, str]:
    return row["episode_id"], row["condition"]


def cut_unfinished_last_line(path: Path) -> None:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        logger.warning("%s: removing an unfinished last line (%d bytes, interrupted write)", path, len(data) - keep)
        with path.open("r+b") as f:
            f.truncate(keep)


def read_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {number} is not valid JSON: the file is corrupt.") from e
    return rows


class ResultsFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        cut_unfinished_last_line(self.path)
        self.rows = read_rows(self.path)
        self.keys = {row_key(r) for r in self.rows}
        if len(self.keys) != len(self.rows):
            raise ValueError(f"{self.path} has the same (episode, condition) more than once.")

    def has(self, episode_id: str, condition_key: str) -> bool:
        return (episode_id, condition_key) in self.keys

    def append(self, row: dict) -> None:
        if row_key(row) in self.keys:
            raise ValueError(f"{self.path} already has {row_key(row)}.")
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)
        self.keys.add(row_key(row))
