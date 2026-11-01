"""Persistent normalized-exact matching; paraphrases are outside this contract."""
import json
import sqlite3
import unicodedata
from pathlib import Path

from .contracts import DuplicateMatch


def normalize_question(question: str) -> str:
    """Apply NFKC, case folding and whitespace collapse; retain punctuation."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    if not normalized:
        raise ValueError("question must not be empty")
    return normalized


class DuplicateIndex:
    """Use one shared SQLite file for workers on a single machine.

    Registration is the authoritative atomic check; a preceding lookup alone
    cannot prevent races. Each worker should open its own connection.
    """
    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path), timeout=30)
        self._db.execute("""CREATE TABLE IF NOT EXISTS questions (
            normalized_question TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            source TEXT NOT NULL
        )""")
        self._db.commit()

    def find_duplicate(self, question: str) -> DuplicateMatch | None:
        normalized = normalize_question(question)
        row = self._db.execute(
            "SELECT record_id, source FROM questions WHERE normalized_question = ?", (normalized,)
        ).fetchone()
        return DuplicateMatch(row[0], row[1], normalized) if row else None

    def _insert_question(self, record_id: str, question: str, source: str) -> bool:
        """Validate and insert within the caller's transaction; report insertion."""
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a nonempty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a nonempty string")
        normalized = normalize_question(question)
        cursor = self._db.execute(
            "INSERT OR IGNORE INTO questions VALUES (?, ?, ?)", (normalized, record_id, source)
        )
        return cursor.rowcount == 1

    def register_question(self, record_id: str, question: str, *, source: str = "synthetic") -> DuplicateMatch | None:
        """Return None on insertion, or the existing match on duplication."""
        with self._db:
            inserted = self._insert_question(record_id, question, source)
        if not inserted:
            return self.find_duplicate(question)
        return None

    def import_jsonl(self, path: str | Path, *, source: str) -> int:
        """Index id/question fields from JSONL; original .json files are JSONL too."""
        inserted = 0
        with Path(path).open(encoding="utf-8") as stream, self._db:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    inserted += self._insert_question(record["id"], record["question"], source)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid index record") from exc
        return inserted

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
