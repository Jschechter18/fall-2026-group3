"""The repository root for tests, found the same way however deep a test file sits.

Tests import ``REPO_ROOT`` from here instead of counting ``Path(__file__).parents[N]``, so a test
file can move between folders without its paths changing. ``pytest.ini`` puts ``src/tests`` on the
import path.
"""

from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
