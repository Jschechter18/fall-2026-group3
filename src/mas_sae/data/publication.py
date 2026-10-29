"""Shared readiness checks for distributed result and activation readers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PUBLICATION_FILE = "publication.json"
READY_FILE = "publication_ready.json"
COMPLETE_FILE = "publication_complete.json"


def read_publication(root: Path) -> dict[str, Any]:
    """Read a run's ownership identity, rejecting malformed publication metadata."""
    try:
        identity = json.loads((root / PUBLICATION_FILE).read_text())
        if (not isinstance(identity, dict) or identity.get("schema_version") != 1
                or any(not isinstance(identity.get(key), str) or not identity[key]
                       for key in ("run_id", "run_name", "plan_sha256"))):
            raise ValueError("invalid publication identity")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"{root}: missing or invalid distributed publication identity") from error
    return identity


def require_complete_publication(location: str | Path, *, required: bool = False) -> None:
    """Reject unfinished distributed runs, leaving ordinary legacy stores usable.

    Activation readers receive a layer directory or tensor path, so inspect
    ancestors for the owning run. Result readers can require the metadata even
    for older distributed runs that predate publication identities.
    """
    location = Path(location)
    for root in (location, *location.parents):
        if not ((root / PUBLICATION_FILE).exists() or (root / COMPLETE_FILE).exists()
                or (root / "run_layout.json").exists()):
            continue
        identity = read_publication(root)
        try:
            complete = json.loads((root / COMPLETE_FILE).read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError(f"{root}: distributed run publication is incomplete; recover or re-finalize it") from error
        if complete != identity:
            raise RuntimeError(f"{root}: distributed publication completion identity does not match")
        return
    if required:
        raise RuntimeError(f"{location}: distributed run publication is incomplete or predates completion markers; re-finalize it")
