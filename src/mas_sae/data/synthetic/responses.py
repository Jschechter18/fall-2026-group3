"""JSON response parsing shared by generation and dataset auditing."""
import json


def parse_response(text: str) -> dict:
    """Parse one JSON object, allowing a surrounding Markdown code fence."""
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].strip().lower() in ("```json", "```") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1])
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("Expected one JSON object.")
    return raw
