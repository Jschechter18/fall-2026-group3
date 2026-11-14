"""Score the returned human-audit files against the Claude labels (#237).

    python scripts/labels/score_human_audit.py --config configs/labels/human_audit_validation_100.yaml

Checks each ``<packet>/annotations/<annotator>.csv`` against the packet (same items, unedited text,
valid labels), joins Claude's labels through ``selection.json`` and writes ``<packet>/scoring/``:
``scores.json`` (agreement, kappa, confusion matrix, precision/recall/F1 by class, per-item labels,
input and output hashes), ``disagreements.csv`` and ``report.md``. The output folder must not exist:
each scoring run is kept.
"""

import argparse
import json
from pathlib import Path

from mas_sae.evaluation.human_audit import load_audit_config, score_audit


def main() -> None:
    """Score the packet for ``--config`` and print the headline numbers."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: <packet>/scoring)")
    args = parser.parse_args()

    record = score_audit(load_audit_config(args.config), output=args.out)
    scores = record["scores"]
    print(json.dumps({
        "agree": scores["agree"], "n": scores["n"], "agreement": scores["agreement"], "ci95": scores["ci95"],
        "cohen_kappa": scores["cohen_kappa"], "git_dirty": record["git_dirty"],
    }, indent=2))


if __name__ == "__main__":
    main()
