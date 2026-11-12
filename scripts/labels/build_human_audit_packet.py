"""Build the blind annotation packet for the human audit of the Claude labels (#237).

    python scripts/labels/build_human_audit_packet.py --config configs/labels/human_audit_validation_100.yaml

Writes ``<output>/annotations/<annotator>.csv`` (one blind file per person), ``GUIDE.md`` (the
instructions, with Claude's labeling rules) and ``selection.json`` (item id -> episode id, seed, input
and output hashes). The output folder must not exist: the packet is built once.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mas_sae.evaluation.human_audit import build_packet, load_audit_config


def main() -> None:
    """Build the packet for ``--config`` and print where each item went."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_audit_config(args.config)
    selection = build_packet(config)
    print(json.dumps({
        "output": config["output"], "pool": selection["pool"], "left_out_of_pool": selection["left_out_of_pool"],
        "items_per_annotator": dict(Counter(it["annotator"] for it in selection["items"])),
        "items_per_source": dict(Counter(it["dataset_source"] for it in selection["items"])),
    }, indent=2))


if __name__ == "__main__":
    main()
