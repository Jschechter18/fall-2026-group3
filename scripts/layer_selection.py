"""Layer selection under the 6 Oct split contract (development partitions only).

Uses the official full nine-layer export and Behavior v0.1.2 labels:

    python scripts/layer_selection.py --config configs/layer_selection_full9.yaml --inspect   # FIRST
    python scripts/layer_selection.py --config configs/layer_selection_full9.yaml             # fit train, rank validation
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from mas_sae.probe.layer_selection import LayerSelectionConfig, inspect_inputs, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/layer_selection_full9.yaml"))
    parser.add_argument("--inspect", action="store_true",
                        help="print schemas of labels/manifest/interactions + alignment check, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = LayerSelectionConfig.from_yaml(args.config)

    if args.inspect:
        inspect_inputs(cfg)
        return

    run_dir = run(cfg)
    df = pd.read_csv(run_dir / "layer_comparison.csv")
    show = ["name", "balanced_accuracy", "ci95_lo", "ci95_hi", "within_noise_of_best", "median_chosen_C"]
    print("\n" + df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    comp = json.loads((run_dir / "layer_selection_summary.json").read_text())["comparison"]
    print(f"\nHighest observed layer: {comp['highest_observed_layer']}")
    print(f"Not statistically separated from it: {', '.join(comp['not_separated_from_highest']) or 'none'}")
    print(comp["selection_statement"])
    print(f"Full results: {run_dir}")


if __name__ == "__main__":
    main()
