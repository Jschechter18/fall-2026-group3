"""SAE feature selection (issue #89) and the offline causal-proxy table on real activations of the config's layer.

    python scripts/feature_selection.py --config configs/feature_selection_layer33.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from mas_sae.probe.feature_selection import FeatureSelectionConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/feature_selection_layer33.yaml"))
    parser.add_argument("--matched-from", type=Path,
                        help="Completed feature-selection run to reuse for matched-size evaluation")
    parser.add_argument("--edit-doses", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1],
                        help="Positive edit sizes as fractions of each original activation norm")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = FeatureSelectionConfig.from_yaml(args.config)
    if args.matched_from:
        from mas_sae.probe.matched_edits import run as run_matched
        run_dir = run_matched(cfg, args.matched_from, args.edit_doses, args.config)
        print(f"Matched-size evaluation: {run_dir}")
        return
    run_dir = run(cfg, config_path=args.config)

    summary = json.loads((run_dir / "feature_selection_summary.json").read_text())
    for name, s in summary["saes"].items():
        p = s["probe"]
        print(f"{name}: probe balanced accuracy {p['balanced_accuracy_eval']:.3f} "
              f"[{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]; targeted features {s['targeted_features']}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + pd.read_csv(run_dir / "causal_proxy_table.csv").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{summary['outcome_note']}\nFull results: {run_dir}")


if __name__ == "__main__":
    main()
