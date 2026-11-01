"""Reference activations for causal-replay steering: each targeted and random-control feature's mean
activation where it fires on the fit partition, for the SAEs of one feature-selection run, plus per-latent
statistics of every latent (firing frequency, mean activation where firing, RMS, decoder norm) that the
replay draws contribution-matched random controls from (``random_controls: contribution``).

The result is written next to the summary it comes from, so each layer has its own:

    python scripts/feature_reference_activations.py --config configs/feature_selection_layer33.yaml \
        --summary results/feature_selection/20261016T003252665105Z/feature_selection_summary.json
    python scripts/feature_reference_activations.py --config configs/feature_selection_layer25.yaml \
        --summary results/feature_selection/20261018T025217187826Z/feature_selection_summary.json
"""

import argparse
import json
import logging
from pathlib import Path

from mas_sae.experiments.artifacts import get_git_commit, get_run_command, sha256_file
from mas_sae.probe.feature_selection import FeatureSelectionConfig, reference_activations

OUTPUT_NAME = "feature_reference_activations.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the feature-selection config of that run")
    parser.add_argument("--summary", type=Path, required=True, help="its feature_selection_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = args.summary.parent / OUTPUT_NAME
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to recompute it.")
    cfg = FeatureSelectionConfig.from_yaml(args.config)
    summary = json.loads(args.summary.read_text())
    saes = reference_activations(cfg, summary)
    out.write_text(json.dumps({
        "layer": cfg.layer,
        "fit_partition": cfg.fit_partition,
        "feature_selection_summary": str(args.summary),
        "feature_selection_summary_sha256": sha256_file(args.summary),
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "git_commit": get_git_commit(),
        "command": get_run_command(),
        "saes": saes,
    }, indent=2))
    for name, s in saes.items():
        refs = sorted(f["ref"] for f in s["features"].values())
        print(f"{name}: {len(refs)} features, ref median {refs[len(refs) // 2]:.3f} (min {refs[0]:.3f}, max {refs[-1]:.3f})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
