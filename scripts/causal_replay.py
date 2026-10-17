"""Causal replay (issue #90): regenerate Solver A2 with SAE ablations at layer 33 and write the table.

    # GPU smoke test (a few episodes, real model and SAEs), then the full run
    python scripts/causal_replay.py --config configs/causal_replay_layer33.yaml --limit 5
    python scripts/causal_replay.py --config configs/causal_replay_layer33.yaml

    # continue an interrupted run, or rebuild the table from whatever has been generated so far
    python scripts/causal_replay.py --config configs/causal_replay_layer33.yaml --resume results/causal_replay/<run>
    python scripts/causal_replay.py --analyse-only results/causal_replay/<run>

    # CPU smoke test: real episodes and labels, tiny random model and SAEs (checks outputs, not numbers)
    python scripts/causal_replay.py --config configs/causal_replay_smoke_validation.yaml --smoke
"""

import argparse
import dataclasses
import logging
from pathlib import Path

import pandas as pd

from mas_sae.causal import replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/causal_replay_layer33.yaml"))
    parser.add_argument("--limit", type=int, help="replay only the first N episodes of the seeded order")
    parser.add_argument("--resume", type=Path, help="run directory to continue")
    parser.add_argument("--analyse-only", type=Path, help="rebuild the table for this run directory and exit")
    parser.add_argument("--smoke", action="store_true", help="tiny random model and SAEs on CPU")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.analyse_only:
        run_dir = args.analyse_only
        replay.analyse(run_dir)
    else:
        cfg = replay.ReplayConfig.from_yaml(args.config)
        if args.limit:
            cfg = dataclasses.replace(cfg, limit=args.limit)
        if args.smoke:
            from mas_sae.causal.smoke import smoke_components
            from mas_sae.utils.versioning import create_versioned_run_dir
            episodes = replay.load_episodes(cfg)
            comp = smoke_components(replay.load_paragraphs(episodes, cfg.musique_revision), cfg.n_random_draws)
            run_dir = args.resume or create_versioned_run_dir(cfg.output_dir)
            replay.replay(cfg, episodes, comp, run_dir, args.config)
            replay.analyse(run_dir)
        else:
            run_dir = replay.run(cfg, config_path=args.config, resume=args.resume)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n" + pd.read_csv(run_dir / "causal_table.csv").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFull results: {run_dir}")


if __name__ == "__main__":
    main()
