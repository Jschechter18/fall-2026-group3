"""Causal replay (issue #90): regenerate Solver A2 with SAE ablations at the configured layer and write the table.

Steering (issue #139) runs the same replay with each feature set clamped to s * ref instead of zeroed. The
reference activations come from the layer's feature-selection run, and are computed once per layer first:

    python scripts/feature_reference_activations.py --config configs/feature_selection_layer33.yaml \
        --summary results/feature_selection/20261016T003252665105Z/feature_selection_summary.json
    python scripts/causal_replay.py --config configs/causal_steering_layer33.yaml --limit 5    # check s/generation
    python scripts/causal_replay.py --config configs/causal_steering_layer33.yaml

(and the same with configs/feature_selection_layer25.yaml, its summary and configs/causal_steering_layer25.yaml).

Edit-size-matched replays (issue #146) rescale every targeted and random edit to the same size, with random controls
matched on firing frequency and decoded contribution. Those draws use per-latent statistics that the reference file
has only once recomputed by this version (add --overwrite to the command above); then:

    python scripts/causal_replay.py --config configs/causal_edit_size_layer25.yaml --limit 5   # check s/generation
    python scripts/causal_replay.py --config configs/causal_edit_size_layer25.yaml            # all 517 episodes

    # GPU smoke test (a few episodes, real model and SAEs), then the full run
    python scripts/causal_replay.py --config configs/causal_replay_layer33.yaml --limit 5
    python scripts/causal_replay.py --config configs/causal_replay_layer33.yaml

    # plain replays of every episode and the replay check only (replay_check.json), before the SAE conditions;
    # the full run starts with the same step, and --resume continues from it
    python scripts/causal_replay.py --config configs/causal_replay_layer33.yaml --preflight-only

    # continue an interrupted run, or rebuild the table from whatever has been generated so far
    python scripts/causal_replay.py --config configs/causal_replay_layer33.yaml --resume results/causal_replay/<run>
    python scripts/causal_replay.py --analyse-only results/causal_replay/<run>

    # CPU smoke test: real episodes and labels, tiny random model and SAEs (checks outputs, not numbers)
    python scripts/causal_replay.py --config configs/causal_replay_smoke_validation.yaml --smoke
"""

import argparse
import dataclasses
import json
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
    parser.add_argument("--preflight-only", action="store_true",
                        help="run the plain replays and the replay check, then stop")
    parser.add_argument("--prepare-inputs", action="store_true",
                        help="build/validate labels and feature references using input_preparation in the YAML; no Solver load")
    parser.add_argument("--label-only", type=Path, metavar="RUN",
                        help="coordinator-only Claude labeling and analysis of a completed generation run")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --label-only: preview distinct pending Claude requests without API calls or writes")
    args = parser.parse_args()
    if args.prepare_inputs and args.analyse_only:
        parser.error("--prepare-inputs and --analyse-only are separate stages")
    if args.dry_run and not args.label_only:
        parser.error("--dry-run requires --label-only")
    if args.label_only and (args.prepare_inputs or args.analyse_only or args.resume or args.preflight_only or args.smoke or args.limit):
        parser.error("--label-only is a separate coordinator stage; use only RUN and optional --dry-run")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.label_only:
        from mas_sae.causal.semantic import label_run
        run_dir = args.label_only
        report = label_run(run_dir, dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        if args.dry_run:
            return
        replay.analyse(run_dir)
    elif args.analyse_only:
        run_dir = args.analyse_only
        replay.analyse(run_dir)
    else:
        cfg = replay.ReplayConfig.from_yaml(args.config)
        if args.prepare_inputs:
            if args.smoke or args.resume or args.preflight_only or args.limit:
                parser.error("--prepare-inputs uses only --config; do not combine it with generation options")
            from mas_sae.causal.prepare import prepare_inputs
            prepare_inputs(cfg)
            print("Replay inputs prepared and validated.")
            return
        if args.limit:
            cfg = dataclasses.replace(cfg, limit=args.limit)
        if args.smoke:
            from mas_sae.causal.smoke import smoke_components, tiny_features, tiny_saes
            from mas_sae.utils.versioning import create_versioned_run_dir
            episodes = replay.load_episodes(cfg)
            saes = tiny_saes()
            comp = smoke_components(replay.load_paragraphs(episodes, cfg.musique_revision, cfg.capstone_data_root / cfg.contexts_jsonl if cfg.contexts_jsonl else None), cfg.layer, saes,
                                    tiny_features(saes, n_random_draws=cfg.n_random_draws))
            if cfg.probe_direction:
                # A smoke run uses toy components throughout, including its learned direction.
                from mas_sae.causal.smoke import HIDDEN
                comp.probe_vector = list(range(1, HIDDEN + 1))
            run_dir = args.resume or create_versioned_run_dir(cfg.output_dir)
            replay.replay(cfg, episodes, comp, run_dir, args.config, args.preflight_only)
            if not args.preflight_only and cfg.label_policy != "claude":
                replay.analyse(run_dir)
        else:
            run_dir = replay.run(cfg, config_path=args.config, resume=args.resume, preflight_only=args.preflight_only)
        if args.preflight_only:
            check = json.loads((run_dir / "replay_check.json").read_text())
            print(json.dumps({k: v for k, v in check.items() if k != "criterion"}, indent=2))
            print(f"\nReplay check {'passed' if check['passed'] else 'FAILED'}: {run_dir}")
            return
        if cfg.label_policy == "claude":
            print(f"\nGenerations saved: {run_dir}\nRun --label-only {run_dir} on the coordinator to label and analyse.")
            return

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n" + pd.read_csv(run_dir / "causal_table.csv").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFull results: {run_dir}")


if __name__ == "__main__":
    main()
