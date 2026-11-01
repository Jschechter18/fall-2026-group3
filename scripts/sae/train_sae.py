"""Train one SAE, run a tuning study, or finish a validation-only trial.

Single runs skip test evaluation unless --evaluate-test is supplied. They are
registered as completed models; use tuning trials for later --finish-run continuation.
Finishing a selected trial evaluates the test set when test.pt exists.

python scripts/sae/train_sae.py \
  --hp-tune \
  --config configs/sae_hp_tuning/layer_33.yaml \
  --dry-run
"""

import argparse
from pathlib import Path

import yaml

from mas_sae.sae.hyperparamters import Hyperparameters as HP, validate_hyperparameters
from mas_sae.sae.training import finish_run, train_sae
from mas_sae.sae.tuning import expand_search_space, run_tuning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hp-tune", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and list tuning candidates.")
    parser.add_argument("--finish-run", type=Path, help="Continue this trial and publish the finished model under runs/.")
    parser.add_argument("--epochs", type=int, help="Total epoch budget for --finish-run.")
    parser.add_argument("--run-name")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--sparsity-mode", choices=["topk", "l1"])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the test set for a single run after model selection is complete. "
                             "Finishing a trial evaluates it automatically.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.hp_tune != (args.config is not None):
        parser.error("--hp-tune and --config must be provided together.")
    if args.dry_run and not args.hp_tune:
        parser.error("--dry-run requires --hp-tune and --config.")
    if (args.finish_run is not None) != (args.epochs is not None):
        parser.error("--finish-run and --epochs must be provided together.")
    single_settings = (args.run_name, args.layer, args.sparsity_mode, args.top_k)
    if args.hp_tune or args.finish_run is not None:
        if args.evaluate_test:
            parser.error("--evaluate-test is for single-run training only; finishing evaluates the test set automatically.")
        if any(value is not None for value in single_settings):
            parser.error("Tuning and continuation use saved settings, not single-run overrides.")
        if args.hp_tune and args.finish_run is not None:
            parser.error("Choose either tuning or finishing a trial.")

    if args.hp_tune:
        try:
            with args.config.open() as file:
                config = yaml.safe_load(file)
            candidates = expand_search_space(config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
        if args.dry_run:
            for number, (group, hp) in enumerate(candidates, start=1):
                print(f"{number}: {group}: {hp}")
            print(f"Total trials: {len(candidates)}")
            return
        study = run_tuning(config, project_root)
        print(f"Study results: {study}")
        print("Review trials.csv and selection.json before finishing a trial.")
        return

    if args.finish_run is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive (a total, not additional, epoch budget).")
        result = finish_run(args.finish_run.resolve(), args.epochs, project_root)
    else:
        if args.run_name is None or args.layer is None:
            parser.error("Single-run training requires --run-name and --layer.")
        hp = HP()
        if args.sparsity_mode is not None:
            hp.sparsity_mode = args.sparsity_mode
        if args.top_k is not None:
            hp.top_k = args.top_k
        try:
            validate_hyperparameters(hp)
        except ValueError as error:
            parser.error(str(error))
        result = train_sae(
            hp=hp,
            activation_location=(Path("data") / "activations" / args.run_name
                                 / f"layer_{args.layer:02d}"),
            activation_run_name=args.run_name, layer=args.layer,
            results_root=project_root / "results" / "sae" / "musique",
            project_root=project_root,
            evaluate_test=args.evaluate_test,
        )
    print(f"Checkpoint saved to {result.checkpoint_path}")


if __name__ == "__main__":
    main()
