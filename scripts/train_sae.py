"""
python scripts/train_sae.py \
  --run-name natural_4b_full9_partitioned \
  --layer 33 \
  --sparsity-mode topk \
  --top-k 512
  
python scripts/train_sae.py \
  --run-name natural_4b_full9_partitioned \
  --layer 33
"""

import argparse
from pathlib import Path

from mas_sae.sae.hyperparamters import Hyperparameters as HP
from mas_sae.sae.training import train_sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--hp-tune", type=bool, default=False, required=False)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sparsity-mode", choices=["topk", "l1"], default="l1")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    hp = HP()
    hp.sparsity_mode = args.sparsity_mode
    if args.top_k is not None:
        hp.top_k = args.top_k
    if hp.sparsity_mode == "topk" and (
        hp.top_k is None or not 1 <= hp.top_k <= hp.latent_dim
    ):
        parser.error(f"TopK requires 1 <= top_k <= {hp.latent_dim}.")

    project_root = Path(__file__).resolve().parents[1]
    result = train_sae(
        hp=hp,
        activation_location=(
            Path("data") / "activations" / args.run_name / f"layer_{args.layer:02d}"
        ),
        activation_run_name=args.run_name,
        layer=args.layer,
        results_root=project_root / "results" / "sae" / "musique",
        project_root=project_root,
    )
    print(f"Checkpoint saved to {result.checkpoint_path}")


if __name__ == "__main__":
    main()
