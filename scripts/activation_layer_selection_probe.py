"""
Activation-layer selection: use a linear probe on real Solver-Critic Attempt-2
activations to decide WHICH layer is most predictive of solver_accepted_feedback,
before any SAE gets trained specifically for downstream analysis.

Uses the real train/validation split produced by the collection pipeline
directly (results/collection/<run_name>/{train,validation}/interactions.jsonl),
rather than a random split -- addressing Josh's earlier review comment that
fit_probe should support a real, predetermined split.

NOTE: as of this writing, run_name="scale_smoke_1q" is a 1-question smoke
test (3 train episodes, a handful of validation episodes) -- far too small
to draw a real layer-selection conclusion from. This run validates that the
loading/probing/comparison mechanics work correctly end-to-end on real data;
re-run against a larger collection run (once #36's scaled collection lands)
before trusting the recommended layer.

Run from the repo root with the `capstone` conda env active:
    python scripts/activation_layer_selection_probe.py
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from mas_sae.probe.data import load_real_layer_split
from mas_sae.probe.model import fit_probe


@dataclass
class ActivationLayerSelectionConfig:
    seed: int = 42
    run_name: str = "scale_smoke_1q"
    candidate_layers: list = field(default_factory=lambda: ["08", "17", "25", "33"])
    data_root: Path = Path("data/activations")
    results_root: Path = Path("results/collection")
    output_dir: Path = Path("results/activation_layer_selection")


def main():
    config = ActivationLayerSelectionConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for layer in config.candidate_layers:
        X_train, y_train = load_real_layer_split(
            config.run_name, "train", layer, config.data_root, config.results_root
        )
        X_val, y_val = load_real_layer_split(
            config.run_name, "validation", layer, config.data_root, config.results_root
        )

        print(f"layer_{layer}: {X_train.shape[0]} train episodes, {X_val.shape[0]} val episodes")

        try:
            probe, scaler, metrics, *_ = fit_probe(X_train, y_train, X_val, y_val, seed=config.seed)
            metrics.pop("f1", None)  # excluded here: unreliable on tiny smoke-test splits
                                      # (see conversation notes -- can look good on a
                                      # constant/degenerate prediction due to class
                                      # imbalance). fit_probe() itself still computes it
                                      # for callers that do want it (e.g. probe_pipeline.py).
            results[f"layer_{layer}"] = metrics
            print(f"  metrics: {metrics}")
        except ValueError as e:
            # With this small a smoke-test dataset, a split can end up with
            # only one class present, which several sklearn metrics can't
            # compute. Record the failure rather than crashing the whole run.
            results[f"layer_{layer}"] = {"error": str(e)}
            print(f"  could not fit/score probe on this split: {e}")

    scored = {k: v for k, v in results.items() if "auroc" in v}
    if scored:
        ranked = sorted(scored.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
        recommended_layer = ranked[0][0]
        print(f"\nRecommended layer: {recommended_layer} (by validation AUROC)")
    else:
        ranked = []
        recommended_layer = None
        print(
            "\nNo layer produced a valid AUROC on this run -- expected on a "
            "dataset this small. Re-run against a larger collection run."
        )

    with open(config.output_dir / f"{config.run_name}_layer_selection_results.json", "w") as f:
        json.dump(
            {
                "run_name": config.run_name,
                "results_by_layer": results,
                "ranked_layers": [name for name, _ in ranked],
                "recommended_layer": recommended_layer,
            },
            f,
            indent=2,
        )

    print(f"\nAll layer-selection outputs written to {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
