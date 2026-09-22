from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.data.musique import load_musique_examples
from mas_sae.experiments.collection import collect_examples
from mas_sae.experiments.collection_artifacts import (
    build_resolved_config,
    ensure_output_available,
    save_collection_artifacts,
)
from mas_sae.experiments.collection_config import load_collection_config
from mas_sae.models.loader import load_gemma


logger = logging.getLogger(__name__)
RESULT_ROOT = Path("results/collection")
ACTIVATION_ROOT = Path("data/activations")


def parse_args() -> argparse.Namespace:
    """Parse activation-collection command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect Solver-Critic activations from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a collection YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one config-driven activation collection."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_collection_config(args.config)
    model_id = config["model"]["id"]
    source_split = config["dataset"]["source_split"]
    num_questions = config["dataset"]["num_questions"]
    layers = config["collection"]["layers"]
    seed = config["collection"]["seed"]
    run_name = config["output"]["run_name"]

    ensure_output_available(RESULT_ROOT, run_name, source_split)
    candidate_sites = [
        f"model.language_model.layers.{layer}" for layer in layers
    ]

    logger.info(
        "Starting run=%s split=%s questions=%d",
        run_name,
        source_split,
        num_questions,
    )
    logger.info("Loading model %s", model_id)

    model, processor = load_gemma(model_id=model_id)
    solver = Solver(model, processor)
    critic = Critic(model, processor)
    validator = Validator(model, processor)

    resolved_config = build_resolved_config(
        config, model, solver, critic, validator
    )
    examples = load_musique_examples(
        source_split=source_split,
        num_questions=num_questions,
    )

    result = collect_examples(
        examples=examples,
        source_split=source_split,
        model=model,
        solver=solver,
        critic=critic,
        validator=validator,
        candidate_sites=candidate_sites,
        base_seed=seed,
    )

    summary = save_collection_artifacts(
        activation_root=ACTIVATION_ROOT,
        result_root=RESULT_ROOT,
        run_name=run_name,
        source_split=source_split,
        candidate_sites=candidate_sites,
        attempt1_by_site=result["attempt1_by_site"],
        attempt2_by_site=result["attempt2_by_site"],
        records=result["records"],
        resolved_config=resolved_config,
    )

    logger.info(
        "Completed run=%s split=%s questions=%d episodes=%d",
        run_name,
        source_split,
        summary["num_questions"],
        summary["num_episodes"],
    )


if __name__ == "__main__":
    main()
