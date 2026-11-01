import argparse
import json
import logging
from pathlib import Path

from mas_sae.agents.critic import CriticCondition
from mas_sae.experiments.alignment import (
    validate_activation_alignment,
)
from mas_sae.experiments.records import read_jsonl


LAYERS = [8, 17, 25, 33]
HIDDEN_DIM = 2560
ATTEMPT1_SPLIT = "pilot_attempt1"
ATTEMPT2_SPLIT = "pilot_attempt2"
EXPECTED_CONDITIONS = {
    condition.value
    for condition in CriticCondition
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="pilot_12q",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    interactions = read_jsonl(
        Path("logs")
        / "issue14"
        / args.run_name
        / "interactions.jsonl"
    )

    report = validate_activation_alignment(
        store_root=(
            Path("data")
            / "activations"
            / args.run_name
        ),
        interactions=interactions,
        layers=LAYERS,
        attempt1_split=ATTEMPT1_SPLIT,
        attempt2_split=ATTEMPT2_SPLIT,
        expected_conditions=EXPECTED_CONDITIONS,
        hidden_dim=HIDDEN_DIM,
        expected_num_questions=args.num_questions,
    )

    output = (
        Path("results")
        / "alignment"
        / args.run_name
        / "alignment_report.json"
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    logging.info(
        "Alignment validation passed: %s",
        output,
    )


if __name__ == "__main__":
    main()
