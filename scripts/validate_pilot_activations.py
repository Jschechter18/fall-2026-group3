import argparse
import json
import logging
from pathlib import Path

from mas_sae.experiments.alignment import (
    validate_pilot_alignment,
)
from mas_sae.experiments.records import read_jsonl


LAYERS = [8, 17, 25, 33]


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

    logging.basicConfig(
        level=logging.INFO,
    )

    interactions = read_jsonl(
        Path("logs")
        / "issue14"
        / args.run_name
        / "interactions.jsonl"
    )

    report = validate_pilot_alignment(
        Path("data")
        / "activations"
        / args.run_name,
        interactions,
        LAYERS,
        args.num_questions,
    )

    output = (
        Path("results")
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
