import argparse
import json
import logging
from pathlib import Path

from mas_sae.experiments.records import read_jsonl
from mas_sae.probe.data import load_acceptance_probe_data
from mas_sae.sae.dataloader import create_sae_dataloader


LAYERS = [8, 17, 25, 33]
HIDDEN_DIM = 2560


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="pilot_12q",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    interactions = read_jsonl(
        Path("logs")
        / "issue14"
        / args.run_name
        / "interactions.jsonl"
    )

    num_episodes = len(interactions)
    num_questions = len({
        str(row["question_id"])
        for row in interactions
    })
    labeled_rows = [
        row
        for row in interactions
        if row["solver_accepted_feedback"] is not None
    ]

    labels = [
        int(row["solver_accepted_feedback"])
        for row in labeled_rows
    ]

    layer_results = {}

    for layer in LAYERS:
        location = (
            Path("data")
            / "activations"
            / args.run_name
            / f"layer_{layer:02d}"
        )

        attempt1_loader = create_sae_dataloader(
            batch_size=8,
            split="pilot_attempt1",
            num_workers=0,
            location=location,
        )
        attempt2_loader = create_sae_dataloader(
            batch_size=8,
            split="pilot_attempt2",
            num_workers=0,
            location=location,
        )

        attempt1_rows = sum(
            batch.shape[0]
            for batch in attempt1_loader
        )
        attempt2_rows = sum(
            batch.shape[0]
            for batch in attempt2_loader
        )

        X, y = load_acceptance_probe_data(
            location,
            interactions,
        )

        if attempt1_rows != num_questions:
            raise ValueError(
                f"Layer {layer}: unexpected Attempt 1 row count."
            )

        if attempt2_rows != num_episodes:
            raise ValueError(
                f"Layer {layer}: unexpected Attempt 2 row count."
            )

        if X.shape != (
            len(labeled_rows),
            HIDDEN_DIM,
        ):
            raise ValueError(
                f"Layer {layer}: unexpected probe-data shape."
            )

        if y.tolist() != labels:
            raise ValueError(
                f"Layer {layer}: probe labels are misaligned."
            )

        layer_results[str(layer)] = {
            "attempt1_rows": attempt1_rows,
            "attempt2_rows": attempt2_rows,
            "probe_rows": len(y),
        }

    report = {
        "status": "pass",
        "questions": num_questions,
        "episodes": num_episodes,
        "accepted": sum(labels),
        "rejected": len(labels) - sum(labels),
        "excluded_noncommittal": (
            num_episodes - len(labels)
        ),
        "layers": layer_results,
    }

    output = (
        Path("results")
        / args.run_name
        / "integration_report.json"
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
        "Pilot integration passed: %s",
        output,
    )


if __name__ == "__main__":
    main()
