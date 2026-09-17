from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

from mas_sae.agents.critic import Critic
from mas_sae.agents.solver import Solver
from mas_sae.agents.validator import Validator
from mas_sae.data.activation_store import ActivationStore
from mas_sae.experiments.pipeline import run_question
from mas_sae.experiments.records import write_jsonl
from mas_sae.models.loader import load_gemma


MODEL_ID = "google/gemma-3-4b-it"

# Candidate layers for technical feasibility only.
# We are not claiming a final scientific layer yet.
CANDIDATE_LAYERS = (
    8,
    17,
    25,
    33,
)

CANDIDATE_SITES = [
    f"model.language_model.layers.{layer}"
    for layer in CANDIDATE_LAYERS
]


def get_examples(
    num_questions: int,
) -> list[dict]:
    dataset = load_dataset(
        "dgslibisey/MuSiQue",
        split="validation",
    )

    examples: list[dict] = []

    for example in dataset:
        if not example.get(
            "answerable",
            True,
        ):
            continue

        examples.append(example)

        if len(examples) >= num_questions:
            break

    if len(examples) < num_questions:
        raise RuntimeError(
            f"Requested {num_questions} questions "
            f"but only found {len(examples)} usable examples."
        )

    return examples


def site_slug(
    module_name: str,
) -> str:
    layer = module_name.rsplit(".", 1)[-1]

    return f"layer_{int(layer):02d}"


def save_and_verify(
    *,
    store: ActivationStore,
    split: str,
    tensor: torch.Tensor,
) -> None:
    """Save a tensor then immediately verify the round trip."""

    store.save_activations(
        split,
        tensor,
    )

    loaded = store.load_activations(
        split,
    )

    if not torch.equal(
        tensor.cpu(),
        loaded,
    ):
        raise RuntimeError(
            f"Activation round trip failed for {split}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-questions",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    args = parser.parse_args()

    activation_root = (
        Path("data")
        / "activations"
        / args.run_name
    )

    log_root = (
        Path("logs")
        / "issue14"
        / args.run_name
    )

    activation_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Gemma...")
    model, processor = load_gemma()

    solver = Solver(
        model,
        processor,
    )

    critic = Critic(
        model,
        processor,
    )

    validator = Validator(
        model,
        processor,
    )

    examples = get_examples(
        args.num_questions
    )

    attempt1_by_site = {
        site: []
        for site in CANDIDATE_SITES
    }

    attempt2_by_site = {
        site: []
        for site in CANDIDATE_SITES
    }

    records: list[dict] = []

    # =========================================================
    # Run questions.
    # =========================================================

    for question_index, example in enumerate(
        examples
    ):
        seed = (
            args.seed
            + question_index
        )

        print()
        print(
            f"QUESTION {question_index + 1}/"
            f"{args.num_questions}"
        )
        print("ID:", example["id"])
        print("Question:", example["question"])
        print("Gold:", example["answer"])

        result = run_question(
            question_id=str(example["id"]),
            question=example["question"],
            paragraphs=example["paragraphs"],
            gold=example["answer"],
            aliases=(
                example.get(
                    "answer_aliases",
                    [],
                )
                or []
            ),
            model=model,
            solver=solver,
            critic=critic,
            validator=validator,
            candidate_sites=CANDIDATE_SITES,
            seed=seed,
        )

        # -----------------------------------------------------
        # Attempt 1 is saved ONCE per question.
        # -----------------------------------------------------

        attempt1_index = len(
            attempt1_by_site[
                CANDIDATE_SITES[0]
            ]
        )

        for site in CANDIDATE_SITES:
            attempt1_by_site[
                site
            ].append(
                result[
                    "attempt1_activations"
                ][site]
            )

        print(
            "Attempt 1:",
            result["attempt1"],
        )

        # -----------------------------------------------------
        # Three Attempt-2 episodes.
        # -----------------------------------------------------

        for episode in result["episodes"]:
            attempt2_index = len(
                attempt2_by_site[
                    CANDIDATE_SITES[0]
                ]
            )

            for site in CANDIDATE_SITES:
                attempt2_by_site[
                    site
                ].append(
                    episode[
                        "attempt2_activations"
                    ][site]
                )

            record = dict(
                episode["record"]
            )

            record[
                "attempt1_activation_index"
            ] = attempt1_index

            record[
                "attempt2_activation_index"
            ] = attempt2_index

            records.append(record)

            print(
                " ",
                record["critic_condition"],
                "->",
                record["solver_attempt_2"],
                "| verdict:",
                record["critic_verdict"],
                "| accepted:",
                record[
                    "solver_accepted_feedback"
                ],
                "| correct:",
                record[
                    "solver_attempt_2_correct"
                ],
                "| validator:",
                record[
                    "validator_final_correct"
                ],
            )

    # =========================================================
    # Save activation matrices.
    # =========================================================

    activation_shapes = {}

    for site in CANDIDATE_SITES:
        slug = site_slug(site)

        attempt1_tensor = torch.cat(
            attempt1_by_site[site],
            dim=0,
        )

        attempt2_tensor = torch.cat(
            attempt2_by_site[site],
            dim=0,
        )

        expected_a1 = (
            args.num_questions,
            2560,
        )

        expected_a2 = (
            args.num_questions * 3,
            2560,
        )

        if tuple(
            attempt1_tensor.shape
        ) != expected_a1:
            raise RuntimeError(
                f"{slug}: Attempt1 was "
                f"{tuple(attempt1_tensor.shape)}, "
                f"expected {expected_a1}."
            )

        if tuple(
            attempt2_tensor.shape
        ) != expected_a2:
            raise RuntimeError(
                f"{slug}: Attempt2 was "
                f"{tuple(attempt2_tensor.shape)}, "
                f"expected {expected_a2}."
            )

        local_store = ActivationStore(
            str(
                activation_root
                / slug
            )
        )

        save_and_verify(
            store=local_store,
            split="pilot_attempt1",
            tensor=attempt1_tensor,
        )

        save_and_verify(
            store=local_store,
            split="pilot_attempt2",
            tensor=attempt2_tensor,
        )

        activation_shapes[slug] = {
            "attempt1": list(
                attempt1_tensor.shape
            ),
            "attempt2": list(
                attempt2_tensor.shape
            ),
        }

        print(
            slug,
            "Attempt1:",
            tuple(
                attempt1_tensor.shape
            ),
            "Attempt2:",
            tuple(
                attempt2_tensor.shape
            ),
        )

    # =========================================================
    # Save minimal behavioral artifacts.
    # =========================================================

    write_jsonl(
        log_root
        / "interactions.jsonl",
        records,
    )

    accepted = sum(
        row[
            "solver_accepted_feedback"
        ]
        is True
        for row in records
    )

    rejected = sum(
        row[
            "solver_accepted_feedback"
        ]
        is False
        for row in records
    )

    noncommittal = sum(
        row[
            "solver_accepted_feedback"
        ]
        is None
        for row in records
    )

    summary = {
        "model_id": MODEL_ID,
        "seed": args.seed,
        "num_questions": (
            args.num_questions
        ),
        "num_episodes": len(records),
        "candidate_sites": (
            CANDIDATE_SITES
        ),
        "activation_shapes": (
            activation_shapes
        ),
        "accepted": accepted,
        "rejected": rejected,
        "unlabeled_noncommittal": (
            noncommittal
        ),
    }

    with (
        log_root
        / "pilot_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
