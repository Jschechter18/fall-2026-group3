from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import load_dataset


MUSIQUE_DATASET_ID = "dgslibisey/MuSiQue"
SUPPORTED_SOURCE_SPLITS = {"train", "validation"}


def load_musique_examples(
    source_split: str,
    num_questions: int | None = None,
) -> list[dict[str, Any]]:
    """Load answerable MuSiQue examples from an explicit source split."""
    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            f"Unsupported MuSiQue source split: {source_split!r}. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_SPLITS)}."
        )

    if num_questions is not None and num_questions <= 0:
        raise ValueError("num_questions must be greater than zero.")

    dataset = load_dataset(
        MUSIQUE_DATASET_ID,
        split=source_split,
    )

    examples: list[dict[str, Any]] = []

    for example in dataset:
        if not example.get("answerable", True):
            continue

        examples.append(dict(example))

        if (
            num_questions is not None
            and len(examples) >= num_questions
        ):
            break

    if (
        num_questions is not None
        and len(examples) < num_questions
    ):
        raise RuntimeError(
            f"Requested {num_questions} answerable questions "
            f"from {source_split!r}, but only found {len(examples)}."
        )

    return examples


def download_musique(
    output_dir: Path,
) -> tuple[Path, Path]:
    """Download MuSiQue train and validation splits as JSON files."""
    dataset = load_dataset(MUSIQUE_DATASET_ID)

    train = dataset["train"]
    validation = dataset["validation"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = output_dir / "train.json"
    validation_file = output_dir / "validation.json"

    train.to_json(train_file)
    validation.to_json(validation_file)

    return train_file, validation_file
