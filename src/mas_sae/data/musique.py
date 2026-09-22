# src/mas_sae/data/musique.py

from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

from datasets import load_dataset


MUSIQUE_DATASET_ID = "dgslibisey/MuSiQue"
SUPPORTED_SOURCE_SPLITS = {"train", "validation"}


class MuSiQueExample(TypedDict):
    """MuSiQue fields required by activation collection."""

    id: str
    question: str
    paragraphs: list[dict[str, Any]]
    answer: str
    answer_aliases: NotRequired[list[str]]
    answerable: NotRequired[bool]


def download_musique(output_dir: Path) -> tuple[Path, Path]:
    """Helper function to download MuSiQue dataset. This dataset is currently a community maintained one from huggingface. We will use this for the smoke test, but load
    from the actual source after early development.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory where the MuSiQue dataset will be saved

    Returns
    -------
    tuple[Path, Path]
        Paths to the train and validation JSON files.
    """
    dataset = load_dataset(MUSIQUE_DATASET_ID)

    train = dataset["train"]
    validation = dataset["validation"]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.json"
    validation_file = output_dir / "validation.json"


    train.to_json(train_file)
    validation.to_json(validation_file)

    return train_file, validation_file


def load_musique_examples(
    source_split: str,
    num_questions: int | None = None,
) -> list[MuSiQueExample]:
    """Load answerable MuSiQue examples from one source split in dataset order."""
    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            f"Unsupported MuSiQue source split: {source_split!r}. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_SPLITS)}."
        )

    if num_questions is not None and num_questions <= 0:
        raise ValueError("num_questions must be greater than zero.")

    dataset = load_dataset(MUSIQUE_DATASET_ID, split=source_split)
    examples: list[MuSiQueExample] = []

    for example in dataset:
        if not example.get("answerable", True):
            continue

        examples.append(cast(MuSiQueExample, dict(example)))

        if num_questions is not None and len(examples) >= num_questions:
            break

    if num_questions is not None and len(examples) < num_questions:
        raise RuntimeError(
            f"Requested {num_questions} answerable questions from "
            f"{source_split!r}, but only found {len(examples)}."
        )

    return examples
