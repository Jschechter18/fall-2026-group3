# src/mas_sae/data/musique.py

from pathlib import Path

from datasets import load_dataset

def download_musique(output_dir: Path) -> tuple[Path, Path]:
    """Helper function to download MuSiQue dataset.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory where the MuSiQue dataset will be saved

    Returns
    -------
    tuple[Path, Path]
        Paths to the train and validation JSON files.
    """
    dataset = load_dataset("dgslibisey/MuSiQue")

    train = dataset["train"]
    validation = dataset["validation"]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.json"
    validation_file = output_dir / "validation.json"


    train.to_json(train_file)
    validation.to_json(validation_file)

    return train_file, validation_file