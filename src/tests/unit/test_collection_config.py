from pathlib import Path

import pytest

from mas_sae.experiments.collection_config import (
    load_collection_config,
)


def test_load_collection_config(tmp_path: Path) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: train
  num_questions: 10
collection:
  layers: [8, 17, 25, 33]
  seed: 42
output:
  run_name: train_10q
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_collection_config(path)

    assert config["dataset"]["source_split"] == "train"
    assert config["dataset"]["num_questions"] == 10
    assert config["collection"]["layers"] == [8, 17, 25, 33]
    assert config["collection"]["seed"] == 42


def test_collection_config_rejects_unknown_source_split(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: test
  num_questions: 10
collection:
  layers: [8, 17]
  seed: 42
output:
  run_name: bad_split
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source_split",
    ):
        load_collection_config(path)


def test_collection_config_rejects_non_mapping_section(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
dataset:
  source_split: train
  num_questions: 10
collection:
  layers: [8, 17]
  seed: 42
output:
  run_name: bad_model
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model must be a mapping"):
        load_collection_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_questions", "true", "num_questions"),
        ("seed", "true", "seed"),
    ],
)
def test_collection_config_rejects_boolean_integers(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    num_questions = value if field == "num_questions" else "10"
    seed = value if field == "seed" else "42"

    path = tmp_path / "collection.yaml"
    path.write_text(
        f"""
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: train
  num_questions: {num_questions}
collection:
  layers: [8, 17]
  seed: {seed}
output:
  run_name: validation_test
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_collection_config(path)


def test_collection_config_rejects_duplicate_layers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collection.yaml"
    path.write_text(
        """
model:
  id: google/gemma-3-4b-it
dataset:
  source_split: train
  num_questions: 10
collection:
  layers: [8, 8]
  seed: 42
output:
  run_name: duplicate_layers
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_collection_config(path)
