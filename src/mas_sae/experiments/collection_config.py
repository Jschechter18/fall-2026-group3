from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import yaml

from mas_sae.data.musique import SUPPORTED_SOURCE_SPLITS


class ModelConfig(TypedDict):
    id: str


class DatasetConfig(TypedDict):
    source_split: str
    num_questions: int


class CollectionSettings(TypedDict):
    layers: list[int]
    seed: int


class OutputConfig(TypedDict):
    run_name: str


class CollectionConfig(TypedDict):
    model: ModelConfig
    dataset: DatasetConfig
    collection: CollectionSettings
    output: OutputConfig


def load_collection_config(path: str | Path) -> CollectionConfig:
    """Load and validate one activation-collection YAML config."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Collection config must be a mapping.")

    required_sections = {"model", "dataset", "collection", "output"}
    missing = required_sections - raw.keys()

    if missing:
        raise ValueError(
            f"Collection config is missing sections: {sorted(missing)}."
        )

    for section in required_sections:
        if not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a mapping.")

    model = raw["model"]
    dataset = raw["dataset"]
    collection = raw["collection"]
    output = raw["output"]

    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model.id must be a non-empty string.")

    source_split = dataset.get("source_split")
    if source_split not in SUPPORTED_SOURCE_SPLITS:
        raise ValueError(
            "dataset.source_split must be one of "
            f"{sorted(SUPPORTED_SOURCE_SPLITS)}."
        )

    num_questions = dataset.get("num_questions")
    if type(num_questions) is not int or num_questions <= 0:
        raise ValueError(
            "dataset.num_questions must be a positive integer."
        )

    layers = collection.get("layers")
    if (
        not isinstance(layers, list)
        or not layers
        or not all(type(layer) is int and layer >= 0 for layer in layers)
    ):
        raise ValueError(
            "collection.layers must be a non-empty list "
            "of non-negative integers."
        )

    if len(set(layers)) != len(layers):
        raise ValueError("collection.layers must not contain duplicates.")

    seed = collection.get("seed")
    if type(seed) is not int:
        raise ValueError("collection.seed must be an integer.")

    run_name = output.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("output.run_name must be a non-empty string.")

    return raw
