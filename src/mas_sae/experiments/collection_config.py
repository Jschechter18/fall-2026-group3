from __future__ import annotations

from pathlib import Path
from typing import Any, NotRequired, TypedDict

import yaml

from mas_sae.data.musique import (
    SUPPORTED_SOURCE_SPLITS,
    validate_experiment_split_proportions,
    validate_proportions,
)
from mas_sae.experiments.conditions import (
    OMITTED,
    resolve_conditions,
    resolve_presentations,
)
from mas_sae.experiments.reproducibility import validate_commit_revision
from mas_sae.models.roles import resolve_roles


SUPPORTED_SAMPLING_STRATEGIES = ("first_n", "random", "stratified", "listed")


class ModelConfig(TypedDict):
    id: str


class SamplingConfig(TypedDict):
    """How questions are drawn from the MuSiQue source split.

    ``first_n`` is the original behaviour (dataset order). ``random`` draws
    a seeded uniform sample. ``stratified`` draws a seeded sample with the
    given ``hop_proportions`` over hop groups (``2hop``, ``3hop``, ``4hop``).
    ``listed`` collects the questions in ``question_ids_file`` (a JSON list
    of ids), in that order.
    ``seed`` defaults to ``collection.seed`` when omitted.
    """

    strategy: str
    seed: NotRequired[int]
    hop_proportions: NotRequired[dict[str, float]]
    question_ids_file: NotRequired[str]


class ExperimentSplitConfig(TypedDict):
    """Question-level scientific split, separate from ``source_split``.

    ``proportions`` maps a subset of ``discovery`` / ``validation`` /
    ``intervention`` to fractions summing to one. ``seed`` defaults to
    ``collection.seed`` when omitted.
    """

    proportions: dict[str, float]
    seed: NotRequired[int]


class DatasetConfig(TypedDict):
    source_split: str
    num_questions: int
    revision: NotRequired[str]
    path: NotRequired[str]
    sampling: NotRequired[SamplingConfig]
    experiment_split: NotRequired[ExperimentSplitConfig]


class ManifestDatasetConfig(TypedDict):
    inputs: list[dict[str, Any]]
    intervention: dict[str, Any]
    split: dict[str, Any]


class CollectionSettings(TypedDict):
    """Activation layers, seed, and an optional protocol label.

    ``protocol_version`` is a run label that the collection script maps
    to critic and target behaviour; the loader only checks that it is a
    non-empty string. Omitted means the script's default, so existing
    configs are unchanged.
    """

    layers: list[int]
    seed: int
    protocol_version: NotRequired[str]
    active_conditions: NotRequired[list[str]]
    a2_presentations: NotRequired[list[str]]


class OutputConfig(TypedDict):
    """Run naming and operational settings, excluded from the config hash.

    ``chunk_size`` is how many questions are committed together during a
    resumable collection; it does not affect results.
    """

    run_name: str
    chunk_size: NotRequired[int]


class CollectionConfig(TypedDict):
    model: NotRequired[ModelConfig]
    roles: NotRequired[dict[str, Any]]
    dataset: DatasetConfig | ManifestDatasetConfig
    collection: CollectionSettings
    output: OutputConfig


def load_collection_config(path: str | Path) -> CollectionConfig:
    """Load and validate one activation-collection YAML config."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Collection config must be a mapping.")

    required_sections = {"dataset", "collection", "output"}
    required_sections.add("roles" if "roles" in raw else "model")
    missing = required_sections - raw.keys()

    if missing:
        raise ValueError(
            f"Collection config is missing sections: {sorted(missing)}."
        )

    for section in required_sections:
        if not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a mapping.")

    # Validates the legacy ``model`` section or the explicit ``roles``
    # section (including ``model.id``), and ``active_conditions``.
    resolve_roles(raw)
    resolve_conditions(raw["collection"].get("active_conditions", OMITTED))
    resolve_presentations(raw["collection"].get("a2_presentations", OMITTED))
    dataset = raw["dataset"]
    collection = raw["collection"]
    output = raw["output"]

    if "inputs" in dataset:
        _validate_manifest_dataset(dataset)
    else:
        source_split = dataset.get("source_split")
        if source_split not in SUPPORTED_SOURCE_SPLITS:
            raise ValueError(
                "dataset.source_split must be one of "
                f"{sorted(SUPPORTED_SOURCE_SPLITS)}."
            )

        if "revision" in dataset:
            validate_commit_revision(dataset["revision"], "dataset")

        if "path" in dataset:
            if not isinstance(dataset["path"], str) or not dataset["path"].strip():
                raise ValueError("dataset.path must be a nonempty string.")
            if "revision" in dataset:
                raise ValueError("dataset.path and dataset.revision are mutually exclusive.")
            if not Path(dataset["path"]).is_file():
                raise ValueError(f"dataset.path does not exist: {dataset['path']}")

        num_questions = dataset.get("num_questions")
        if type(num_questions) is not int or num_questions <= 0:
            raise ValueError(
                "dataset.num_questions must be a positive integer."
            )

        if "sampling" in dataset:
            _validate_sampling(dataset["sampling"])

        if "experiment_split" in dataset:
            _validate_experiment_split(dataset["experiment_split"])

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

    if "protocol_version" in collection:
        protocol_version = collection["protocol_version"]
        if (
            not isinstance(protocol_version, str)
            or not protocol_version.strip()
        ):
            raise ValueError(
                "collection.protocol_version must be a non-empty string "
                "when provided."
            )

    run_name = output.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("output.run_name must be a non-empty string.")

    if "chunk_size" in output and (
        type(output["chunk_size"]) is not int or output["chunk_size"] <= 0
    ):
        raise ValueError("output.chunk_size must be a positive integer when provided.")

    return raw


def _validate_optional_seed(section: dict[str, Any], name: str) -> None:
    if "seed" in section and type(section["seed"]) is not int:
        raise ValueError(f"{name}.seed must be an integer when provided.")


def _validate_sampling(sampling: Any) -> None:
    """Validate the optional ``dataset.sampling`` section."""
    if not isinstance(sampling, dict):
        raise ValueError("dataset.sampling must be a mapping.")

    strategy = sampling.get("strategy")
    if strategy not in SUPPORTED_SAMPLING_STRATEGIES:
        raise ValueError(
            "dataset.sampling.strategy must be one of "
            f"{list(SUPPORTED_SAMPLING_STRATEGIES)}."
        )

    _validate_optional_seed(sampling, "dataset.sampling")

    has_proportions = "hop_proportions" in sampling

    if strategy == "stratified" and not has_proportions:
        raise ValueError(
            "dataset.sampling.hop_proportions is required when "
            "strategy is 'stratified'."
        )

    if strategy != "stratified" and has_proportions:
        raise ValueError(
            "dataset.sampling.hop_proportions is only allowed when "
            "strategy is 'stratified'."
        )

    has_ids_file = "question_ids_file" in sampling

    if strategy == "listed" and (
        not isinstance(sampling.get("question_ids_file"), str)
        or not sampling["question_ids_file"].strip()
    ):
        raise ValueError(
            "dataset.sampling.question_ids_file is required when "
            "strategy is 'listed'."
        )

    if strategy != "listed" and has_ids_file:
        raise ValueError(
            "dataset.sampling.question_ids_file is only allowed when "
            "strategy is 'listed'."
        )

    if has_proportions:
        proportions = sampling["hop_proportions"]
        validate_proportions(
            proportions, name="dataset.sampling.hop_proportions"
        )

        for group in proportions:
            if not group.endswith("hop") or not group[:-3].isdigit():
                raise ValueError(
                    "dataset.sampling.hop_proportions keys must be hop "
                    f"groups such as '2hop', got {group!r}."
                )


def _validate_experiment_split(experiment_split: Any) -> None:
    """Validate the optional ``dataset.experiment_split`` section."""
    if not isinstance(experiment_split, dict):
        raise ValueError("dataset.experiment_split must be a mapping.")

    if "proportions" not in experiment_split:
        raise ValueError(
            "dataset.experiment_split.proportions is required."
        )

    validate_experiment_split_proportions(
        experiment_split["proportions"],
        name="dataset.experiment_split.proportions",
    )

    _validate_optional_seed(experiment_split, "dataset.experiment_split")


def _validate_manifest_dataset(dataset):
    """Manifest inputs replace MuSiQue sampling; research splits are assigned later."""
    if set(dataset) & {"source_split", "num_questions", "sampling", "experiment_split", "revision"}:
        raise ValueError("dataset.inputs cannot be combined with legacy MuSiQue sampling fields")
    if not isinstance(dataset["inputs"], list) or not dataset["inputs"]:
        raise ValueError("dataset.inputs must be a nonempty list")
    reservation = dataset.get("intervention")
    if not isinstance(reservation, dict) or (("fraction" in reservation) == ("question_ids_file" in reservation)):
        raise ValueError("dataset.intervention requires either fraction or question_ids_file")
    if "fraction" in reservation:
        fraction = reservation["fraction"]
        if type(fraction) not in (int, float) or not 0 <= fraction < 1:
            raise ValueError("intervention fraction must be in [0, 1)")
    elif not isinstance(reservation["question_ids_file"], str) or not reservation["question_ids_file"]:
        raise ValueError("intervention question_ids_file must be a path")
    _validate_optional_seed(reservation, "dataset.intervention")
    split = dataset.get("split")
    if not isinstance(split, dict) or set(split.get("proportions", {})) != {"train", "validation", "test"}:
        raise ValueError("dataset.split.proportions must name train, validation, test")
    validate_experiment_split_proportions(split["proportions"], supported_splits=("train", "validation", "test"))
    _validate_optional_seed(split, "dataset.split")
