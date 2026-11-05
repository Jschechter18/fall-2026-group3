"""Every check a stage needs before it spends GPU time, reported together in one error.

Which checks run depends on the stage: GPU stages (choose-r, prepare, run) check the solver weights,
CUDA, GPU memory and disk; labeling stages (choose-r, merge) check the labeler and AWS credentials.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from mas_sae.experiments.artifacts import sha256_file
from mas_sae.intervention.analysis import labeling
from mas_sae.intervention.run_files import layout
from mas_sae.intervention.experiment.representations import checkpoint_sha256, feature_set_problems, load_basis, read_feature_sets

GPU_STAGES = ("choose-r", "prepare", "run")
LABELING_STAGES = ("choose-r", "merge")


class PreflightError(RuntimeError):
    pass


def input_file_problems(config) -> list[str]:
    i = config.inputs
    files = [i.interactions, i.contexts, i.feature_sets,
             *(str(Path(i.stored_labels) / f) for f in ("run_settings.json", "summary.json", "raw_results.jsonl"))]
    problems = [f"missing input file {f}" for f in files if not Path(f).is_file()]
    if not problems:
        settings = json.loads((Path(i.stored_labels) / "run_settings.json").read_text())
        if settings.get("episodes_sha256") != sha256_file(i.interactions):
            problems.append(f"{i.stored_labels} labeled a different interactions file than {i.interactions}")
        if settings.get("partition") != config.partition:
            problems.append(f"stored labels are for partition {settings.get('partition')!r}, not {config.partition!r}")
    return problems


def representation_problems(config) -> list[str]:
    """Checkpoints present, with the sha256 feature_sets.json recorded, and valid feature sets."""
    if not Path(config.inputs.feature_sets).is_file():
        return []                                   # reported with the input files
    entries, problems = read_feature_sets(Path(config.inputs.feature_sets)), []
    for name, spec in config.representations.items():
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name}: not in {config.inputs.feature_sets}")
            continue
        if entry["path"] != spec.path:
            problems.append(f"{name}: config path {spec.path} is not the one feature selection used ({entry['path']})")
            continue
        missing = [p for p in entry["sha256"] if not Path(p).is_file()]
        if missing:
            problems.append(f"{name}: missing checkpoint files {missing}")
            continue
        if checkpoint_sha256(spec) != entry["sha256"]:
            problems.append(f"{name}: checkpoint files differ from the ones feature selection used")
            continue
        _, width = load_basis(spec, config.layer)
        problems += feature_set_problems(name, entry, width, config.experiment.number_of_targeted_features,
                                         config.experiment.random_draw_pool_size)
    return problems


def solver_weight_problems(config) -> list[str]:
    """The pinned revision of the Solver must already be in the local cache: no download mid-run."""
    spec = config.solver
    cache_dir = Path(spec.get("cache_dir", "checkpoints/huggingface"))
    snapshot = cache_dir / f"models--{spec['id'].replace('/', '--')}" / "snapshots" / spec["revision"]
    if not (snapshot / "config.json").is_file() or not any(snapshot.glob("*.safetensors")):
        return [f"solver weights for {spec['id']} at revision {spec['revision']} are not cached in {snapshot}"]
    return []


def machine_problems(config) -> list[str]:
    import torch
    if not torch.cuda.is_available():
        return ["CUDA is not available"]
    problems = []
    device = torch.device(config.solver.get("device", "cuda:0"))
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if total_gb < config.machine.min_gpu_memory_gb:
        problems.append(f"GPU has {total_gb:.1f} GB, at least {config.machine.min_gpu_memory_gb} GB is needed")
    return problems


def output_problems(config) -> list[str]:
    out = Path(config.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out):
            pass
    except OSError as e:
        return [f"output directory {out} is not writable ({e})"]
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < config.machine.min_free_disk_gb:
        return [f"{free_gb:.1f} GB free in {out}, at least {config.machine.min_free_disk_gb} GB is needed"]
    return []


def labeler_problems(config) -> list[str]:
    labels_dir = Path(config.inputs.stored_labels)
    if not (labels_dir / "run_settings.json").is_file() or not (labels_dir / "summary.json").is_file():
        return []                                   # reported with the input files
    return labeling.protocol_problems(labeling.read_protocol(labels_dir))


def aws_credential_problems(config) -> list[str]:
    try:
        import boto3
        session = boto3.Session(profile_name=config.labeling.aws_profile)
        session.client("sts", region_name=config.labeling.region).get_caller_identity()
    except Exception as e:  # noqa: BLE001 - any failure here means labeling cannot run
        return [f"AWS credentials for profile {config.labeling.aws_profile!r} do not work "
                f"(run `aws sso login --profile {config.labeling.aws_profile}`): {e}"]
    return []


def stage_problems(config, stage: str, shard: int | None) -> list[str]:
    """What this stage needs from earlier stages."""
    run_dir, problems = Path(config.output_dir), []
    if stage == "prepare":
        if config.experiment.edit_size is None:
            problems.append("experiment.edit_size is not set: run choose-r, choose r, and write it into the config")
        if not layout.choose_r_report_csv(run_dir).is_file():
            problems.append(f"no choose-r report at {layout.choose_r_report_csv(run_dir)}: run choose-r first")
    if stage in ("run", "merge"):
        for path in (layout.frozen_run(run_dir), layout.shards(run_dir)):
            if not path.is_file():
                problems.append(f"missing {path}: run prepare first")
    if stage == "run" and layout.shards(run_dir).is_file():
        n = len(json.loads(layout.shards(run_dir).read_text())["shards"])
        if shard is None or not 0 <= shard < n:
            problems.append(f"--shard must be one of 0..{n - 1}")
    return problems


def run_preflight(config, stage: str, shard: int | None = None, labels_stubbed: bool = False) -> None:
    """Raise one PreflightError listing everything this stage is missing.

    ``labels_stubbed`` (smoke runs only) skips the AWS check: no request is sent."""
    problems = input_file_problems(config) + representation_problems(config) + output_problems(config)
    problems += stage_problems(config, stage, shard)
    if stage in GPU_STAGES:
        problems += solver_weight_problems(config) + machine_problems(config)
    if stage in LABELING_STAGES:
        problems += labeler_problems(config) + ([] if labels_stubbed else aws_credential_problems(config))
    if problems:
        raise PreflightError(f"Preflight for {stage!r} failed ({len(problems)} problems):\n  - "
                             + "\n  - ".join(problems))
