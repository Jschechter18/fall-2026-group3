"""Train and evaluate one SAE using the standard run artifact layout."""

from dataclasses import asdict, dataclass, fields, replace
import json
import shutil
from pathlib import Path

import torch

from mas_sae.experiments.artifacts import (
    append_sae_version,
    create_sae_run_directory,
    get_git_provenance,
    build_run_id,
    get_git_commit,
    get_run_command,
    sha256_file,
    utc_now,
    update_run_manifest,
    write_run_config,
    write_sae_provenance,
    write_run_history,
    write_json_atomic,
)
from mas_sae.experiments.reproducibility import seed_everything
from mas_sae.sae.hyperparamters import Hyperparameters as HP, validate_hyperparameters
from mas_sae.sae.training_state import save_training_state, restore_training_state
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE
from mas_sae.sae.dataloader import ActivationDataset, create_sae_dataloader
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.callbacks.checkpointing import CheckpointEvaluatorCallback, load_sae_checkpoint
from mas_sae.sae.callbacks.early_stopping import EarlyStoppingCallback


@dataclass
class TrainingResult:
    """Artifacts and frozen-checkpoint metrics from a completed training run."""

    run_directory: Path
    best_epoch: int
    metrics: dict[str, dict[str, float | None]]

    @property
    def checkpoint_path(self) -> Path:
        return self.run_directory / "checkpoints" / "best_checkpoint.pt"


def train_sae(
    hp: HP,
    activation_location: Path,
    activation_run_name: str,
    layer: int,
    results_root: Path,
    project_root: Path,
    *,
    evaluate_test: bool = True,
    register_version: bool = True,
    resume_directory: Path | None = None,
    run_context: dict | None = None,
    output_directory: Path | None = None,
) -> TrainingResult:
    """Train one SAE, evaluate its best checkpoint, and write run artifacts.

    The supplied hyperparameters are copied before inferring the input dimension.
    Test evaluation is performed only when enabled and test.pt exists.
    Validation-only trials save epoch-boundary state for later continuation.
    
    Parameters:
    ----------
    hp : HP
        The hyperparameters for the SAE training.
    activation_location : Path
        The location of the activation files.
    activation_run_name : str
        The name of the activation run.
    layer : int
        The layer of the SAE being trained.
    results_root : Path
        The root directory for storing results.
    project_root : Path
        The root directory of the project.
    evaluate_test : bool, optional
        Whether to evaluate the test set. Defaults to True.
    register_version : bool, optional
        Whether to register the version. Defaults to True.
    resume_directory : Path | None, optional
        The directory to resume training from. Defaults to None.
    run_context : dict | None, optional
        The context of the run. Defaults to None.
    output_directory : Path | None, optional
        The output directory for the run. Defaults to None.
    
    Returns:
    ----------
    TrainingResult
        The result of the training run, including the run directory, the best epoch, and the evaluation metrics.
    """
    hp = replace(hp)
    validate_hyperparameters(hp)
    seed_everything(hp.seed)

    git_provenance = get_git_provenance()
    state = None
    if resume_directory is None:
        run_directory = create_sae_run_directory(
            run_name=f"sae-l{hp.latent_dim}",
            layer=layer,
            results_root=results_root,
            subdirectories=("checkpoints",),
            output_directory=output_directory,
        )
        manifest = json.loads((run_directory / "manifest.json").read_text())
        if run_context is not None:
            manifest["tuning"] = run_context
            write_json_atomic(run_directory / "manifest.json", manifest)
    else:
        run_directory = Path(resume_directory)
        saved_config = json.loads((run_directory / "config.json").read_text())
        manifest = json.loads((run_directory / "manifest.json").read_text())
        if manifest.get("finalized") or saved_config.get("evaluate_test", True):
            raise ValueError("Only unfinalized, validation-only trials can be continued.")
        for name, value in asdict(hp).items():
            if name not in {"epochs", "input_dim"} and value != saved_config[name]:
                raise ValueError(f"Cannot change {name} when continuing a trial.")
        if layer != saved_config["layer"] or activation_run_name != saved_config["activation_run_name"]:
            raise ValueError("Cannot change the activation source or layer on continuation.")
        for split in ("train", "validation"):
            if sha256_file(activation_location / f"{split}.pt") != manifest["activation_files"][split]["sha256"]:
                raise ValueError(f"The {split} activations changed since this trial was trained.")
        state = torch.load(run_directory / "checkpoints" / "last_checkpoint.pt",
                           map_location="cpu", weights_only=True)
        if hp.epochs < len(state["history"]):
            raise ValueError("epochs is a total budget and cannot be below completed epochs.")
        run_context = saved_config.get("tuning")
        events = manifest.setdefault("continuations", [])
        events.append({"started_at": utc_now().isoformat(), "epochs": hp.epochs,
                       "command": get_run_command(), **git_provenance})
        write_json_atomic(run_directory / "manifest.json", manifest)
        update_run_manifest(run_directory, status="running")

    try:
        activation_files = {
            split: activation_location / f"{split}.pt"
            for split in ("train", "validation")
        }
        if evaluate_test and (activation_location / "test.pt").is_file():
            activation_files["test"] = activation_location / "test.pt"
        if state is None:
            write_sae_provenance(run_directory, activation_files, git_provenance)
        elif "test" in activation_files:
            manifest["activation_files"]["test"] = {
                "path": str(activation_files["test"].resolve()),
                "sha256": sha256_file(activation_files["test"]),
            }
            write_json_atomic(run_directory / "manifest.json", manifest)
            update_run_manifest(run_directory, status="running")

        train_dataloader = create_sae_dataloader(
            hp.batch_size,
            split="train",
            num_workers=2,
            location=activation_location,
            shuffle=hp.train_shuffle,
        )
        val_dataloader = create_sae_dataloader(
            hp.batch_size,
            split="validation",
            num_workers=2,
            location=activation_location,
        )

        test_dataloader = None
        if "test" in activation_files:
            test_dataloader = create_sae_dataloader(
                hp.batch_size,
                split="test",
                num_workers=2,
                location=activation_location,
            )

        hp.input_dim = int(next(iter(train_dataloader)).shape[-1])

        config = {
            **asdict(hp),
            "layer": layer,
            "activation_run_name": activation_run_name,
            "evaluate_test": evaluate_test,
            "register_version": register_version,
        }

        if run_context is not None:
            config["tuning"] = run_context

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        training_remaining = state is None or (
            not state["stopped"] and len(state["history"]) < hp.epochs
        )
        if state is not None and training_remaining and device.type != state["device_type"]:
            raise ValueError("Continue on the same device type to restore RNG state.")

        model = SAE(
            input_dim=hp.input_dim,
            hidden_dim=hp.hidden_dim,
            latent_dim=hp.latent_dim,
            sparsity_mode=hp.sparsity_mode,
            top_k=hp.top_k,
        ).to(device)

        assert isinstance(train_dataloader.dataset, ActivationDataset)

        train_activations = train_dataloader.dataset.activations
        activation_rms_scale = train_activations.float().square().mean().sqrt()

        if not torch.isfinite(activation_rms_scale) or activation_rms_scale <= 0:
            raise ValueError("Training activations must have a finite, positive RMS.")

        with torch.no_grad():
            model.input_scale.copy_(activation_rms_scale)

        config.update({
            "activation_rms_scale": activation_rms_scale.item(),
            "input_normalization": "Divide by one RMS over all training activation elements; reuse for every split.",
            "decoder_output": "Multiply reconstruction by the training RMS to restore original units.",
            "decoder_normalization": "Unit L2 norm per column at initialization and after each optimizer step.",
            "reconstruction_loss": "Mean squared error in RMS-normalized units over examples and input dimensions.",
            "sparsity_loss": (
                "Mean absolute latent activation over examples and features."
                if hp.sparsity_mode == "l1" else
                "Not applied; weighted_sparsity_loss is recorded as zero for compatibility."
            ),
            "total_loss": (
                "reconstruction_loss + sparsity_coefficient * sparsity_loss"
                if hp.sparsity_mode == "l1" else "reconstruction_loss"
            ),
            "selection_metric": "val_loss",
            "variance_explained": "1 - reconstruction SSE / SSE around the training-mean activation; null if baseline SSE is zero.",
            "persistent_inactivity": "No positive activation in the last inactivity_window_examples training examples; null until that many examples have been observed. Evaluations do not update counters.",
            "firing_frequencies": "Fraction of examples with positive activation per feature; percentiles include unused features.",
            "sparsity_mechanism": (
                "L1 activation penalty" if hp.sparsity_mode == "l1" else
                "Per-example TopK of nonnegative encoder activations."
            ),
        })
        # Retain validation-only identity until successful finalization, so a
        # failed continuation remains retryable.
        if state is not None:
            config["evaluate_test"] = False
        write_run_config(run_directory, config)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=hp.lr_patience,
            factor=0.1,
        )

        runner = ModelRunner(
            model, sparsity_coefficient=hp.sparsity_coefficient, optimizer=optimizer,
            training_mean=train_activations.float().mean(dim=0),
            inactivity_window_examples=hp.inactivity_window_examples,
        )

        checkpoint_evaluator = CheckpointEvaluatorCallback(run_directory / "checkpoints")
        early_stopping = EarlyStoppingCallback(hp.patience)

        epoch_history = []
        stopped = False
        if state is not None:
            # Evaluation-only finalization loads the best checkpoint below;
            # optimizer and device-specific RNG state are needed only to train.
            if training_remaining:
                restore_training_state(state, runner, scheduler, checkpoint_evaluator, early_stopping)
            epoch_history = state["history"]
            stopped = state["stopped"]
        stop_reason = "early_stopping" if stopped else "max_epochs"
        for epoch in range(len(epoch_history), hp.epochs):
            if stopped:
                break
            train_metrics = runner.train_epoch(train_dataloader)
            val_metrics = runner.val_epoch(val_dataloader)
            print(
                f"Epoch {epoch + 1}/{hp.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )

            if not all(torch.isfinite(torch.tensor(metrics["loss"]))
                       for metrics in (train_metrics, val_metrics)):
                raise ValueError("Training produced a nonfinite loss.")
            scheduler.step(val_metrics["loss"])
            checkpoint_evaluator.on_validation_end(
                train_metrics, val_metrics, epoch, model, optimizer, scheduler
            )
            epoch_metrics = {
                "epoch": epoch + 1,
                # After scheduler.step(); used for the next epoch.
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
                epoch_metrics.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            epoch_history.append(epoch_metrics)

            write_run_history(run_directory, epoch_history)

            stopped = early_stopping.on_validation_end(val_metrics["loss"])
            if not register_version or state is not None:
                save_training_state(
                    run_directory / "checkpoints" / "last_checkpoint.pt",
                    runner, scheduler, checkpoint_evaluator, early_stopping,
                    epoch_history, stopped,
                )
            if stopped:
                stop_reason = "early_stopping"
                print(f"Early stopping after epoch {epoch+1}")
                break

        model = load_sae_checkpoint(
            run_directory / "checkpoints" / "best_checkpoint.pt", device=device,
        )
        runner.model = model
        # Use identical frozen weights for every split; no training-state updates.
        frozen_metrics = {}
        feature_frequencies = {}
        evaluation_loaders = {"train": train_dataloader, "validation": val_dataloader}
        if test_dataloader is not None:
            evaluation_loaders["test"] = test_dataloader
        for split, dataloader in evaluation_loaders.items():
            frozen_metrics[split] = runner.val_epoch(dataloader)
            frequencies = runner.last_feature_frequencies
            if frequencies is None:
                raise RuntimeError(f"Evaluation did not produce feature frequencies for {split}.")
            feature_frequencies[split] = frequencies.tolist()
        test_metrics = frozen_metrics.get("test")
        if test_metrics is not None:
            print(f"Test Loss: {test_metrics['loss']:.4f}")

        best_epoch_metrics = min(epoch_history, key=lambda metrics: metrics["val_loss"])

        summary = {
            "sparsity_mode": hp.sparsity_mode,
            "top_k": hp.top_k,
            "total_loss": config["total_loss"],
            "epochs_completed": len(epoch_history),
            "stop_reason": stop_reason,
            "best_epoch": best_epoch_metrics["epoch"],
            "frozen_checkpoint_metrics": frozen_metrics,
            "feature_frequencies_path": "feature_frequencies.json",
            "best_val_metrics": {
                key.removeprefix("val_"): value
                for key, value in best_epoch_metrics.items() if key.startswith("val_")
            },
        }

        write_json_atomic(run_directory / "feature_frequencies.json", {
            "checkpoint": "checkpoints/best_checkpoint.pt",
            "epoch": best_epoch_metrics["epoch"],
            "definition": "Fraction of split examples with a positive activation; array index is latent feature index.",
            "frequencies": feature_frequencies,
        })

        write_run_history(
            run_directory,
            epoch_history,
            test_loss=test_metrics["loss"] if test_metrics is not None else None,
            summary=summary,
        )

    except (Exception, KeyboardInterrupt) as error:
        update_run_manifest(
            run_directory,
            status="failed",
            error_message=f"{type(error).__name__}: {error}",
        )
        raise

    update_run_manifest(run_directory, status="completed")
    result = TrainingResult(
        run_directory=run_directory,
        best_epoch=best_epoch_metrics["epoch"],
        metrics=frozen_metrics,
    )
    if register_version:
        _register_run(result, project_root)
    return result


def _register_run(result: TrainingResult, project_root: Path) -> None:
    """Register a completed model using the standard artifacts and CSV."""
    run = result.run_directory
    history = json.loads((run / "history.json").read_text())
    append_sae_version(
        run, project_root=project_root,
        best_val_score=history["summary"]["best_val_metrics"]["loss"],
        test_score=history["test_loss"],
    )
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["finalized"] = True
    write_json_atomic(run / "manifest.json", manifest)
    config = json.loads((run / "config.json").read_text())
    config.update(evaluate_test="test" in result.metrics, register_version=True)
    write_run_config(run, config)


def finish_run(run_directory: Path, epochs: int, project_root: Path) -> TrainingResult:
    """Continue a chosen trial and publish its completed artifacts under runs/.
    
    Tuning trials remain untouched. Continuation works on a copy inside the
    study, which is moved into runs/ only after training and evaluation succeed.
    Legacy validation-only runs without a study continue in place.
    
    Parameters
    ----------
    run_directory : Path
        The directory of the run to be finished.
    epochs : int
        The number of epochs to continue training for.
    project_root : Path
        The root directory of the project.
    
    Returns
    -------
    TrainingResult
        The result of the finished training run.
    """
    run_directory = run_directory.resolve()
    project_root = project_root.resolve()
    source = run_directory
    config = json.loads((source / "config.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("finalized"):
        raise ValueError("Only unfinalized trials can be continued.")
    hp = HP(**{field.name: config[field.name] for field in fields(HP)})
    hp.epochs = epochs
    validate_hyperparameters(hp)
    record_path: Path | None = None
    if "tuning" in config:
        study = project_root / Path(config["tuning"]["study_directory"])
        record_path = Path(study, "finalizations", f"{source.name}.json")
        run_directory = study / "finishing" / source.name
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["status"] == "completed":
                raise ValueError("This trial has already been published; see its finalization record.")
            # Retry publication without training or evaluating the test set again.
            return _publish_trial(run_directory, record_path, record, project_root)
        if not run_directory.exists():
            # Preserve the search-stage weights, metrics, and configuration.
            temporary = run_directory.with_name(run_directory.name + ".tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(source, temporary)
            temporary.rename(run_directory)

    activation_location = (project_root / "data" / "activations"
                           / config["activation_run_name"] / f"layer_{config['layer']:02d}")
    results_root = project_root / "results" / "sae" / "musique"
    result = train_sae(
        hp=hp,
        activation_location=activation_location,
        activation_run_name=config["activation_run_name"],
        layer=config["layer"],
        results_root=results_root,
        project_root=project_root,
        resume_directory=run_directory,
        register_version=record_path is None,
    )
    if record_path is not None:
        final_id = build_run_id(utc_now(), get_git_commit())
        destination = results_root / "runs" / f"layer_{config['layer']:02d}" / final_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((run_directory / "manifest.json").read_text())
        manifest.update(
            run_id=final_id,
            source_trial_directory=str(source.relative_to(project_root)),
            source_trial_run_id=manifest["run_id"],
            published_at=utc_now().isoformat(),
        )
        write_json_atomic(run_directory / "manifest.json", manifest)
        record_path.parent.mkdir(exist_ok=True)
        record = {
            "status": "publishing",
            "trial_id": config["tuning"]["trial_id"],
            "source_trial_directory": str(source.relative_to(project_root)),
            "run_directory": str(destination.relative_to(project_root)),
            "checkpoint_path": str((destination / "checkpoints/best_checkpoint.pt").relative_to(project_root)),
            "total_epoch_budget": epochs,
            "best_epoch": result.best_epoch,
            "metrics": result.metrics,
        }
        write_json_atomic(record_path, record)
        result = _publish_trial(run_directory, record_path, record, project_root)
    return result


def _publish_trial(staging: Path, record_path: Path, record: dict,
                   project_root: Path) -> TrainingResult:
    """Move completed artifacts and register them; retry safely after interruption."""
    destination = project_root / record["run_directory"]
    if not destination.exists():
        staging.rename(destination)
    result = TrainingResult(destination, record["best_epoch"], record["metrics"])
    _register_run(result, project_root)
    record["status"] = "completed"
    write_json_atomic(record_path, record)
    return result
