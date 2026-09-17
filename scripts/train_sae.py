from dataclasses import asdict
from pathlib import Path

import torch

from mas_sae.experiments.artifacts import (
    create_sae_run_directory,
    update_run_manifest,
    write_run_config,
    write_run_history
)
from mas_sae.sae.hyperparamters import Hyperparameters as HP
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE
from mas_sae.sae.dataloader import create_sae_dataloader
from mas_sae.sae.model_runner import ModelRunner
from mas_sae.sae.callbacks.checkpointing import CheckpointEvaluatorCallback
from mas_sae.sae.callbacks.early_stopping import EarlyStoppingCallback


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    results_root = PROJECT_ROOT / "results" / "sae" / "musique"
    subdirectories = ("checkpoints",)
    
    hp = HP()
    
    run_directory = create_sae_run_directory(
        run_name=f"sae-l{hp.latent_dim}",
        results_root=results_root,
        subdirectories=subdirectories,
    )
    
    try:
        write_run_config(run_directory, asdict(hp))
        
        model = SAE(input_dim=hp.input_dim, hidden_dim=hp.hidden_dim, latent_dim=hp.latent_dim)
        
        train_dataloader = create_sae_dataloader(hp.batch_size, split='train', num_workers=2)
        val_dataloader = create_sae_dataloader(hp.batch_size, split='val', num_workers=2)
        test_dataloader = create_sae_dataloader(hp.batch_size, split='test', num_workers=2)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.lr_patience, gamma=0.1)
        
        runner = ModelRunner(model, sparsity_coefficient=hp.sparsity_coefficient, optimizer=optimizer)
        
        checkpoint_evaluator = CheckpointEvaluatorCallback(run_directory / "checkpoints")
        early_stopping = EarlyStoppingCallback(hp.patience)
        
        best_loss = float('inf')
        epoch_history = []
        for epoch in range(hp.epochs):
            train_loss = runner.train_epoch(train_dataloader)
            val_loss = runner.val_epoch(val_dataloader)
            print(f"Epoch {epoch+1}/{hp.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            scheduler.step()
            checkpoint_evaluator.on_validation_end(train_loss, val_loss, best_loss, epoch,
                                                   model, optimizer, scheduler, run_directory)
            epoch_history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
            write_run_history(run_directory, epoch_history)
            
            if early_stopping.on_validation_end(val_loss):
                print(f"Early stopping after epoch {epoch+1}")
                break
        
        test_loss = runner.test(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}")
        
        write_run_history(run_directory, epoch_history, test_loss=test_loss)
        
    except (Exception, KeyboardInterrupt) as error:
        update_run_manifest(
            run_directory,
            status="failed",
            error_message=f"{type(error).__name__}: {error}",
        )
        raise
    
    update_run_manifest(run_directory, status="completed")
    
if __name__ == "__main__":
    main()
    