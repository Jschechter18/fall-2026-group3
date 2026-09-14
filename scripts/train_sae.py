import torch

from mas_sae.sae.hyperparamters import Hyperparameters as HP
from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE
from mas_sae.sae.dataloader import create_sae_dataloader
from mas_sae.sae.model_runner import ModelRunner




def main():
    hp = HP()
    
    model = SAE(input_dim=hp.input_dim, hidden_dim=hp.hidden_dim, latent_dim=hp.latent_dim)
    
    train_dataloader = create_sae_dataloader(32, split='train', num_workers=2)
    val_dataloader = create_sae_dataloader(32, split='val', num_workers=2)
    test_dataloader = create_sae_dataloader(32, split='test', num_workers=2)
    optimizer = torch.optim.Adam(model.parameters())
    
    runner = ModelRunner(model, sparsity_coefficient=hp.sparsity_coefficient, optimizer=optimizer)
    
    for epoch in range(hp.epochs):
        train_loss = runner.train_epoch(train_dataloader)
        val_loss = runner.val_epoch(val_dataloader)
        print(f"Epoch {epoch+1}/{hp.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    test_loss = runner.test(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    
        

if __name__ == "__main__":
    main()