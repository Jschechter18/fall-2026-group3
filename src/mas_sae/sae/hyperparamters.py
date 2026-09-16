from dataclasses import dataclass


@dataclass
class Hyperparameters:
    epochs: int = 10
    batch_size: int = 32
    
    lr: float = 1e-4
    
    input_dim: int = 4
    hidden_dim: int = 8
    latent_dim: int = 64
    num_layers: int = 2

    sparsity_coefficient: float = 1e-3
