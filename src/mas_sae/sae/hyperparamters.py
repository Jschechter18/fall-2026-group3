from dataclasses import dataclass

@dataclass
class Hyperparameters:
    epochs = 2
    
    input_dim = 4
    hidden_dim = 8
    latent_dim = 64
    
    sparsity_coefficient = 1e-3