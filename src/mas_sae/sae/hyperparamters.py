from dataclasses import dataclass

@dataclass
class Hyperparameters:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 256
    train_shuffle: bool = True
    
    lr: float = 3e-4
    
    input_dim: int = 768
    hidden_dim: int = 8
    latent_dim: int = 12888
    
    patience: int = 20
    lr_patience: int = 5

    sparsity_mode: str = "l1"
    top_k: int | None = 512
    sparsity_coefficient: float = 0.1
    inactivity_window_examples: int = 10000

    tied_initialization: bool = True

    aux_coefficient: float = 0.0
    aux_k: int = 512
    aux_dead_window_examples: int = 10000


def validate_hyperparameters(hp: Hyperparameters) -> None:
    """Reject invalid settings before allocating a model or creating a run."""
    import math

    for name in ("epochs", "batch_size", "latent_dim", "patience",
                 "lr_patience", "inactivity_window_examples", "aux_k",
                 "aux_dead_window_examples"):
        value = getattr(hp, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(hp.seed) is not int or not 0 <= hp.seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if type(hp.train_shuffle) is not bool:
        raise ValueError("train_shuffle must be a boolean.")
    if type(hp.tied_initialization) is not bool:
        raise ValueError("tied_initialization must be a boolean.")
    for name in ("lr", "sparsity_coefficient", "aux_coefficient"):
        value = getattr(hp, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if hp.lr <= 0 or hp.sparsity_coefficient < 0:
        raise ValueError("lr must be positive and sparsity_coefficient nonnegative.")
    if hp.aux_coefficient < 0:
        raise ValueError("aux_coefficient must be nonnegative.")
    if hp.aux_coefficient > 0:
        if hp.sparsity_mode != "topk":
            raise ValueError("AuxK requires topk sparsity mode.")
        if hp.aux_k > hp.latent_dim:
            raise ValueError("aux_k must not exceed latent_dim when AuxK is enabled.")
    if not isinstance(hp.sparsity_mode, str) or hp.sparsity_mode not in {"l1", "topk"}:
        raise ValueError("sparsity_mode must be l1 or topk.")
    if hp.sparsity_mode == "topk" and (
        type(hp.top_k) is not int or not 1 <= hp.top_k <= hp.latent_dim
    ):
        raise ValueError("TopK requires an integer top_k between 1 and latent_dim.")
