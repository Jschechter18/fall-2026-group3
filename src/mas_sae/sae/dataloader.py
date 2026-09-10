import torch
from torch.utils.data import Dataset, DataLoader

from mas_sae.data.activation_store import ActivationStore


class ActivationDataset(Dataset):
    def __init__(self, location: str = "s3://bucket"):
        self.activation_store = ActivationStore(location)
        
        self.activations: torch.Tensor = self.activation_store.load_activations()

    def __len__(self) -> int:
        return self.activations.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


def create_sae_dataloader(batch_size: int, num_workers: int, shuffle: bool = False,  location: str | None = None):
    if location:
        dataset =  ActivationDataset(location)
    else:
        dataset = ActivationDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
