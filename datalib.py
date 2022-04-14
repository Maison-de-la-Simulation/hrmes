import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from conf import (train_test_split_index, batch_size, device)

class HRMESDataset(Dataset):

    def __init__(self, maps, indices):
        self.indices = indices
        self.maps = maps

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.maps[idx[:-1]], self.maps[idx[-1]]

def make_datasets(maps, data):
    train_ds = HRMESDataset(maps, data[:train_test_split_index])
    test_ds = HRMESDataset(maps, data[train_test_split_index:])
    print(f"There are {len(train_ds)} training samples and {len(test_ds)} test_samples")
    return train_ds, test_ds

def make_dataloaders(train_ds, test_ds):
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=2
    )
    return train_dataloader, test_dataloader
