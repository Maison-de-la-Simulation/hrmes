import xarray as xr
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.decomposition import PCA

from conf import (
    length, device, mask_dataset_path, msft_dataset_path, msft_dataset_path_prefix,
    batch_size, train_test_split_ratio
)

class SimulationBatch:

    def __init__(self, simulations, mask=None, bathy=None):
        self.simulations = simulations
        self.mask = mask
        self.bathy = bathy

    @staticmethod
    def load_simulation(index=0):
        signals = []
        for filename in msft_dataset_path[index]:
            path = os.path.join(msft_dataset_path_prefix, filename)
            signals.append(xr.open_dataset(path, decode_times=False))
        return xr.concat(signals, dim="time").astype(np.float32)

    @classmethod
    def load(cls):
        simulations = []
        for i in range(len(msft_dataset_path)):
            simulations.append(cls.load_simulation(i))
        return cls(simulations)

    def load_mask_bathy(self):
        mask_ds = xr.open_dataset(mask_dataset_path, decode_times=False)
        self.mask = mask_ds.tmask[0, 0, :, :]
        self.bathy = np.sum(mask_ds.e3t_0[0,:,:,:] * mask_ds.tmask[0,:,:], axis=0)

    def __iter__(self):
        return iter(self.simulations)

    @classmethod
    def concat(cls, sim_batch1, sim_batch2):
        return cls(
            [*sim_batch1, *sim_batch2],
            mask=sim_batch1.mask,
            bathy=sim_batch1.bathy
        )

    def __getitem__(self, index):
        return self.simulations[index]

    @staticmethod
    def get_ssca(dataset):
        dataset = np.array(dataset.MSFT)
        dataset[dataset == 0.] = np.nan
        x, y = dataset.shape[1:3]
        nbyears = dataset.shape[0] // 12
        arr = np.reshape(dataset, (nbyears, 12, x, y))
        arr = np.mean(arr, axis=0)
        arr = np.tile(arr, (nbyears, 1, 1, 1))
        arr = np.reshape(arr, (12 * nbyears, x, y))
        return dataset - arr

    def convert2ssca(self):
        for i in range(len(self.simulations)):
            self.simulations[i] = self.get_ssca(self.simulations[i])

    def infer_mask(self):
        self.bool_mask = torch.tensor(
            np.asarray(np.isfinite(self.simulations[0][0])), dtype=bool
        ).to(device=device)
        self.int_mask = self.bool_mask.to(device=device, dtype=int)

    @staticmethod
    def min_max_normalization(array):
        m = np.nanmin(array, keepdims=True)
        M = np.nanmax(array, keepdims=True)
        return (array - m) / (M - m)

    @staticmethod
    def pointwise_normalization(ssca):
        m = np.nanmean(ssca, axis=None, keepdims=True)
        std = np.nanstd(ssca, axis=None, keepdims=True)
        return (ssca - m) / (2.0 * std)

    @staticmethod
    def neighbor_normalization(ssca):
        ssca_tensor = torch.from_numpy(ssca)
        unfolded = torch.nn.Unfold(
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1
        )(ssca_tensor.unsqueeze(0)).numpy()
        m = torch.from_numpy(
            np.nanmean(unfolded, axis=1, keepdims=True).reshape(1, 332, 362)
        )
        std = torch.from_numpy(
            np.nanstd(unfolded, axis=1, keepdims=True).reshape(1, 332, 362)
        )
        return (ssca_tensor - m) / (2.0 * std)

    def _normalize(self, normalize_fun, sim_batch):
        for i in range(len(sim_batch)):
            sim_batch[i] = normalize_fun(sim_batch[i])

    def normalize(self, method="minmax"):
        if method == "minmax":
            self._normalize(self.min_max_normalization, self.simulations)
        elif method == "pointwise_stats":
            self._normalize(self.pointwise_normalization, self.simulations)
        elif method == "neighbor_stats":
            self._normalize(self.neighbor_normalization, self.simulations)
        else:
            raise NotImplementedError()

    def apply_mask(self):
        bool_mask = self.bool_mask.cpu().numpy()
        self.masked_simulations = [None] * len(self.simulations)
        for i in range(len(self.simulations)):
            self.masked_simulations[i] = self.simulations[i][:, bool_mask]

    def compute_pca(self, n_comp, transform=True):
        tmp = np.concatenate(self.masked_simulations, axis=0)
        pca = PCA(n_comp, whiten=False)
        pca.fit(tmp)
        if transform:
            for i in range(len(self.masked_simulations)):
                self.masked_simulations[i] = pca.transform(self.masked_simulations[i])
            self._normalize(self.pointwise_normalization, self.masked_simulations)
        self.pca = pca
        return pca

    def to_torch(self):
        for i in range(len(self.masked_simulations)):
            self.masked_simulations[i] = torch.from_numpy(self.masked_simulations[i])

    def make_input_indices(self):
        self.indices = []
        for j in range(len(self.masked_simulations)):
            data = []
            for i in range(self.masked_simulations[j].shape[0] - length):
                data.append(torch.arange(i, i + length + 1))
            self.indices.append(torch.stack(data, axis=0))

    def make_train_test_ds(self, index):
        indices = self.indices[index]
        train_test_split_index = int(len(indices) * train_test_split_ratio)
        train_ds = HRMESDataset(
            self.simulations[index],
            self.masked_simulations[index],
            indices[:train_test_split_index]
        )
        test_ds = HRMESDataset(
            self.simulations[index],
            self.masked_simulations[index],
            indices[train_test_split_index:]
        )
        return train_ds, test_ds

    def train_test_ds(self):
        train_ds, test_ds = zip(
            *[self.make_train_test_ds(i) for i in range(len(self.simulations))]
        )
        return ConcatDataset(train_ds), ConcatDataset(test_ds)

    @staticmethod
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


class HRMESDataset(Dataset):

    def __init__(self, raw_maps, pca_maps, indices):
        self.indices = indices
        self.raw_maps = raw_maps
        self.pca_maps = pca_maps

    def __len__(self):
        return self.indices.shape[0]

    def _get(self, maps, idx):
        idx = self.indices[idx]
        return maps[idx[:-1]], maps[idx[-1]]

    def get_raw(self, idx):
        return self._get(self.raw_maps, idx)

    def get_pca(self, idx):
        return self._get(self.pca_maps, idx)

    def get(self, idx):
        return self.get_raw(idx), self.get_pca(idx)

    def __getitem__(self, idx):
        return self.get_pca(idx)
