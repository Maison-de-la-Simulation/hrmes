import xarray as xr
import numpy as np
import torch
from sklearn.decomposition import PCA

from conf import (
    length, device, mask_dataset_path, msft_dataset_path
)

def load_msft():
    return xr.open_dataset(msft_dataset_path, decode_times=False)

def load_mask_ds():
    return xr.open_dataset(mask_dataset_path, decode_times=False)

def get_mask(mask_ds):
    return mask_ds.tmask[0, 0, :, :]

def get_bathy(mask_ds):
    return np.sum(mask_ds.e3t_0[0,:,:,:] * mask_ds.tmask[0,:,:], axis=0)

def get_ssca(dataset):
    dataset = np.array(dataset)
    x, y = dataset.shape[1:3]
    nbyears = dataset.shape[0] // 12
    arr = np.reshape(dataset, (nbyears, 12, x, y))
    arr = np.mean(arr, axis=0)
    arr = np.tile(arr, (nbyears, 1, 1, 1))
    arr = np.reshape(arr, (12 * nbyears, x, y))
    return dataset - arr

def min_max_normalization(array):
    m = np.nanmin(array, keepdims=True)
    M = np.nanmax(array, keepdims=True)
    return (array - m) / (M - m)

def pointwise_normalization(ssca):
    m = np.nanmean(ssca, axis=None, keepdims=True)
    std = np.nanstd(ssca, axis=None, keepdims=True)
    return (ssca - m) / (2.0 * std)

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

def make_data(maps):
    data = []
    for i in range(maps.shape[0] - length):
        data.append(torch.arange(i, i + length + 1))
    return torch.stack(data, axis=0)

def make_mask(ssca):
    bool_mask = torch.tensor(
        np.asarray(np.isfinite(ssca[0])), dtype=bool
    ).to(device=device)
    int_mask = bool_mask.to(device=device, dtype=int)
    return bool_mask, int_mask

def pca(ssca, n_comp):
    pca = PCA(n_comp, whiten=False)
    pca.fit(ssca)
    return pca, pca.transform(ssca)
