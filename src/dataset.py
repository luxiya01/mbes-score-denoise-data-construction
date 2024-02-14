import numpy as np
import os
import torch
from torch.utils.data import Dataset
from collections import namedtuple

Norm = namedtuple('Norm', ['mean', 'std', 'min', 'max'])

class MBESDataset(Dataset):
    def __init__(self, folder, transform=None,
                 normalize=True,
                 normalize_dataset=False,):
        self.data_folder = os.path.abspath(folder)
        self.data_dict = sorted([x for x in os.listdir(self.data_folder)
                                 if x.endswith('.npz') and 'norm' not in x],
                                key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.normalize = normalize
        self.normalize_dataset = normalize_dataset
        self.dataset_norm = self._compute_dataset_norm()
        self.patch_norms = self._compute_patch_norms()

    def _compute_dataset_norm(self):
        # Load dataset norm if exists
        dataset_norm_path = os.path.join(self.data_folder, 'all_data_norm.npz')
        if os.path.exists(dataset_norm_path):
            dataset_norm = np.load(dataset_norm_path, allow_pickle=True)
            return dataset_norm['dataset_norm'].item()
        # Compute dataset norm
        dataset_path = os.path.join(os.path.dirname(self.data_folder),
                                    'all_data.npz')
        dataset = np.load(dataset_path, allow_pickle=True)
        dataset = {k: torch.from_numpy(v).to(torch.float32)
                   for k, v in dataset.items() if k != 'datetime'}
        dataset_norm = self._compute_norm_stats_for_data(dataset)
        np.savez(dataset_norm_path, dataset_norm=dataset_norm)
        return dataset_norm

    def _compute_patch_norms(self):
        # Load patch norms if exists
        patch_norm_path = os.path.join(self.data_folder, 'patch_norms.npz')
        if os.path.exists(patch_norm_path):
            patch_norms = np.load(patch_norm_path, allow_pickle=True)
            return patch_norms['patch_norms'].item()
        patch_norms = {}
        # Compute patch norms
        for idx, filename in enumerate(self.data_dict):
            filepath = os.path.join(self.data_folder, filename)
            data = np.load(filepath)
            data = {k: torch.from_numpy(v).to(torch.float32)
                    for k, v in data.items()
                    if k not in ['datetime']}
            patch_norms[idx] = self._compute_norm_stats_for_data(data)
        np.savez(patch_norm_path, patch_norms=patch_norms)
        return patch_norms

    def _compute_norm_stats_for_data(self, data):
        return {k: Norm(mean=np.nanmean(v), std=np.nanstd(v),
                        min=np.nanmin(v), max=np.nanmax(v))
                for k, v in data.items() if k not in
                ['datetime', 'idx', 'start_ping', 'start_beam', 'end_ping', 'end_beam']}

    def _normalize_patch(self, data):
        idx = int(data['idx'])
        if not self.normalize:
            return data
        norm = self.patch_norms[idx]
        if self.normalize_dataset:
            norm = self.dataset_norm
        data.update(
            {k: (v - norm[k].min) / (norm[k].max - norm[k].min)
             for k, v in data.items() if k in norm})
        return data

    def denormalize_patch(self, data):
        idx = int(data['idx'])
        if not self.normalize:
            return data
        norm = self.patch_norms[idx]
        if self.normalize_dataset:
            norm = self.dataset_norm
        data.update(
            {k: v * (norm[k].max - norm[k].min) + norm[k].min
             for k, v in data.items() if k in norm})
        return data

    def __getitem__(self, idx):
        """
        Returns a dictionary with the data for the patch at index idx.
        If self.normalize, the return data is normalized to [0, 1] using
        either patch or dataset statistics.
        To denormalize the data, use the denormalize_patch() method.
        """
        filename = self.data_dict[idx]
        filepath = os.path.join(self.data_folder, filename)
        data = np.load(filepath)
        data = {k: torch.from_numpy(v).to(torch.float32)
                for k, v in data.items()
                if k not in ['datetime']}
        data = self._normalize_patch(data)

        return data

    def __len__(self):
        return len(self.data_dict)
