import numpy as np
import torch
from torch.utils.data import Dataset
from simple_diffusion.utils import numpy_to_pil
import os

class CustomDataset(Dataset):
    def __init__(self, path_to_npz_files, polar=False, transforms=None):
        self.transforms = transforms
        self.data = []
        for npz_file in sorted(os.listdir(path_to_npz_files)):
            data = np.load(os.path.join(path_to_npz_files, npz_file))
            data = data['q_nm']
            if polar:
                data = np.array([np.abs(data), np.angle(data)])
            else:
                data = np.array([data.real, data.imag])
            data = np.transpose(data, (1, 2, 0)) # (width, height, channels)
            self.data.append(data)
        self.data = np.array(self.data) # (n_samples, width, height, channels)

        # TODO: Save training data as images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transforms(self.data[idx]) if self.transforms is not None else self.data[idx]