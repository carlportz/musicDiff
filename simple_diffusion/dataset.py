import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path_to_npz_files, transforms=None):
        self.transforms = transforms
        self.data = []
        for npz_file in os.listdir(path_to_npz_files):
            data = np.load(os.path.join(path_to_npz_files, npz_file))
            data = data['q_nm']
            data = np.array([data.real, data.imag])
            data = np.transpose(data, (1, 2, 0))
            self.data.append(data)
        self.data = np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transforms(self.data[idx])