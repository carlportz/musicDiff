import numpy as np
import torch
from torch.utils.data import Dataset
from simple_diffusion.utils import numpy_to_pil
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, path_to_npz_files, polar=False, transforms=None):
        self.transforms = transforms
        self.polar = polar
        self.data = []
        for npz_file in sorted(os.listdir(path_to_npz_files)):
            data = np.load(os.path.join(path_to_npz_files, npz_file))
            data = data['q_nm']
            if self.polar:
                data = np.array([np.abs(data), np.angle(data)])
            else:
                data = np.array([data.real, data.imag])
            data = np.transpose(data, (1, 2, 0)) # (width, height, channels)
            self.data.append(data)
        self.data = np.array(self.data) # (n_samples, width, height, channels)

    # TODO: move conversion of data to transforms
    # implement reverse transform for image generation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transforms(self.data[idx]) if self.transforms is not None else self.data[idx]

    def save_as_images(self, out_dir="images"):
        print("Saving training data as images...")
        for idx, data in enumerate(self.data):
            np.save(f"{out_dir}/sample_{idx}.npy", data)

            if self.polar:
                data = np.stack([data[:,:,0] / np.max(data[:,:,0]) * 255, (data[:,:,1] / np.max(np.abs(data[:,:,1])) + 1) * 0.5 * 255, np.zeros_like(data[:,:,0])], axis=-1).astype("uint8")
            
                image = Image.fromarray(data)
                image.save(f"{out_dir}/sample_{idx}_polar.png")

            else:
                data = data / np.max(np.linalg.norm(data, axis=-1)) * 255
                data = np.stack([data[:,:,0], data[:,:,1], np.zeros_like(data[:,:,0])], axis=-1).astype("uint8")
                
                image = Image.fromarray(data)
                image.save(f"{out_dir}/sample_{idx}.png")