import os
from datetime import datetime

from PIL import Image
import torch
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt


def save_samples(generated_data, epoch, args, contexts=None):
    # samples have shape (batch_size, height, width, n_channels)
    # and is in [-1, 1]
    samples = generated_data["sample"]

    # normalize channels by norm if real and imaginary parts are given
    if not args.polar:
        samples = samples / np.linalg.norm(samples, axis=-1, keepdims=True)
    
    # convert to [0, 255] and round to integers
    images_processed = ((samples + 1) * 0.5 * 255).round().astype("uint8")

    # add third channel of zeros
    if images_processed.shape[3] == 2:
        images_processed = np.concatenate([images_processed, np.zeros((args.eval_batch_size, args.resolution, args.resolution, 1))], axis=3).astype("uint8")

    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
    out_dir = f"./{args.samples_dir}/{current_date}_{args.dataset_name}_{epoch}/"
    os.makedirs(out_dir)
    for idx, (data, image) in enumerate(zip(samples, images_processed)):
        image = Image.fromarray(image)
        if contexts is not None:
            image.save(f"{out_dir}/{epoch}_{contexts[idx]}_{idx}.png")
            np.save(f"{out_dir}/{epoch}_{contexts[idx]}_{idx}.npy", data)
        else:
            image.save(f"{out_dir}/{epoch}_{idx}.png")
            np.save(f"{out_dir}/{epoch}_{idx}.npy", data)

    # utils.save_image(generated_images["sample_pt"],
    #                  f"{out_dir}/{epoch}_grid.jpeg",
    #                  nrow=args.eval_batch_size // 4)


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
        f"should be numpy array or torch tensor. Got {type(tensor)}.")
