"""Module containing code for loading and pre-processing the data."""
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data.plot import plot_image_stack

SURFACE_VOLUME = "surface_volume"
MAX_VALUE = 65535.0


def read(
    fragment_path: Path, z_min: int, z_max: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the mask, label and 3D x-ray data of the given fragment.

    Args:
        fragment_path: path to directory with fragment to read
        z_min: start reading x-ray .tif files starting at this depth
        z_max: stop reading x-ray .tif files from this depth

    Returns:
        the mask, label and x-ray image stack
    """
    mask = _read_image(fragment_path / "mask.png")
    label = _read_image(fragment_path / "inklabels.png")

    path = fragment_path / SURFACE_VOLUME
    images = []

    for file in tqdm(sorted(os.listdir(path)), desc=f"Reading {path}"):
        name, ext = os.path.splitext(file)

        if z_min <= int(name) < z_max and ext == ".tif":
            images.append(_read_image(path / file))

    images = np.stack(images, axis=0)

    return mask, label, images


def _read_image(path: Path) -> np.ndarray:
    """Read the given image and return it as a numpy array."""
    return np.array(Image.open(path), dtype=np.float32)


def preprocess(
    mask: np.ndarray, label: np.ndarray, images: np.ndarray, half_width: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess the data from a single fragment."""
    images = images / MAX_VALUE  # Normalize pixel values to [0, 1] range

    # Don't read pixels too close to the border
    not_border = np.zeros(mask.shape)
    not_border[half_width:-half_width, half_width:-half_width] = 1.0
    mask = mask * not_border

    return tuple(map(lambda x: torch.from_numpy(x), [mask, label, images]))


def main():
    """Load some example training data and show it."""
    data_path = Path(__file__).parent / "train" / "1"
    z_min = 27
    num_slices = 10
    mask, label, images = read(data_path, z_min, z_min + num_slices)

    ir_img = Image.open(data_path / "ir.png")

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ir_img)
    ax[0].set_title("IR")
    ax[1].imshow(mask)
    ax[1].set_title("mask")
    ax[2].imshow(label)
    ax[2].set_title("label")

    plot_image_stack(images, "x-ray data")

    mask, label, images = preprocess(mask, label, images, half_width=5)
    print("Preprocessing outputs ...")
    print(f"Mask: {mask.shape}, {mask.dtype}, {mask.min()}, {mask.max()}")
    print(f"Label: {label.shape}, {label.dtype}, {label.min()}, {label.max()}")
    print(f"Images: {images.shape}, {images.dtype}, {images.min()}, {images.max()}")


if __name__ == "__main__":
    main()
