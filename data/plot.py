"""Module containing utility functions for plotting."""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def plot_image_stack(images: np.ndarray, title: str):
    """Plot a stack of images."""
    num_rows = 2
    num_cols = images.shape[0] // num_rows
    fig, ax = plt.subplots(num_rows, num_cols)

    for i in range(images.shape[0]):
        row = i // num_cols
        col = i % num_cols
        ax[row, col].imshow(images[i])

    ax[0, 0].title.set_text(title)
    plt.show()


def plot_image_with_rectangle(
    image: np.ndarray, title: str, center: Tuple[int, int], **kwargs
):
    """Plot an image with a painted rectangle on top."""
    fig, ax = plt.subplots()
    ax.imshow(image)
    rectangle = patches.Rectangle(center, **kwargs)
    ax.add_patch(rectangle)
    plt.title(title)
    plt.show()


def plot_loss(epochs: List[int], losses: List[float]):
    """Plot the given loss values."""
    plt.plot(epochs, losses, c="blue")
    plt.title("Loss per epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
    plt.close()
