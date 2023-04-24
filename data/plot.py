"""Module containing utility functions for plotting."""
import matplotlib.pyplot as plt
import numpy as np


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
