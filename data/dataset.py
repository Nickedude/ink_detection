"""Module containing datasets."""
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from data.plot import plot_image_stack
from data.preprocessor import preprocess, read


class SubVolumeDataset(Dataset):
    def __init__(self, fragment_path: Path, half_width: int, device: torch.device):
        """Create a dataset that holds subvolumes of a fragment.

        Args:
            fragment_path: path to the fragment this dataset should yield data from
            half_width: return data from a [half_width*2 x half_width*2] window around each pixel
            device: the device to place the loaded data on, e.g. 'cuda'

        """
        self.fragment_path = fragment_path

        # Preprocessing settings
        self.z_min = 27
        self.z_max = self.z_min + 10
        self.half_width = half_width

        # Read the actual data
        self.mask, self.label, self.data = read(fragment_path, self.z_min, self.z_max)
        self.mask, self.label, self.data = preprocess(
            self.mask, self.label, self.data, device
        )

        print(self.mask.shape)
        self.indices = torch.nonzero(self.mask)
        self.length = len(self.indices)

    def __len__(self):
        """Number of pixels this dataset contains."""
        return self.length

    def __getitem__(self, index):
        x, y = self.indices[index]
        label = self.label[x, y]
        subvolume = self.data[
            :,
            x - self.half_width : x + self.half_width + 1,
            y - self.half_width : y + self.half_width + 1,
        ]

        return subvolume, label


def main():
    """Create a dataset and print some information about it."""
    path = Path(__file__).parent / "train" / "1"
    device = torch.device("cpu")

    print(f"Creating a dataset from {path} on device {device}")
    half_width = 5
    dataset = SubVolumeDataset(path, half_width, device)
    print(f"Length of dataset: {len(dataset)}")

    index = 36373  # Manually selected index containing some nice data
    subvolume, label = dataset[index]
    plot_image_stack(subvolume.numpy(), f"X-ray data with label = {label}")

    label_to_plot = dataset.label.numpy()
    offset = 2500
    label_to_plot = label_to_plot[:1000, offset:5000]

    fig, ax = plt.subplots()
    ax.imshow(label_to_plot)
    row, col = tuple(dataset.indices[index].numpy())
    col -= offset
    print(col, row)
    size = half_width * 2 + 1
    rectangle = patches.Rectangle((col, row), width=size, height=size, color="red")
    ax.add_patch(rectangle)
    plt.title("Label mask, data from dataset shown in red square")
    plt.show()


if __name__ == "__main__":
    main()
