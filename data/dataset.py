"""Module containing datasets."""
from functools import cached_property
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from data.definitions import IndexRange2D, IndexRange3D
from data.plot import plot_image_stack, plot_image_with_rectangle
from data.preprocessor import preprocess, read

DEFAULT_RANGES = IndexRange3D(
    z_min=27,
    z_max=37,
)


class SubVolumeDataset(Dataset):
    """A dataset that holds a subvolume of a fragment."""

    def __init__(
        self,
        fragment_path: Path,
        half_width: int,
        indices_to_read: IndexRange3D = DEFAULT_RANGES,
        exclude_indices: IndexRange2D = None,
        add_dimension: bool = False,
    ):
        """Create a dataset that holds a subvolume of a fragment.

        Args:
            fragment_path: path to the fragment this dataset should yield data from
            half_width: return data from a [half_width*2 x half_width*2] window around each pixel
            indices_to_read: min/max indices to read for the three (x,y,z) dimensional X-ray data
            exclude_indices: min/max indices (relative to indices_to_read) of a 2D window to
                exclude, e.g. for evaluation data
            add_dimension: if set to True, will add an extra dimension to the data, useful when
                using 3D convolutions

        """
        self.fragment_path = fragment_path
        self.add_dimension = add_dimension

        # Preprocessing settings
        self.half_width = half_width
        self.indices_to_read = indices_to_read
        self.exclude_indices = exclude_indices

        # Read the actual data
        self.mask, self.label, self.data = read(
            fragment_path, self.indices_to_read.z_min, self.indices_to_read.z_max
        )

        self.mask = self.mask[self.indices_to_read.xs, self.indices_to_read.ys]
        self.label = self.label[self.indices_to_read.xs, self.indices_to_read.ys]
        self.data = self.data[
            :, self.indices_to_read.xs, self.indices_to_read.ys
        ]  # Z sliced above

        if self.exclude_indices is not None:
            self.mask[self.exclude_indices.xs, self.exclude_indices.ys] = 0.0

        self.mask, self.label, self.data = preprocess(
            self.mask, self.label, self.data, half_width
        )

        self.indices = torch.nonzero(self.mask)  # Shape N x 2
        self.length = len(self.indices)

    def __len__(self) -> int:
        """Number of pixels this dataset contains."""
        return self.length

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the subvolume of data at the given index."""
        x, y = self.indices[index]
        label = self.label[x, y]
        subvolume = self.data[
            :,
            x - self.half_width : x + self.half_width + 1,
            y - self.half_width : y + self.half_width + 1,
        ]

        shape = (1, *subvolume.shape) if self.add_dimension else subvolume.shape
        return subvolume.view(shape), label.view(1)

    @cached_property
    def num_positive_samples(self):
        labels = self.label[self.indices[:, 0], self.indices[:, 1]]
        return torch.sum(labels)

    @cached_property
    def num_negative_samples(self):
        return self.length - self.num_positive_samples

    def subsample(self, factor: int):
        """Subsample the dataset by the given factor."""
        self.indices = self.indices[::factor]
        self.length = len(self.indices)


def main():
    """Create a dataset and print some information about it."""
    path = Path(__file__).parent / "train" / "1"

    print(f"Creating a dataset from {path}")
    half_width = 30
    dataset = SubVolumeDataset(path, half_width, DEFAULT_RANGES)
    print(f"Length of dataset: {len(dataset)}")

    # Selected index containing data with positive label
    positive_indices = torch.nonzero(dataset.label)
    positive_indices = dataset.indices == positive_indices[0]
    positive_indices = torch.logical_and(positive_indices[:, 0], positive_indices[:, 1])
    index = torch.nonzero(positive_indices).numpy().item()
    print(f"Reading index: {index}")

    subvolume, label = dataset[index]
    print(f"Loaded data with shape: {subvolume.shape}")

    plot_image_stack(subvolume.numpy()[0], f"X-ray data with label = {label}")

    label_to_plot = dataset.label.numpy()
    row, col = tuple(dataset.indices[index].numpy())
    size = half_width * 2 + 1
    plot_image_with_rectangle(
        label_to_plot,
        title="Label, visualized data shown in red square, eval data shown in blue square",
        center=(col, row),
        width=size,
        height=size,
        color="red",
        fill=False,
    )


if __name__ == "__main__":
    main()
