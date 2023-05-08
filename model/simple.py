"""Module containing a very simple example of a neural network to use for ink-detection."""
import torch.nn as nn


def build() -> nn.Sequential:
    """Build a very simple example of a neural network to use for ink-detection."""
    return nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Flatten(start_dim=1),
        nn.LazyLinear(out_features=128),
        nn.ReLU(),
        nn.LazyLinear(out_features=1),
        nn.Sigmoid()
    )
