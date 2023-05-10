import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A ResNet block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Sequential = None,
    ):
        """Create a ResNet block."""
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of this ResNet block."""
        output = self.conv1(x)
        output = self.conv2(output)
        residual = self.downsample(x) if self.downsample else x
        output = output + residual
        return self.relu(output)
