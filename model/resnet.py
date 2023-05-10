"""An implementation of ResNet 34."""
import torch
import torch.nn as nn

from model.blocks import ResidualBlock


class ResNet(nn.Module):
    """An implementation of ResNet 18 adapted to an input resolution of 63x63 pixels.

    See: https://arxiv.org/pdf/1512.03385.pdf.
    Figure 3 and table 1 are helpful to understand the architecture.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=10, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_group_0 = self.create_block_group(
            in_channels=64, out_channels=64, num_blocks=2, stride=1
        )
        self.layer_group_1 = self.create_block_group(
            in_channels=64,
            out_channels=128,
            num_blocks=2,
            stride=2,
        )
        self.layer_group_2 = self.create_block_group(
            in_channels=128,
            out_channels=256,
            num_blocks=2,
            stride=2,
        )
        self.layer_group_3 = self.create_block_group(
            in_channels=256,
            out_channels=512,
            num_blocks=2,
            stride=2,
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(in_features=512, out_features=1)
        self.sigmoid = nn.Sigmoid()  # Output is probability of ink in the center pixel

    @staticmethod
    def create_block_group(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a group of residual blocks.

        Args:
            in_channels: the number of input channels to this group of residual blocks
            out_channels: the number of output channels from this group of residual blocks
            num_blocks: the number of residual blocks in this group

        """
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        group = [ResidualBlock(in_channels, out_channels, stride, downsample)]

        for i in range(num_blocks - 1):
            group.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of this ResNet.

        Args:
            x: Input tensor of shape [N, C=10, H=63, W=63]

        """
        # [(W−K+2P)/S]+1
        x = self.conv1(x)  # [N, 10, 63, 63] -> [N, 64, 63, 63]
        x = self.max_pool(x)  # [N, 64, 63, 63] -> [N, 64, 32, 32]
        x = self.layer_group_0(x)  # [N, 64, 32, 32] -> [N, 64, 32, 32]
        x = self.layer_group_1(x)  # [N, 64, 32, 32] -> [N, 128, 16, 16]
        x = self.layer_group_2(x)  # [N, 128, 16, 16] -> [N, 256, 8, 8]
        x = self.layer_group_3(x)  # [N, 256, 8, 8] -> [N, 512, 4, 4]
        x = self.avg_pool(x)  # [N, 512, 4, 4] -> [N, 512, 1, 1]
        x = x.view(x.shape[0], x.shape[1])  # [N, 512, 1, 1] -> [N, 512]
        x = self.fc(x)

        return self.sigmoid(x)
