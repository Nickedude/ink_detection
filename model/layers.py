"""Module with custom layers."""

import torch.nn as nn


class PrintLayer(nn.Module):
    """Layer for printing intermediate tensors. Useful for debugging."""

    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x
