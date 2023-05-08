"""Useful classes for holding and manipulating data."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class IndexRange2D:
    """Representation of min/max indices for two (x,y) dimensions."""

    x_min: int = 0
    x_max: int = None
    y_min: int = 0
    y_max: int = None

    @property
    def xs(self):
        """A slice corresponding to the x indices."""
        return slice(self.x_min, self.x_max, 1)

    @property
    def ys(self):
        """A slice corresponding to the y indices."""
        return slice(self.y_min, self.y_max, 1)

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the index range."""
        return self.x_max - self.x_min, self.y_max - self.y_min


@dataclass
class IndexRange3D:
    """Representation of min/max indices for three (x,y,z) dimensions."""

    x_min: int = 0
    x_max: int = None
    y_min: int = 0
    y_max: int = None
    z_min: int = 0
    z_max: int = None

    @property
    def xs(self):
        """A slice corresponding to the x indices."""
        return slice(self.x_min, self.x_max, 1)

    @property
    def ys(self):
        """A slice corresponding to the y indices."""
        return slice(self.y_min, self.y_max, 1)

    @property
    def zs(self):
        """A slice corresponding to the z indices."""
        return slice(self.z_min, self.z_max, 1)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """The shape of the index range."""
        return self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min
