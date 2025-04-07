from typing import Any

import attrs
from ivy import Array

from vr180_convert.remapper.base import RemapperBase


@attrs.define()
class DenormalizeRemapper(RemapperBase):
    """Denormalize the coordinates from [-1, 1] to the original image size."""

    scale: tuple[float, float]
    """The scale of the image. Recommended to be the half of the width and height of the result image."""
    center: tuple[float, float]
    """The center of the image. Recommended to be the center of the result image."""

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        scale = self.scale
        center = self.center
        x = x * scale[0] + center[0]
        y = y * scale[1] + center[1]
        return x, y

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        scale = self.scale
        center = self.center
        x = (x - center[0]) / scale[0]
        y = (y - center[1]) / scale[1]
        return x, y
