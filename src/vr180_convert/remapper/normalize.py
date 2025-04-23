from typing import Any, Literal

import attrs
from ivy import Array

from vr180_convert.remapper.base import RemapperBase


@attrs.define()
class NormalizeRemapper(RemapperBase):
    """Normalize the coordinates to [-1, 1]."""

    center: tuple[float, float] | None = None
    """The center of the image. If None, the center is the center of the image."""
    scale: tuple[float, float] | Literal["min", "max"] | None = None
    """The scale of the image. If "min" or None, the scale is the minimum of the width and height.
    If "max", the scale is the maximum of the width and height."""

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        center = self.center or (x.shape[-2] / 2, x.shape[-1] / 2)
        scale = (
            min(x.shape[-2], x.shape[-1])
            if self.scale in ["min", None]
            else max(x.shape[-2], x.shape[-1]) if self.scale == "max" else self.scale
        )
        x = (x - center[0]) / scale * 2
        y = (y - center[1]) / scale * 2
        return x, y

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        center = self.center or (x.shape[-2] / 2, x.shape[-1] / 2)
        scale = (
            min(x.shape[-2], x.shape[-1])
            if self.scale in ["min", None]
            else max(x.shape[-2], x.shape[-1]) if self.scale == "max" else self.scale
        )
        x = x * scale[0] + center[0]
        y = y * scale[1] + center[1]
        return x, y
