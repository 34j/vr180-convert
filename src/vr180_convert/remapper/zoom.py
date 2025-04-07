from typing import Any

import attrs
from ivy import Array

from vr180_convert.remapper.base import RemapperBase


@attrs.define()
class ZoomRemapper(RemapperBase):
    """Zoom the image."""

    scale: float
    """The zoom scale."""

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        x = x / self.scale
        y = y / self.scale
        return x, y

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        x = x * self.scale
        y = y * self.scale
        return x, y
