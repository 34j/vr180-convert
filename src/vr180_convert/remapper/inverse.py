from typing import Any, TypeVar

import attrs
from array_api.latest import Array

from vr180_convert.remapper.base import RemapperBase

T = TypeVar("T", bound=RemapperBase)


@attrs.define()
class InverseRemapper[T: RemapperBase](RemapperBase):
    """Remapper that calls inverse_transform() in transform() and vice versa."""

    transformer: T
    """The transformer to be inverted."""

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        return self.transformer.inverse_remap(x, y, **kwargs)

    def inverse_remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        return self.transformer.remap(x, y, **kwargs)
