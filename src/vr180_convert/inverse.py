from typing import Any, TypeVar

import attrs
from array_api.latest import Array

from .base import TransformerBase

T = TypeVar("T", bound=TransformerBase)


@attrs.define()
class InverseTransformer[T: TransformerBase](TransformerBase):
    """Remapper that calls inverse_transform() in transform() and vice versa."""

    transformer: T
    """The transformer to be inverted."""

    def transform(self, image: Array, /, **kwargs: Any) -> Array:
        return self.transformer.inverse_transform(image, **kwargs)

    def inverse_transform(self, image: Array, /, **kwargs: Any) -> Array:
        return self.transformer.transform(image, **kwargs)
