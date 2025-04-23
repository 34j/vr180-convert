from typing import Any, Generic, TypeVar

import attrs
from ivy import Array

from .base import TransformerBase

T = TypeVar("T", bound=TransformerBase)


@attrs.define()
class InverseTransformer(TransformerBase, Generic[T]):
    """Remapper that calls inverse_transform() in transform() and vice versa."""

    transformer: T
    """The transformer to be inverted."""

    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        return self.transformer.inverse_transform(x, **kwargs)

    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        return self.transformer.transform(x, **kwargs)
