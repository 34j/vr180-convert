from typing import Any

import ivy
from ivy import Array

from .base import TransformerBase
from .inverse import InverseTransformer


class Divider(TransformerBase):
    """Divides the input image into two halves."""

    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        return ivy.stack(
            [x[..., :, : x.shape[-1] // 2], x[..., :, x.shape[-1] // 2 :]], axis=-3
        )

    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        return ivy.concat(x[..., 0, :, :], x[..., 1, :, :], axis=-1)


def Concater() -> InverseTransformer[Divider]:
    """Concats the two halves of the image."""
    return InverseTransformer(Divider())
