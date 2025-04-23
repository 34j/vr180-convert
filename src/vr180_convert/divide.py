from typing import Any, Literal

import ivy
from ivy import Array

from .base import TransformerBase
from .inverse import InverseTransformer


class Divider(TransformerBase):
    """Divides the input image into two halves."""

    direction: Literal["horizontal", "vertical"] = "horizontal"

    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        if self.direction == "horizontal":
            return ivy.stack(
                (x[..., :, : x.shape[-1] // 2, :], x[..., :, x.shape[-1] // 2 :, :]),
                axis=-4,
            )
        elif self.direction == "vertical":
            return ivy.stack(
                (x[..., : x.shape[-2] // 2, :, :], x[..., x.shape[-2] // 2 :, :, :]),
                axis=-4,
            )

    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        return ivy.concat(
            (x[..., 0, :, :, :], x[..., 1, :, :, :]),
            axis=-2 if self.direction == "horizontal" else -3,
        )


def Concater() -> InverseTransformer[Divider]:
    """Concats the two halves of the image."""
    return InverseTransformer(Divider())
