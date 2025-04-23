from typing import Any, Literal

import ivy
from ivy import Array

from .base import TransformerBase
from .inverse import InverseTransformer


class Divider(TransformerBase):
    """Divides the input image into two halves."""

    direction: Literal["horizontal", "vertical"] = "horizontal"

    def transform(self, image: Array, /, **kwargs: Any) -> Array:
        if self.direction == "horizontal":
            return ivy.stack(
                (
                    image[..., :, : image.shape[-1] // 2, :],
                    image[..., :, image.shape[-1] // 2 :, :],
                ),
                axis=-4,
            )
        elif self.direction == "vertical":
            return ivy.stack(
                (
                    image[..., : image.shape[-2] // 2, :, :],
                    image[..., image.shape[-2] // 2 :, :, :],
                ),
                axis=-4,
            )

    def inverse_transform(self, image: Array, /, **kwargs: Any) -> Array:
        return ivy.concat(
            (image[..., 0, :, :, :], image[..., 1, :, :, :]),
            axis=-2 if self.direction == "horizontal" else -3,
        )


def Concater() -> InverseTransformer[Divider]:
    """Concats the two halves of the image."""
    return InverseTransformer(Divider())
