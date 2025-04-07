from typing import Any

import ivy
from ivy import Array

from .base import TransformerBase


class Divider(TransformerBase):
    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        return ivy.stack(
            [x[..., :, : x.shape[-1] // 2], x[..., :, x.shape[-1] // 2 :]], axis=-3
        )

    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        return ivy.concat(x[..., 0, :, :], x[..., 1, :, :], axis=-1)
