from __future__ import annotations

from abc import ABCMeta
from typing import Any

import attrs
import ivy
import numpy as np
import torch
import torch.nn.functional as F
from ivy import Array, NativeArray

from ..base import TransformerBase
from .base import RemapperBase


def _remap(
    image: Array | NativeArray,
    x: Array | NativeArray,
    y: Array | NativeArray,
    /,
    **kwargs: Any,
) -> Any:
    """
    Remap the image using the given x and y coordinates.

    Parameters
    ----------
    image : Array | NativeArray
        The image to be remapped of shape (..., width, height, channels).
    x : Array | NativeArray
        The x coordinates of shape (..., width, height).
    y : Array | NativeArray
        The y coordinates of shape (..., width, height).
    **kwargs : Any
        Additional keyword arguments to be passed to the remap function.

    """
    shape = image.shape
    image = image.reshape((-1, *shape[-3:]))
    image = ivy.moveaxis(image, -1, 1)
    x = x.reshape((-1, *shape[-3:-1]))
    y = y.reshape((-1, *shape[-3:-1]))
    x = 2 * x / (x.shape[-2] - 1) - 1
    y = 2 * y / (y.shape[-1] - 1) - 1
    result = F.grid_sample(
        torch.tensor(image).float(),
        torch.tensor(ivy.stack([x, y], axis=-1)).float(),
        **kwargs,
    ).moveaxis(1, -1)
    return ivy.asarray(result.reshape((*shape[:-3], *result.shape[-3:])))


@attrs.define(kw_only=True)
class RemapperTransformer(TransformerBase, metaclass=ABCMeta):
    remappers: list[RemapperBase]
    size_output: tuple[int, int]

    """Base class for transformers."""

    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        xmap, ymap = np.meshgrid(
            np.arange(self.size_output[0]), np.arange(self.size_output[1])
        )
        for remapper in [*self.remappers]:
            if remapper.requires_image:
                image = _remap(
                    x,
                    xmap,
                    ymap,
                )
                xmap, ymap = remapper.remap(xmap, ymap, image=image, **kwargs)
            else:
                xmap, ymap = remapper.remap(xmap, ymap, **kwargs)
        return _remap(
            x,
            xmap,
            ymap,
        )

    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        raise NotImplementedError()
