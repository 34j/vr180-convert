from __future__ import annotations

from abc import ABCMeta
from typing import Any

import attrs
import ivy
import torch
import torch.nn.functional as F
from ivy import Array, NativeArray

from vr180_convert.base import TransformerBase

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
        The image to be remapped of shape (..., height, width, channels).
    x : Array | NativeArray
        The x coordinates of shape (..., height, width).
    y : Array | NativeArray
        The y coordinates of shape (..., height, width).
    **kwargs : Any
        Additional keyword arguments to be passed to the remap function.

    """
    shape_extra = ivy.broadcast_shapes(
        ivy.shape(image)[:-3], ivy.shape(x)[:-2], ivy.shape(y)[:-2]
    )
    image = ivy.broadcast_to(image, (*shape_extra, *image.shape[-3:])).reshape(
        (-1, *image.shape[-3:])
    )
    image = ivy.moveaxis(image, -1, 1)
    x = ivy.broadcast_to(x, (*shape_extra, *x.shape[-2:])).reshape((-1, *x.shape[-2:]))
    y = ivy.broadcast_to(y, (*shape_extra, *y.shape[-2:])).reshape((-1, *y.shape[-2:]))
    x = 2 * x / (image.shape[-1] - 1) - 1
    y = 2 * y / (image.shape[-2] - 1) - 1
    xy = xy = ivy.stack([x, y], axis=-1)
    if ivy.current_backend_str() != "torch":
        image = torch.from_numpy(ivy.to_numpy(image))
        xy = torch.from_numpy(ivy.to_numpy(ivy.stack([x, y], axis=-1)))
    result = F.grid_sample(image.float(), xy.float(), **kwargs).moveaxis(1, -1)
    result = result.reshape((*shape_extra, *result.shape[-3:]))
    if ivy.current_backend_str() == "torch":
        result = ivy.asarray(result)
    return ivy.asarray(result.cpu().numpy())


@attrs.define(kw_only=True)
class RemapperTransformer(TransformerBase, metaclass=ABCMeta):
    remappers: list[RemapperBase]
    size_output: tuple[int, int]
    remap_kwargs: dict[str, Any] | None = None

    """Base class for transformers."""

    def transform(self, image: Array, /, **kwargs: Any) -> Array:
        image = ivy.asarray(image)
        for i in range(len(self.remappers)):
            remapper = self.remappers[i]
            if remapper.requires_image:

                def inner(
                    x: Array, y: Array, /, i: int = i, **kwargs_inner: Any
                ) -> tuple[Array, Array]:
                    for remapper_before in self.remappers[:i]:
                        x, y = remapper_before.inverse_remap(x, y, **kwargs_inner)
                    return x, y

                remapper.fit(image, inner, **kwargs)

        xmap, ymap = ivy.meshgrid(
            ivy.arange(self.size_output[0]), ivy.arange(self.size_output[1])
        )
        for remapper in reversed(self.remappers):
            xmap, ymap = remapper.remap(xmap, ymap, **kwargs)
        return _remap(image, xmap, ymap, **(self.remap_kwargs or {}))

    def inverse_transform(self, image: Array, /, **kwargs: Any) -> Array:
        raise NotImplementedError()
