from __future__ import annotations

from abc import ABCMeta
from typing import Any

import attrs
import cv2 as cv
import ivy
import numpy as np
from ivy import Array, NativeArray

from ..base import TransformerBase
from .remapper import RemapperBase


def _remap(
    image: Array | NativeArray,
    x: Array | NativeArray,
    y: Array | NativeArray,
    /,
    *args: Any,
    **kwargs: Any,
) -> cv.Mat:
    return cv.remap(
        ivy.to_numpy(image).astype(np.float32),
        ivy.to_numpy(x).astype(np.float32),
        ivy.to_numpy(y).astype(np.float32),
        *args,
        **kwargs,
    )


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
                    cv.INTER_LINEAR,
                )
                xmap, ymap = remapper.remap(xmap, ymap, image=image, **kwargs)
            else:
                xmap, ymap = remapper.remap(xmap, ymap, **kwargs)
        return _remap(
            x,
            xmap,
            ymap,
            cv.INTER_LINEAR,
        )

    def inverse_transform(self, x: Array, /, **kwargs: Any) -> Array:
        raise NotImplementedError()
