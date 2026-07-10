from typing import Any

import array_api_compat
import cv2 as cv
import numpy as np
from array_api.latest import Array

from .base import TransformerBase


class Merger(TransformerBase):
    def transform(self, image: Array, /, **kwargs: Any) -> Array:
        xp = array_api_compat.array_namespace(image)
        colors = [(0, 128, 255), (255, 128, 0)]
        combine = xp.mean(image[0], axis=-1)[..., None] * xp.asarray(colors[0]).reshape(
            [1] * (image[0].ndim - 1) + [3]
        ) + (
            xp.mean(image[1], axis=-1)[..., None]
            * xp.asarray(colors[1]).reshape([1] * (image[1].ndim - 1) + [3])
        )
        combine /= 255
        combine_np = np.asarray(combine)
        cv.putText(
            combine_np,
            "L",
            (0, len(combine_np[1]) // 10),
            cv.FONT_HERSHEY_SIMPLEX,
            len(combine_np) // 1000,
            colors[0],
            2,
            cv.LINE_AA,
        )
        cv.putText(
            combine_np,
            "R",
            (len(combine_np[1]) // 2, len(combine_np[0]) // 10),
            cv.FONT_HERSHEY_SIMPLEX,
            len(combine_np) // 1000,
            colors[1],
            2,
            cv.LINE_AA,
        )
        return xp.asarray(combine_np)
