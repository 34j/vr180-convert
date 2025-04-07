from typing import Any

import cv2 as cv
import ivy
from ivy import Array

from .base import TransformerBase


class Merger(TransformerBase):
    def transform(self, x: Array, /, **kwargs: Any) -> Array:
        colors = [(0, 128, 255), (255, 128, 0)]
        combine = ivy.mean(x[0], axis=-1)[..., None] * ivy.array(colors[0]).reshape(
            [1] * (x[0].ndim - 1) + [3]
        ) + (
            ivy.mean(x[1], axis=-1)[..., None]
            * ivy.array(colors[1]).reshape([1] * (x[1].ndim - 1) + [3])
        )
        combine /= 255
        cv.putText(
            combine,
            "L",
            (0, len(combine[1]) // 10),
            cv.FONT_HERSHEY_SIMPLEX,
            len(combine) // 1000,
            colors[0],
            2,
            cv.LINE_AA,
        )
        cv.putText(
            combine,
            "R",
            (len(combine[1]) // 2, len(combine[0]) // 10),
            cv.FONT_HERSHEY_SIMPLEX,
            len(combine) // 1000,
            colors[1],
            2,
            cv.LINE_AA,
        )
        return combine
