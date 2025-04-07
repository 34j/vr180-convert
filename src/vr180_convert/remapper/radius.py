from typing import Any, Literal, Sequence

import attrs
import ivy
from ivy import Array

from .base import RemapperBase
from .denormalize import DenormalizeRemapper


def _get_radius(input: Array, *, threshold: int = 10) -> float:
    """
    Estimate the radius of the circle in the image.

    Parameters
    ----------
    input : Array
        The input image of shape (height, width, 3).
    threshold : int, optional
        The threshold to determine if a pixel is black, by default 10

    Returns
    -------
    float
        The estimated radius.

    """
    height, width = input.shape[:2]
    if width > height:
        center_row = input[height // 2, :, :]
    else:
        center_row = input[:, width // 2, :]
    del height, width

    # determine if a pixel is black
    center_row_is_black = ivy.mean(center_row, axis=-1) < threshold
    center_row_is_black_deriv = ivy.diff(center_row_is_black.astype(int))

    # first and last 1 in the derivative
    center_row_black_start = ivy.argmax(center_row_is_black_deriv == 1)
    center_row_black_end = ivy.argmax(ivy.flip(center_row_is_black_deriv == -1))
    radius = (center_row_black_end - center_row_black_start) / 2
    return radius


def _get_radius_smart(
    radius: float | Literal["auto", "max"],
    images: Sequence[Array],
) -> float:
    """
    Get radius smartly.

    Parameters
    ----------
    radius : float | Literal[&quot;auto&quot;, &quot;max&quot;]
        The strategy to get the radius.
    images : Sequence[Array]
        Images to be processed.

    Returns
    -------
    float
        The radius.

    """
    if radius == "auto":
        radius_candidates = [_get_radius(image) for image in images]
        radius_ = max(radius_candidates)
    elif radius == "max":
        radius_ = min(images[0].shape[0] / 2, images[0].shape[1] / 2)
    else:
        radius_ = radius
    return radius_


@attrs.define(kw_only=True)
class AutoDenormalizeRemapper(RemapperBase):
    strategy: Literal["auto", "max"] | float
    child: DenormalizeRemapper | None = None
    requires_image: bool = True

    def remap(self, x: Array, y: Array, /, **kwargs: Any) -> tuple[Array, Array]:
        image = kwargs.pop("image")
        if image is None:
            raise ValueError("Image must be provided.")
        radius = _get_radius_smart(self.strategy, [image])
        self.child = DenormalizeRemapper(
            scale=(radius, radius), center=(image.shape[0] // 2, image.shape[1] // 2)
        )
        return self.child.remap(x, y, **kwargs)

    def inverse_remap(
        self, x: Array, y: Array, /, **kwargs: Any
    ) -> tuple[Array, Array]:
        if self.child is None:
            raise ValueError("Remapper has not been called yet.")
        return self.child.inverse_remap(x, y, **kwargs)
