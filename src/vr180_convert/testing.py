from __future__ import annotations

from colorsys import hls_to_rgb
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


def generate_test_image(
    size: int = 2048, path: str | Path | None = None
) -> NDArray[np.uint8]:
    """
    Generate a test image.

    Parameters
    ----------
    size : int, optional
        The height and width of the image, by default 2048
    path : str | Path | None, optional
        The path to save the image, by default None
        If None, the image will not be saved.

    Returns
    -------
    NDArray[np.uint8]
        The generated image.

    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    scale = size // 512 + 1
    # draw circle
    for radius in np.linspace(0, center, 10, endpoint=True):
        color = hls_to_rgb(radius / center, 0.5, 1)
        color = tuple(int(c * 255) for c in color)  # type: ignore
        cv.circle(img, (center, center), int(radius), color, scale)
        for angle in np.linspace(0, np.pi * 2, 4, endpoint=False):
            cv.putText(
                img,
                f"{radius / center * 90:g}",
                (
                    int(center + np.cos(angle) * radius),
                    int(center + np.sin(angle) * radius),
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                scale // 2,
                color,
                scale // 2,
            )
    # draw line
    for angle in np.linspace(0, np.pi * 2, 24, endpoint=False):
        color = hls_to_rgb(angle / (np.pi * 2), 0.5, 1)
        color = tuple(int(c * 255) for c in color)  # type: ignore
        x = center + np.cos(angle) * center
        y = center + np.sin(angle) * center
        cv.line(img, (center, center), (int(x), int(y)), color, scale)
    if path:
        cv.imwrite(Path(path).as_posix(), img)
    return img
