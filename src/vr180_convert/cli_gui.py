"""GUI utilities for vr180-convert CLI."""

from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import Any

import cv2 as cv
from numpy.typing import NDArray

LOG = getLogger(__name__)


def get_position_gui(
    image_paths: Sequence[Path | NDArray[Any]],
) -> list[tuple[int, int]]:
    """
    Open a full-screen GUI for clicking corresponding points on images.

    Parameters
    ----------
    image_paths : Sequence[Path | NDArray[Any]]
        Images to display sequentially for point selection.

    Returns
    -------
    list[tuple[int, int]]
        (x, y) coordinates of each clicked point.

    """
    images = [cv.imread(str(p)) if isinstance(p, Path) else p for p in image_paths]
    window = "Select position (click corresponding points)"
    cv.namedWindow(window, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    points: list[tuple[int, int]] = []
    current = 0

    def _on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            points.append((x, y))
            LOG.info("Point %d on image %d: (%d, %d)", len(points), current + 1, x, y)

    cv.setMouseCallback(window, _on_mouse)
    cv.imshow(window, images[current])

    while True:
        cv.waitKey(10)
        if len(points) > current:
            current += 1
            if current >= len(images):
                break
            cv.imshow(window, images[current])

    cv.destroyAllWindows()
    return points
