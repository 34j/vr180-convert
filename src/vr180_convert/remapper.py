from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Literal, Sequence

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .transformer import (
    DenormalizeTransformer,
    NormalizeTransformer,
    TransformerBase,
    get_radius,
)

LOG = getLogger(__name__)


def get_map(
    transformer: TransformerBase,
    *,
    radius: float,
    size_input: tuple[int, int],
    size_output: tuple[int, int] = (2048, 2048),
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Generate remap map.

    Parameters
    ----------
    transformer : TransformerBase
        Transformer to be applied.
    radius : float
        Radius of the fisheye image.
    size_input : tuple[int, int]
        Size of the input image.
    size_output : tuple[int, int], optional
        Size of the output image, by default (2048, 2048)

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        xmap and ymap.

    """
    xmap, ymap = np.meshgrid(np.arange(size_output[0]), np.arange(size_output[1]))
    xmap, ymap = (
        NormalizeTransformer()
        * transformer
        * DenormalizeTransformer(
            scale=(radius, radius), center=(size_input[1] // 2, size_input[0] // 2)
        )
    ).transform(xmap, ymap)
    xmap, ymap = xmap.astype(np.float32), ymap.astype(np.float32)
    return xmap, ymap


def apply(
    transformer: TransformerBase,
    *,
    in_paths: Sequence[Path | str] | Path | str,
    out_paths: Sequence[Path | str] | None | Path | str = None,
    size_output: tuple[int, int] = (2048, 2048),
    interpolation: int = cv.INTER_LANCZOS4,
    boarder_mode: int = cv.BORDER_CONSTANT,
    boarder_value: int | tuple[int, int, int] = 0,
    radius: float | Literal["auto", "max"] = "auto",
) -> Sequence[NDArray[np.uint8]]:
    """
    Apply transformer to images.

    Parameters
    ----------
    transformer : TransformerBase
        Transformer to be applied.
    in_paths : Sequence[Path  |  str] | Path | str
        Input image paths.
    out_paths : Sequence[Path  |  str] | None | Path | str, optional
        Output image paths, by default None
        If None, do not save the images.
    size_output : tuple[int, int], optional
        Size of the output image, by default (2048, 2048)
    interpolation : int, optional
        Interpolation method for opencv, by default cv.INTER_LANCZOS4
    boarder_mode : int, optional
        Boarder mode for opencv, by default cv.BORDER_CONSTANT
    boarder_value : int | tuple[int, int, int], optional
        Boarder value for opencv, by default 0
    radius : float | Literal[&quot;auto&quot;, &quot;max&quot;], optional
        Radius of the fisheye image, by default &quot;auto&quot;

    Returns
    -------
    Sequence[NDArray[np.uint8]]
        Images after transformation.

    """
    # note that str is Sequence
    in_paths_ = [in_paths] if isinstance(in_paths, (str, Path)) else in_paths
    out_paths_ = [out_paths] if isinstance(out_paths, (str, Path)) else out_paths
    del in_paths, out_paths

    images = [cv.imread(Path(from_path).as_posix()) for from_path in in_paths_]
    if radius == "auto":
        radius_candidates = [get_radius(image) for image in images]
        radius_ = max(radius_candidates)
    elif radius == "max":
        radius_ = min(images[0].shape[0] / 2, images[0].shape[1] / 2)
    else:
        radius_ = radius
    LOG.info(f"Radius: {radius_}, strategy: {radius}, image shape: {images[0].shape}")

    xmap, ymap = get_map(
        radius=radius_,
        transformer=transformer,
        size_output=size_output,
        size_input=(images[0].shape[0], images[0].shape[1]),
    )

    images = [
        cv.remap(
            img,
            xmap,
            ymap,
            interpolation=interpolation,
            borderMode=boarder_mode,
            borderValue=boarder_value,
        )
        for img in images
    ]

    if out_paths_ is not None:
        for to_path, image in zip(out_paths_, images):
            cv.imwrite(Path(to_path).as_posix(), image)
    return images


def apply_lr(
    transformer: TransformerBase | tuple[TransformerBase, TransformerBase],
    *,
    left_path: Path | str,
    right_path: Path | str,
    out_path: Path | str,
    size_output: tuple[int, int] = (2048, 2048),
    interpolation: int = cv.INTER_LANCZOS4,
    boarder_mode: int = cv.BORDER_CONSTANT,
    boarder_value: int | tuple[int, int, int] = 0,
    radius: float | Literal["auto", "max"] = "auto",
    merge: bool = False,
) -> None:
    """
    Apply transformer to a pair of images.

    Parameters
    ----------
    transformer : TransformerBase
        Transformer to be applied.
    left_path : Path | str
        Left image path.
    right_path : Path | str
        Right image path.
    in_paths : Sequence[Path  |  str] | Path | str
        Input image paths.
    out_path : Path | str
        Output image path.
    size_output : tuple[int, int], optional
        Size of the output image, by default (2048, 2048)
    interpolation : int, optional
        Interpolation method for opencv, by default cv.INTER_LANCZOS4
    boarder_mode : int, optional
        Boarder mode for opencv, by default cv.BORDER_CONSTANT
    boarder_value : int | tuple[int, int, int], optional
        Boarder value for opencv, by default 0
    radius : float | Literal[&quot;auto&quot;, &quot;max&quot;], optional
        Radius of the fisheye image, by default &quot;auto&quot;
    merge : bool, optional
        Whether to merge the images mainly for calibration, by default False

    """
    images: Sequence[NDArray[np.uint8]]
    if isinstance(transformer, tuple):
        images = [
            apply(
                in_paths=in_path,
                out_paths=None,
                transformer=transformer,
                size_output=size_output,
                interpolation=interpolation,
                boarder_mode=boarder_mode,
                boarder_value=boarder_value,
                radius=radius,
            )[0]
            for transformer, in_path in zip(transformer, [left_path, right_path])
        ]
    else:
        images = apply(
            in_paths=[left_path, right_path],
            out_paths=None,
            transformer=transformer,
            size_output=size_output,
            interpolation=interpolation,
            boarder_mode=boarder_mode,
            boarder_value=boarder_value,
            radius=radius,
        )
    if merge:
        # https://en.wikipedia.org/wiki/Anaglyph_3D
        # 3d glass -> L: red filter, R: blue filter
        # L: blue layer, R: red layer
        colors = [(0, 0, 255), (255, 0, 0)]
        combine = np.mean(images[0], axis=-1)[..., None] * np.array(colors[0]).reshape(
            [1] * (images[0].ndim - 1) + [3]
        ) + (
            np.mean(images[1], axis=-1)[..., None]
            * np.array(colors[1]).reshape([1] * (images[1].ndim - 1) + [3])
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
    else:
        combine = np.concatenate(images, axis=1)
    cv.imwrite(Path(out_path).as_posix(), combine)
    LOG.info(f"Saved to {Path(out_path).absolute()}")
