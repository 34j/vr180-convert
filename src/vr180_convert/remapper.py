from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Literal, Sequence

import attrs
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin


from .transformer import DenormalizeTransformer, NormalizeTransformer, TransformerBase, get_radius


def get_map(
    transformer: TransformerBase,
    *,
    radius: float,
    size_input: tuple[int, int],
    size_output: tuple[int, int] = (2048, 2048),
):
    xmap, ymap = np.meshgrid(np.arange(size_output[0]), np.arange(size_output[1]))
    xmap, ymap = (
        NormalizeTransformer()
        * transformer
        * DenormalizeTransformer(scale=(radius, radius), center=(size_input[0] // 2, size_input[1] // 2))
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
    boarder_value: int | tuple[int, int, int] = (0, 33, 0),
    radius: float | Literal["auto", "max"] = "auto",
) -> Sequence[NDArray[np.uint8]]:
    # note that str is Sequence
    in_paths_ = [in_paths] if isinstance(in_paths, (str, Path)) else in_paths
    out_paths_ = [out_paths] if isinstance(out_paths, (str, Path)) else out_paths
    del in_paths, out_paths
    
    images = [cv.imread(Path(from_path).as_posix()) for from_path in in_paths_]
    if radius == "auto":
        radius_candidates = [get_radius(image) for image in images]
        radius_ = max(radius_candidates)
    elif radius == "max":
        radius_ = max(images[0].shape[0] / 2, images[0].shape[1] / 2)
    else:
        radius_ = radius
    
    xmap, ymap = get_map(radius=radius_, transformer=transformer, size_output=size_output, size_input=(images[0].shape[0], images[0].shape[1]))
    
    images = [ cv.remap(
        img,
        xmap,
        ymap,
        interpolation=interpolation,
        borderMode=boarder_mode,
        borderValue=boarder_value,
    ) for img in images
    ]
    
    if out_paths_:
        for to_path, image in zip(out_paths_, images):
            cv.imwrite(Path(to_path).as_posix(), image)
    return images

def apply_lr(
    transformer: TransformerBase,
    *,
    left_path: Path | str,
    right_path: Path | str,
    out_path: Path | str,
    size: tuple[int, int] = (2048, 2048),
    interpolation: int = cv.INTER_LANCZOS4,
    boarder_mode: int = cv.BORDER_CONSTANT,
    boarder_value: int | tuple[int, int, int] = (0, 33, 0),
    radius: float | Literal["auto", "max"] = "auto",
) -> None:
    images = apply(
        in_paths=[left_path, right_path],
        out_paths=None,
transformer=        transformer,
        size_output=size,
        interpolation=interpolation,
        boarder_mode=boarder_mode,
        boarder_value=boarder_value,
        radius=radius,
    )
    conbine = np.concatenate(images, axis=1)
    cv.imwrite(Path(out_path).as_posix(), conbine)
    